import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from abc import abstractmethod
from collections import deque

from lightning.pytorch.callbacks import BasePredictionWriter

from foldmatch.search.residue_shards import (
    SEQ_SHA1, SEQ_SHA1_BYTES, SEQ_SHA1_TYPE, check_residue_output, large_binary_array, pass_record, shard_schema,
)

#: Default byte budget of a writer with a residue column: buffered rows are written once they hold this many bytes.
DEFAULT_BYTE_BUDGET = 256 * 1024 * 1024


class CoreBatchWriter(BasePredictionWriter):
    def __init__(
            self,
            output_path,
            postfix,
            write_interval="batch"
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.postfix = postfix

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if prediction is None:
            return
        # A prediction is (embeddings, ids) optionally followed by per-item
        # payloads (3- and 4-tuples). These writers store the embedding only,
        # so any extra elements are dropped.
        embeddings, dom_ids = prediction[0], prediction[1]
        deque(map(
            self._write_embedding,
            embeddings,
            dom_ids
        ))

    def file_name(self, dom_id):
        return f'{self.out_path}/{dom_id}.{self.postfix}'

    @abstractmethod
    def _write_embedding(self, embedding, dom_id):
        pass


class CsvBatchWriter(CoreBatchWriter):
    def __init__(
            self,
            output_path,
            postfix="csv",
            write_interval="batch"
    ):
        super().__init__(output_path, postfix, write_interval)

    def _write_embedding(self, embedding, dom_id):
        pd.DataFrame(embedding.to('cpu').numpy()).to_csv(
            self.file_name(dom_id),
            index=False,
            header=False
        )


class TensorBatchWriter(CoreBatchWriter):
    def __init__(
            self,
            output_path,
            postfix="pt",
            write_interval="batch",
            device="cpu"
    ):
        super().__init__(output_path, postfix, write_interval)
        self.device = device

    def _write_embedding(self, embedding, dom_id):
        torch.save(
            embedding.to(self.device),
            self.file_name(dom_id)
        )


class DataFrameStorage(CoreBatchWriter):
    def __init__(
            self,
            output_path,
            df_id,
            postfix="pkl",
            write_interval="batch"
    ):
        super().__init__(output_path, postfix, write_interval)
        self.df_id = df_id
        self.embedding = pd.DataFrame(
            data={},
            columns=['id', 'embedding'],
        )

    def _write_embedding(self, embedding, dom_id):
        self.embedding = pd.concat([
            self.embedding,
            pd.DataFrame(
                data={'id': dom_id, 'embedding': [embedding.to('cpu').numpy()]},
                columns=['id', 'embedding'],
            )
        ], ignore_index=True)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank
        path = f"{self.out_path}/{self.df_id}-{rank}.pkl.gz"
        self.embedding.to_pickle(
            path,
            compression='gzip'
        )


class JsonStorage(DataFrameStorage):
    def __init__(
            self,
            output_path,
            df_id,
            postfix="json",
            write_interval="batch"
    ):
        super().__init__(output_path, df_id, postfix, write_interval)

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank
        path = f"{self.out_path}/{self.df_id}-{rank}.json.gz"
        self.embedding.to_json(
            path,
            orient='records',
            compression='gzip'
        )

class ParquetBatchWriter(BasePredictionWriter):
    """Per-rank Parquet shard ``{output_path}/{df_id}-{rank}.parquet``, written in row groups.

    Without ``residue_output`` the schema is the legacy ``id | embedding`` (no metadata) and a prediction's elements
    beyond ``(embeddings, ids)`` are dropped, so plain passes write the same bytes as before.

    With ``residue_output`` (``"residue_codes"`` for builds, ``"query_encoding"`` for queries) and the scoring
    ``system``, the schema, fixed here, adds the payload column (``large_binary``) and ``seq_sha1``
    (``fixed_size_binary(20)``), with the metadata ``foldmatch.residue_output``, ``foldmatch.system``,
    ``foldmatch.encoding_digest`` and ``foldmatch.system_digest`` (:mod:`foldmatch.search.residue_shards`), and the
    pass record ``foldmatch.pass`` (``min_res_n`` and ``esm_revision`` given here; the device type, its ESM3
    precision and the number of ranks added at the start of the pass, before any row is written). Every prediction
    must then be ``(embeddings, ids, payloads, seq_sha1s)``: a writer with a column refuses one without it, and
    refuses at the start of the pass a module that does not produce it.

    Buffered rows are written as one row group once ``buffer_size`` rows (16,384) or ``byte_budget`` bytes are
    buffered, whichever comes first. The byte budget defaults to 256 MB with a residue column (query encodings take
    128 B per residue) and to none without one, which keeps the plain row groups, and so the plain bytes, unchanged
    (chain embeddings take 6 KiB a row, so 16,384 rows stay below 256 MB anyway).
    """

    def __init__(
            self,
            output_path,
            df_id,
            write_interval="batch",
            buffer_size=16384,
            residue_output=None,
            system=None,
            byte_budget=None,
            min_res_n=None,
            esm_revision=None,
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.df_id = df_id
        self.buffer_size = buffer_size
        self.residue_output = check_residue_output(residue_output)
        if self.residue_output is None and (system is not None or min_res_n is not None or esm_revision is not None):
            raise ValueError("a scoring system and a pass record are only used by a writer with a residue output")
        if self.residue_output is not None and system is None:
            raise ValueError(f"a {self.residue_output} writer needs the scoring system (its schema metadata)")
        self.system = system
        if byte_budget is None and self.residue_output is not None:
            byte_budget = DEFAULT_BYTE_BUDGET
        if byte_budget is not None and int(byte_budget) <= 0:
            raise ValueError(f"byte_budget must be > 0, got {byte_budget!r}")
        self.byte_budget = None if byte_budget is None else int(byte_budget)
        self._writer = None
        self._id_buffer = []
        self._emb_buffer = []
        self._payload_buffer = []
        self._sha1_buffer = []
        self._buffered_bytes = 0
        # Fixed here (the pass record is completed in on_predict_start, before the first row group): every row
        # group of a rank has exactly this schema and metadata.
        self._pass = None if self.residue_output is None else {"min_res_n": min_res_n, "esm_revision": esm_revision}
        self._schema = shard_schema(self.residue_output, system,
                                    None if self._pass is None else pass_record(**self._pass))

    @property
    def schema(self):
        return self._schema

    def _get_writer(self, rank):
        if self._writer is None:
            path = f"{self.out_path}/{self.df_id}-{rank}.parquet"
            self._writer = pq.ParquetWriter(path, self._schema, compression='snappy')
        return self._writer

    def on_predict_start(self, trainer, pl_module):
        # Every rank has the same module and writer, so a mismatch is refused by all of them, before any batch.
        produced = getattr(pl_module, "residue_output", None)
        if self.residue_output is not None and produced != self.residue_output:
            raise ValueError(
                f"this Parquet writer stores {self.residue_output}, but the module produces "
                f"{produced or 'no residue output'}"
            )
        if self.residue_output is not None and self._writer is None:
            # The pass record's device and ranks, known now: the device the module runs on and the world size.
            device = getattr(getattr(pl_module, "device", None), "type", None)
            world_size = getattr(trainer, "world_size", None)
            self._schema = shard_schema(self.residue_output, self.system, pass_record(
                **self._pass, accelerator=device, world_size=None if world_size is None else int(world_size)))

    def _residue_columns(self, prediction, dom_ids):
        if len(prediction) < 4:
            raise ValueError(
                f"this Parquet writer stores {self.residue_output}: a prediction must be (embeddings, ids, payloads, "
                f"seq_sha1s), got a {len(prediction)}-tuple"
            )
        payloads, digests = prediction[2], prediction[3]
        if len(payloads) != len(dom_ids) or len(digests) != len(dom_ids):
            raise ValueError(
                f"{len(dom_ids)} ids but {len(payloads)} {self.residue_output} payloads and {len(digests)} seq_sha1s"
            )
        payloads = [bytes(p) for p in payloads]
        digests = [bytes(d) for d in digests]
        bad = [i for i, d in zip(dom_ids, digests) if len(d) != SEQ_SHA1_BYTES]
        if bad:
            raise ValueError(f"seq_sha1 must be {SEQ_SHA1_BYTES} bytes (e.g. for {bad[0]!r})")
        return payloads, digests

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if prediction is None:
            return
        # Without a residue column, extra tuple elements beyond (embeddings, ids)
        # are dropped: this writer's schema is fixed at construction.
        embeddings, dom_ids = prediction[0], prediction[1]
        if self.residue_output is not None:
            payloads, digests = self._residue_columns(prediction, dom_ids)
        self._id_buffer.extend(dom_ids)
        # Keep flattened float32 arrays (not Python lists) in the buffer — far
        # cheaper in memory until we flush. .flatten() copies, so the buffer
        # never aliases tensors that Lightning may reuse on the next batch.
        flat = [emb.detach().to('cpu', torch.float32).numpy().flatten() for emb in embeddings]
        self._emb_buffer.extend(flat)
        if self.byte_budget is not None:
            self._buffered_bytes += sum(a.nbytes for a in flat) + sum(len(i) for i in dom_ids)
        if self.residue_output is not None:
            self._payload_buffer.extend(payloads)
            self._sha1_buffer.extend(digests)
            self._buffered_bytes += sum(len(p) for p in payloads) + SEQ_SHA1_BYTES * len(digests)
        if len(self._id_buffer) >= self.buffer_size or (
                self.byte_budget is not None and self._buffered_bytes >= self.byte_budget):
            self._flush(trainer.global_rank)

    def _flush(self, rank):
        """Write the buffered rows as a single Parquet row group."""
        if not self._id_buffer:
            return
        columns = {
            'id': self._id_buffer,
            'embedding': pa.array(self._emb_buffer, type=pa.list_(pa.float32())),
        }
        if self.residue_output is not None:
            columns[self.residue_output] = large_binary_array(self._payload_buffer)
            columns[SEQ_SHA1] = pa.array(self._sha1_buffer, type=SEQ_SHA1_TYPE)
        table = pa.table(columns, schema=self._schema)
        self._get_writer(rank).write_table(table)
        self._id_buffer.clear()
        self._emb_buffer.clear()
        self._payload_buffer.clear()
        self._sha1_buffer.clear()
        self._buffered_bytes = 0

    def on_predict_end(self, trainer, pl_module):
        # Flush whatever didn't reach a full buffer before closing.
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()
