import numpy as np
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from abc import abstractmethod
from collections import deque

from lightning.pytorch.callbacks import BasePredictionWriter

from foldmatch.errors import RefusalError
from foldmatch.utils.output_files import name_key, name_problem, probe_folding
from foldmatch.search.residue_shards import (
    ID, RESIDUE_CODES, SEQ_SHA1, SEQ_SHA1_BYTES, SEQ_SHA1_TYPE, check_residue_output, code_table_schema,
    code_table_type, large_binary_array, pass_record, shard_schema,
)

#: Default byte budget of a writer with a residue column or residue embeddings: buffered rows are written once they
#: hold this many bytes.
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
        # The files this writer has written in this run, by the name the folder compares (claim_file)
        self._written = {}
        self._folding = None

    #: What this writer writes, in its refusals ("Each embedding of a run needs a file of its own", "Write the
    #: embeddings as parquet or json").
    _ITEM = "embedding"
    _ITEMS = "embeddings"

    def _what(self, dom_id) -> str:
        """What this writer writes of ``dom_id``, in its refusals."""
        return f"The embedding {dom_id!r}"

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

    def claim_file(self, dom_id) -> str:
        """The file of ``dom_id``, once this writer has checked that it may write it (P9.N D4): the name must make one
        file of the output folder (name_problem), and one that this writer has not written in this run, compared as
        the folder compares names (chains A and a of one structure are one file on a case-insensitive folder).
        Otherwise RefusalError, naming the file: the run stops, the first file kept and this one not written.

        A backstop for names known only during the run (the chains of structures read whole); the inference functions
        refuse the names they know before any model is built. Kept per writer, so per process: two ranks that write
        one file are not seen."""
        suffix = f".{self.postfix}"
        problem = name_problem(dom_id, suffix)
        if problem is not None:
            raise RefusalError(
                f"{self._what(dom_id)} cannot be written to {self.out_path}: its name {problem}, so it cannot name "
                f"the file <name>{suffix} there. Write the {self._ITEMS} as parquet or json, which keep each under its "
                f"id, or give it another name."
            )
        if self._folding is None:
            self._folding = probe_folding(self.out_path)
        key = name_key(f"{dom_id}{suffix}", self._folding)
        first = self._written.get(key)
        if first is not None:
            folded = self._folding.describe()
            note = f" (the folder does not tell {folded} apart)" if folded and first != dom_id else ""
            raise RefusalError(
                f"{self.file_name(first)} would be written twice by this run, for {first!r} and then for "
                f"{dom_id!r}{note}: the first is kept, {dom_id!r} is not written, and the run stops. Each {self._ITEM} "
                f"of a run needs a file of its own: write them as parquet or json, which keep each under its id, or, "
                f"for names that differ only in case, to a folder of a case-sensitive file system."
            )
        self._written[key] = dom_id
        return self.file_name(dom_id)

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
            self.claim_file(dom_id),
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
            self.claim_file(dom_id)
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

    Without ``residue_output`` the schema is the legacy ``id | embedding`` (``list<float32>``, no metadata) and a
    prediction's elements beyond ``(embeddings, ids)`` are dropped, so plain passes write the same bytes as before. A
    row holds one embedding vector (a chain, assembly or sequence embedding), so a tensor with more than one axis
    longer than 1 is refused.

    With ``residue_embeddings=True`` (``fm-embedding ... residue``, ``inference residue-embedding``) a row holds the
    residue embeddings of one chain, a 2-D ``[L, D]`` tensor (ESM3's BOS and EOS rows included), as one list of D
    values per residue: ``embedding`` is ``list<fixed_size_list<float32, D>>``, built from offsets, with no metadata.
    D is stored in the schema, fixed at the first row group (the writer opens lazily; :attr:`schema` is None until
    then), so every embedding must be 2-D and of one D. Not with ``residue_output``, whose ``embedding`` holds chain
    embeddings.

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
    128 B per residue) or with ``residue_embeddings`` (6 KiB per residue at D = 1536, so 16,384 chains would buffer
    tens of GB), and to none otherwise, which keeps the plain row groups, and so the plain bytes, unchanged (chain
    embeddings take 6 KiB a row, so 16,384 rows stay below 256 MB anyway).
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
            residue_embeddings=False,
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.df_id = df_id
        self.buffer_size = buffer_size
        self.residue_output = check_residue_output(residue_output)
        if residue_embeddings and self.residue_output is not None:
            raise ValueError(f"a {self.residue_output} writer stores chain embeddings, not residue embeddings")
        self.residue_embeddings = bool(residue_embeddings)
        if self.residue_output is None and (system is not None or min_res_n is not None or esm_revision is not None):
            raise ValueError("a scoring system and a pass record are only used by a writer with a residue output")
        if self.residue_output is not None and system is None:
            raise ValueError(f"a {self.residue_output} writer needs the scoring system (its schema metadata)")
        self.system = system
        if byte_budget is None and (self.residue_output is not None or self.residue_embeddings):
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
        # Fixed here (the pass record is completed in on_predict_start, before the first row group; residue
        # embeddings at the first row group, once D is known): every row group of a rank has exactly this schema and
        # metadata.
        self._pass = None if self.residue_output is None else {"min_res_n": min_res_n, "esm_revision": esm_revision}
        self._schema = None if self.residue_embeddings else shard_schema(
            self.residue_output, system, None if self._pass is None else pass_record(**self._pass))
        # Residue embeddings: D, set by the first table (its id kept for the refusal of another D).
        self._residue_dim = None
        self._residue_dim_id = None

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

    def _check_shapes(self, embeddings, dom_ids):
        """Refuse an embedding this writer cannot store, before any of the batch is buffered: a row holds one vector
        (a tensor with at most one axis longer than 1) or, with ``residue_embeddings``, one ``[L, D]`` table (D > 0),
        of the first table's D."""
        for emb, dom_id in zip(embeddings, dom_ids):
            shape = list(emb.shape)
            if not self.residue_embeddings:
                if sum(n > 1 for n in shape) > 1:
                    raise RefusalError(
                        f"this Parquet writer stores one embedding vector per row, but {dom_id} has shape {shape}: "
                        f"residue embeddings are written with residue_embeddings=True"
                    )
            elif len(shape) != 2 or shape[1] == 0:
                raise RefusalError(
                    f"this Parquet writer stores residue embeddings, one [L, D] table per row, but {dom_id} has shape "
                    f"{shape}"
                )
            elif self._residue_dim is None:
                self._residue_dim, self._residue_dim_id = shape[1], dom_id
            elif shape[1] != self._residue_dim:
                raise RefusalError(
                    f"this Parquet writer stores residue embeddings of one dimension, but {dom_id} has {shape[1]} "
                    f"values per residue and {self._residue_dim_id} has {self._residue_dim}"
                )

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
        # are dropped: this writer's columns are fixed at construction.
        embeddings, dom_ids = prediction[0], prediction[1]
        if self.residue_output is not None:
            payloads, digests = self._residue_columns(prediction, dom_ids)
        self._check_shapes(embeddings, dom_ids)
        self._id_buffer.extend(dom_ids)
        # Keep flattened float32 arrays (not Python lists) in the buffer — far
        # cheaper in memory until we flush. .flatten() copies, so the buffer
        # never aliases tensors that Lightning may reuse on the next batch.
        # A residue table is kept flattened too: _flush counts its rows.
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
        if self.residue_embeddings:
            embedding = self._residue_embedding_array()
        else:
            embedding = pa.array(self._emb_buffer, type=pa.list_(pa.float32()))
        columns = {
            'id': self._id_buffer,
            'embedding': embedding,
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

    def _residue_embedding_array(self):
        """The buffered residue tables as ``list<fixed_size_list<float32, D>>``: one list of D values per residue,
        grouped into rows by offsets. The first call fixes the schema, with D."""
        dim = self._residue_dim
        if self._schema is None:
            self._schema = pa.schema([('id', pa.string()), ('embedding', pa.list_(pa.list_(pa.float32(), dim)))])
        offsets = np.concatenate(([0], np.cumsum([flat.size // dim for flat in self._emb_buffer])))
        residues = pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(self._emb_buffer)), dim)
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), residues)

    def on_predict_end(self, trainer, pl_module):
        # Flush whatever didn't reach a full buffer before closing.
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()


# =====================================================================================================================
# Residue codes in their own folder (fm-embedding from-sequences residue|chain --output-codes-folder)
# =====================================================================================================================
def residue_code_tables(prediction, code_bytes: int = 8):
    """The residue codes of a prediction ``(embeddings, ids, payloads, seq_sha1s)`` of a module with the
    ``residue_codes`` output: ``(ids, tables, seq_sha1s)``, one writable ``uint8 [L, code_bytes]`` array per id (``[0,
    code_bytes]`` for an empty payload, a protein whose codes could not be computed). ValueError for a prediction
    without the payloads, or a payload that is not whole codes."""
    if prediction is None or len(prediction) < 4:
        raise ValueError(
            f"a residue-code writer needs predictions (embeddings, ids, payloads, seq_sha1s), got "
            f"{'none' if prediction is None else f'a {len(prediction)}-tuple'}: the module computes no residue codes"
        )
    ids, payloads, digests = tuple(prediction[1]), prediction[2], prediction[3]
    if len(payloads) != len(ids) or len(digests) != len(ids):
        raise ValueError(f"{len(ids)} ids but {len(payloads)} residue-code payloads and {len(digests)} seq_sha1s")
    tables = []
    for dom_id, payload in zip(ids, payloads):
        payload = bytes(payload)
        if len(payload) % code_bytes:
            raise ValueError(f"the residue codes of {dom_id!r} are {len(payload)} bytes, not a whole number of "
                             f"{code_bytes}-byte codes")
        tables.append(np.frombuffer(payload, dtype=np.uint8).reshape(-1, code_bytes).copy())
    digests = [bytes(d) for d in digests]
    bad = [i for i, d in zip(ids, digests) if len(d) != SEQ_SHA1_BYTES]
    if bad:
        raise ValueError(f"seq_sha1 must be {SEQ_SHA1_BYTES} bytes (e.g. for {bad[0]!r})")
    return ids, tables, digests


def _check_codes_module(pl_module):
    """Refuse, at the start of the pass and so on every rank, a module that does not compute residue codes."""
    produced = getattr(pl_module, "residue_output", None)
    if produced != RESIDUE_CODES:
        raise ValueError(f"this writer stores {RESIDUE_CODES}, but the module produces "
                         f"{produced or 'no residue output'}")


class _CodeFiles:
    """The residue codes of each id to its own file, ``<codes folder>/<id>.<format>`` (as the embedding of the id is
    ``<output folder>/<id>.<format>``), by the item writer it is mixed into (csv or pt), which writes each ``uint8 [L,
    8]`` table as it writes an embedding."""

    _ITEM = "residue-code table"
    _ITEMS = "residue codes"

    def _what(self, dom_id) -> str:
        return f"The residue codes of {dom_id!r}"

    def on_predict_start(self, trainer, pl_module):
        _check_codes_module(pl_module)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if prediction is None:
            return
        ids, tables, _ = residue_code_tables(prediction)
        deque(map(self._write_embedding, (torch.from_numpy(t) for t in tables), ids))


class CsvCodeWriter(_CodeFiles, CsvBatchWriter):
    """``<output_path>/<id>.csv``: an id's residue codes, L lines of 8 integers (0-255), no header or index (an empty
    file for a protein without codes)."""

    def __init__(self, output_path, write_interval="batch"):
        super().__init__(output_path, postfix="csv", write_interval=write_interval)


class TensorCodeWriter(_CodeFiles, TensorBatchWriter):
    """``<output_path>/<id>.pt``: an id's residue codes, a CPU ``uint8`` tensor ``[L, 8]``."""

    def __init__(self, output_path, write_interval="batch"):
        super().__init__(output_path, postfix="pt", write_interval=write_interval, device="cpu")


class JsonCodeStorage(BasePredictionWriter):
    """``<output_path>/<codes_name>-<rank>.json.gz``: one record ``{"id", "residue_codes"}`` per id, its residue codes
    a list of 8 integers per residue, written at the end of the pass as the embeddings' JSON (orient records, gzip)."""

    def __init__(self, output_path, codes_name, write_interval="batch"):
        super().__init__(write_interval)
        self.out_path = output_path
        self.codes_name = codes_name
        self._ids = []
        self._tables = []

    def on_predict_start(self, trainer, pl_module):
        _check_codes_module(pl_module)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if prediction is None:
            return
        ids, tables, _ = residue_code_tables(prediction)
        self._ids.extend(ids)
        self._tables.extend(tables)

    def on_predict_end(self, trainer, pl_module):
        codes = np.empty(len(self._tables), dtype=object)
        for k, table in enumerate(self._tables):
            codes[k] = table
        pd.DataFrame({'id': pd.Series(self._ids, dtype=object), RESIDUE_CODES: codes}).to_json(
            f"{self.out_path}/{self.codes_name}-{trainer.global_rank}.json.gz",
            orient='records',
            compression='gzip'
        )


class ParquetCodeWriter(BasePredictionWriter):
    """Per-rank ``<output_path>/<codes_name>-<rank>.parquet`` of residue codes, written in row groups: columns ``id |
    residue_codes | seq_sha1`` (``residue_codes`` one ``uint8 [L, 8]`` table per sequence, as
    ``list<fixed_size_list<uint8, 8>>``), with the scoring system's name and digests and the pass record in the
    schema metadata (:func:`foldmatch.search.residue_shards.code_table_schema`: not an embedding shard, which every
    shard reader refuses). The pass record holds ``min_res_n`` and ``esm_revision`` given here, and the device type,
    its ESM3 precision and the number of ranks, added at the start of the pass. Rows are buffered up to ``buffer_size``
    rows or ``byte_budget`` bytes of codes. No file for a rank without rows, as for the embeddings."""

    def __init__(self, output_path, codes_name, system, *, min_res_n=None, esm_revision=None, write_interval="batch",
                 buffer_size=16384, byte_budget=DEFAULT_BYTE_BUDGET):
        super().__init__(write_interval)
        if system is None:
            raise ValueError("a residue-code writer needs the scoring system (its schema metadata)")
        if int(byte_budget) <= 0:
            raise ValueError(f"byte_budget must be > 0, got {byte_budget!r}")
        self.out_path = output_path
        self.codes_name = codes_name
        self.system = system
        self.buffer_size = buffer_size
        self.byte_budget = int(byte_budget)
        self._pass = {"min_res_n": min_res_n, "esm_revision": esm_revision}
        self._schema = code_table_schema(system, pass_record(**self._pass))
        self._writer = None
        self._ids, self._tables, self._digests = [], [], []
        self._buffered_bytes = 0

    @property
    def schema(self):
        return self._schema

    def on_predict_start(self, trainer, pl_module):
        _check_codes_module(pl_module)
        if self._writer is None:
            device = getattr(getattr(pl_module, "device", None), "type", None)
            world_size = getattr(trainer, "world_size", None)
            self._schema = code_table_schema(self.system, pass_record(
                **self._pass, accelerator=device, world_size=None if world_size is None else int(world_size)))

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if prediction is None:
            return
        ids, tables, digests = residue_code_tables(prediction, self.system.code_bytes)
        self._ids.extend(ids)
        self._tables.extend(tables)
        self._digests.extend(digests)
        self._buffered_bytes += sum(t.nbytes for t in tables)
        if len(self._ids) >= self.buffer_size or self._buffered_bytes >= self.byte_budget:
            self._flush(trainer.global_rank)

    def _flush(self, rank):
        if not self._ids:
            return
        width = self.system.code_bytes
        offsets = np.concatenate(([0], np.cumsum([len(t) for t in self._tables]))).astype(np.int32)
        values = np.concatenate(self._tables).reshape(-1) if self._tables else np.empty(0, dtype=np.uint8)
        residues = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.uint8()), width)
        codes = pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), residues)
        table = pa.table({
            ID: pa.array(self._ids, type=pa.string()),
            RESIDUE_CODES: codes.cast(code_table_type(self.system)),
            SEQ_SHA1: pa.array(self._digests, type=SEQ_SHA1_TYPE),
        }, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(f"{self.out_path}/{self.codes_name}-{rank}.parquet", self._schema,
                                            compression='snappy')
        self._writer.write_table(table)
        self._ids.clear()
        self._tables.clear()
        self._digests.clear()
        self._buffered_bytes = 0

    def on_predict_end(self, trainer, pl_module):
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()


def residue_code_writer(out_format, codes_folder, codes_name, system, *, min_res_n=None, esm_revision=None):
    """The writer of the residue codes of an fm-embedding pass (``--output-codes-folder``) in ``out_format``, to
    ``codes_folder``: ``<codes_name>-<rank>.parquet`` or ``.json.gz``, else ``<id>.csv`` or ``<id>.pt`` (which
    ``codes_name`` does not name)."""
    fmt = getattr(out_format, "value", out_format)
    if fmt == "parquet":
        return ParquetCodeWriter(codes_folder, codes_name, system, min_res_n=min_res_n, esm_revision=esm_revision)
    if fmt == "json":
        return JsonCodeStorage(codes_folder, codes_name)
    return TensorCodeWriter(codes_folder) if fmt == "pt" else CsvCodeWriter(codes_folder)
