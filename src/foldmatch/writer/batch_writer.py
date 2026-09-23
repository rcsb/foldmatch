import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from abc import abstractmethod
from collections import deque

from lightning.pytorch.callbacks import BasePredictionWriter


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
        # A module with a 3Di head returns a third element; these writers store the
        # embedding only, so it is unpacked and dropped rather than silently changing
        # what they write.
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
    def __init__(
            self,
            output_path,
            df_id,
            write_interval="batch",
            buffer_size=16384,
            with_three_di=False,
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.df_id = df_id
        self.buffer_size = buffer_size
        self.with_three_di = with_three_di
        self._writer = None
        self._id_buffer = []
        self._emb_buffer = []
        self._three_di_buffer = []
        # The 3Di column is opt-in and decided here rather than per batch: a schema that
        # appeared partway through a run would split the shard into incompatible row
        # groups, and every reader of an existing database keeps the two-column file.
        fields = [('id', pa.string()), ('embedding', pa.list_(pa.float32()))]
        if with_three_di:
            fields.append(('three_di', pa.binary()))
        self._schema = pa.schema(fields)

    def _get_writer(self, rank):
        if self._writer is None:
            path = f"{self.out_path}/{self.df_id}-{rank}.parquet"
            self._writer = pq.ParquetWriter(path, self._schema, compression='snappy')
        return self._writer

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
        embeddings, dom_ids = prediction[0], prediction[1]
        three_di = prediction[2] if len(prediction) > 2 else None
        if self.with_three_di and three_di is None:
            raise ValueError(
                "this writer was built with with_three_di=True but the module returned "
                "no 3Di codes; the shard would be written with an empty column")
        self._id_buffer.extend(dom_ids)
        # Keep flattened float32 arrays (not Python lists) in the buffer — far
        # cheaper in memory until we flush. .flatten() copies, so the buffer
        # never aliases tensors that Lightning may reuse on the next batch.
        self._emb_buffer.extend(
            emb.detach().to('cpu', torch.float32).numpy().flatten()
            for emb in embeddings
        )
        if self.with_three_di:
            # One byte per residue, stored as opaque bytes: the codes are only meaningful
            # against the kernel that defines them, which the database manifest records.
            self._three_di_buffer.extend(codes.tobytes() for codes in three_di)
        if len(self._id_buffer) >= self.buffer_size:
            self._flush(trainer.global_rank)

    def _flush(self, rank):
        """Write the buffered rows as a single Parquet row group."""
        if not self._id_buffer:
            return
        columns = {
            'id': self._id_buffer,
            'embedding': pa.array(self._emb_buffer, type=pa.list_(pa.float32())),
        }
        if self.with_three_di:
            columns['three_di'] = pa.array(self._three_di_buffer, type=pa.binary())
        table = pa.table(columns, schema=self._schema)
        self._get_writer(rank).write_table(table)
        self._id_buffer.clear()
        self._emb_buffer.clear()
        self._three_di_buffer.clear()

    def on_predict_end(self, trainer, pl_module):
        # Flush whatever didn't reach a full buffer before closing.
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()
