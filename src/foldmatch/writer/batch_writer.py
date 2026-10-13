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
    def __init__(
            self,
            output_path,
            df_id,
            write_interval="batch",
            buffer_size=16384,
    ):
        super().__init__(write_interval)
        self.out_path = output_path
        self.df_id = df_id
        self.buffer_size = buffer_size
        self._writer = None
        self._id_buffer = []
        self._emb_buffer = []
        self._schema = pa.schema([
            ('id', pa.string()),
            ('embedding', pa.list_(pa.float32()))
        ])

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
        # Extra tuple elements beyond (embeddings, ids) are dropped: this
        # writer's schema is fixed at construction to id + embedding.
        embeddings, dom_ids = prediction[0], prediction[1]
        self._id_buffer.extend(dom_ids)
        # Keep flattened float32 arrays (not Python lists) in the buffer — far
        # cheaper in memory until we flush. .flatten() copies, so the buffer
        # never aliases tensors that Lightning may reuse on the next batch.
        self._emb_buffer.extend(
            emb.detach().to('cpu', torch.float32).numpy().flatten()
            for emb in embeddings
        )
        if len(self._id_buffer) >= self.buffer_size:
            self._flush(trainer.global_rank)

    def _flush(self, rank):
        """Write the buffered rows as a single Parquet row group."""
        if not self._id_buffer:
            return
        table = pa.table(
            {
                'id': self._id_buffer,
                'embedding': pa.array(self._emb_buffer, type=pa.list_(pa.float32())),
            },
            schema=self._schema,
        )
        self._get_writer(rank).write_table(table)
        self._id_buffer.clear()
        self._emb_buffer.clear()

    def on_predict_end(self, trainer, pl_module):
        # Flush whatever didn't reach a full buffer before closing.
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()
