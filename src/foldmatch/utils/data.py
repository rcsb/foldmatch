import os
import requests
import gzip
import pandas as pd
import torch
import logging

import typer
from requests import RequestException, ConnectTimeout
from io import StringIO, BytesIO

import numpy as np
import pyarrow.parquet as pq

from foldmatch.types.api_types import ResEmbeddingFormat, StructureFormat
from typing import Optional, Iterator, List, Tuple
from pathlib import Path


def load_residue_embedding(
        file_path: Path,
        res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt
):
    """Load a residue-level embedding tensor from a .pt or .csv file."""
    if res_embedding_format == ResEmbeddingFormat.csv:
        values = pd.read_csv(file_path, header=None, index_col=None).values
        return torch.from_numpy(values).float()
    return torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)

def collate_seq_embeddings(
        batch_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads the tensors in a batch to the same size.

    Args:
        batch_list (list of torch.Tensor): A list of samples, where each sample is a tensor of shape (sequence_length, embedding_dim).

    Returns:
        tuple: A tuple containing:
            - padded_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length, embedding_dim), where each sample is padded to the max sequence length.
            - mask_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length) where padded positions are marked as False.
    """
    if batch_list[0] is None:
        return None
    device = batch_list[0].device  # Get the device of the input tensors
    max_len = max(sample.size(0) for sample in batch_list)  # Determine the maximum sequence length
    dim = batch_list[0].size(1)  # Determine the embedding dimension
    batch_size = len(batch_list)  # Determine the batch size

    # Initialize tensors for the padded batch and masks on the same device as the input tensors
    padded_batch = torch.zeros((batch_size, max_len, dim), dtype=batch_list[0].dtype, device=device)
    mask_batch = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)

    for i, sample in enumerate(batch_list):
        seq_len = sample.size(0)  # Get the length of the current sequence
        padded_batch[i, :seq_len] = sample  # Pad the sequence with zeros
        mask_batch[i, :seq_len] = False  # Set mask positions for the actual data to False

    return padded_batch, mask_batch

def identity_collate(batch):
    """Identity collate function for DataLoader."""
    return batch

def collate_embeddings(emb):
    return (
        collate_seq_embeddings([x for x, z in emb]),
        tuple([z for x, z in emb])
    )

def stringio_from_url(url):
    return run_with_retries(
        __stringio_from_url,
        url,
        retries=5,
        delay=60,
        backoff=2,
        exceptions=(RequestException, IOError)
    )

def __stringio_from_url(url):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.content
        if url.endswith('.bcif.gz'):
            with gzip.GzipFile(fileobj=BytesIO(data), mode='rb') as gz:
                decompressed_data = gz.read()
                return BytesIO(decompressed_data)
        if url.endswith('.gz'):
            compressed = BytesIO(data)
            with gzip.open(compressed, 'rt') as f:
                return StringIO(f.read())
        else:
            return StringIO(response.text)
    except (RequestException, ConnectTimeout) as e:
        raise RequestException(f"Error fetching URL {url}: {e}")
    except (OSError, gzip.BadGzipFile) as e:
        raise IOError(f"Error decompressing gzip file {url}: {e}")


def concatenate_tensors(file_list, max_residues, dim=0, res_embedding_format=ResEmbeddingFormat.pt):
    """
    Concatenates a list of tensors stored in individual files along a specified dimension.

    Args:
        file_list (list of str): List of file paths to tensor files.
        max_residues (int): Maximum number of residues allowed in the assembly
        dim (int): The dimension along which to concatenate the tensors. Default is 0.
        res_embedding_format (ResEmbeddingFormat): Format of the residue embedding files (pt or csv).

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    tensors = []
    total_residues = 0
    for file in file_list:
        try:
            tensor = load_residue_embedding(file, res_embedding_format=res_embedding_format)
            total_residues += tensor.shape[0]
            tensors.append(tensor)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error loading tensor from {file}: {e}")
            continue
        if total_residues > max_residues:
            break
    if tensors and len(tensors) > 0:
        tensor_cat = torch.cat(tensors, dim=dim)
        return tensor_cat
    else:
        raise ValueError(f"No valid tensors were loaded to concatenate. {', '.join(file_list)}")

def adapt_csv_to_embedding_chain_stream(src_file, res_embedding_location, res_embedding_format=ResEmbeddingFormat.pt):
    def __parse_row(row):
        r = row.split(",")
        return os.path.join(res_embedding_location, f"{r[0]}.{r[2]}.{res_embedding_format.value}"), f"{r[0]}.{r[2]}"
    return tuple([__parse_row(r.strip()) for r in open(src_file) if len(r.split(",")) > 2])


# Generated by ChatGPT 5.1
import time
from typing import Callable, Iterable, Type, Any
def run_with_retries(
        func: Callable,
        *args,
        retries: int = 3,
        delay: float = 0.5,
        backoff: float = 1.0,
        exceptions: Iterable[Type[BaseException]] = (Exception,),
        **kwargs: Any
) -> Any:
    """
    Execute a function with automatic retries upon exception.

    Parameters
    ----------
    func : Callable
        The function to be executed.
    *args :
        Positional arguments forwarded to the function.
    retries : int, optional
        Maximum number of attempts. Default is 3.
    delay : float, optional
        Initial delay (seconds) before retrying. Default is 0.5.
    backoff : float, optional
        Multiplicative factor for exponential backoff.
        If 1.0, the delay remains constant.
    exceptions : iterable of Exception types, optional
        Exception types that should trigger a retry.
    **kwargs :
        Keyword arguments forwarded to the function.

    Returns
    -------
    Any
        The return value of the function if it succeeds.

    Raises
    ------
    Exception
        Re-raises the final exception after exhausting retries.
    """
    attempt = 0
    current_delay = delay

    while True:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Attempt {attempt} failed, will retry in {current_delay} seconds")
            logger.exception(f"Attempt {attempt} failed with exception: {str(e)}")
            attempt += 1
            if attempt > retries:
                raise e
            time.sleep(current_delay)
            current_delay *= backoff


_POINT_EXTS = ('.pt', '.csv')
_BATCH_EXTS = ('.parquet',)
_SUPPORTED_EXTS = _POINT_EXTS + _BATCH_EXTS

def stream_embeddings(
        path: Path,
        file_extension: Optional[str] = None,
        batch_size: int = 32768,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield (ids, [B, D] float32) batches from a file or directory of .pt / .csv / .parquet.

    Parquet shards are streamed via ``ParquetFile.iter_batches`` (many chains per file).
    .pt and .csv hold one chain per file (ID = filename stem) and are chunked into
    batches of ``batch_size``. A single file is handled as a degenerate directory-of-one;
    ``file_extension`` is ignored in that case.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Embeddings path does not exist: {path}")

    if file_extension is not None and file_extension not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file extension '{file_extension}'. "
            f"Use one of: {', '.join(_SUPPORTED_EXTS)}"
        )

    if p.is_file():
        if p.suffix not in _SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported file extension '{p.suffix}'. "
                f"Use one of: {', '.join(_SUPPORTED_EXTS)}"
            )
        files = [p]
    else:
        extensions = (file_extension,) if file_extension is not None else _SUPPORTED_EXTS
        files = []
        for ext in extensions:
            files.extend(sorted(p.glob(f"*{ext}")))
        if not files:
            raise ValueError(
                f"No embedding files found with extensions {list(extensions)} in {path}"
            )

    parquet_files = [f for f in files if f.suffix in _BATCH_EXTS]
    point_files = [f for f in files if f.suffix in _POINT_EXTS]

    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for record in pf.iter_batches(batch_size=batch_size, columns=['id', 'embedding']):
            ids = record.column('id').to_pylist()
            flat = record.column('embedding').values.to_numpy(zero_copy_only=False)
            arr = np.ascontiguousarray(
                flat.reshape(len(record), -1).astype(np.float32, copy=False)
            )
            yield ids, arr

    ids_buf: list[str] = []
    emb_buf: list[np.ndarray] = []
    for f in point_files:
        if f.suffix == '.pt':
            emb = torch.load(f, map_location='cpu', weights_only=True)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
        else:  # .csv
            emb = pd.read_csv(f, header=None).values
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)
        ids_buf.append(f.stem)
        emb_buf.append(emb.reshape(-1))
        if len(ids_buf) >= batch_size:
            yield ids_buf, np.ascontiguousarray(np.stack(emb_buf))
            ids_buf, emb_buf = [], []
    if ids_buf:
        yield ids_buf, np.ascontiguousarray(np.stack(emb_buf))

STRUCTURE_FORMAT_EXTENSIONS = {
    StructureFormat.pdb: ('.pdb', '.pdb.gz'),
    StructureFormat.mmcif: ('.cif', '.cif.gz'),
    StructureFormat.bciff: ('.bcif', '.bcif.gz'),
}


def scan_structure_folder(folder_path: Path, structure_format: StructureFormat, file_extension: str=None):
    """Scan a folder for structure files and return stream tuples (name, path, name)."""
    if file_extension is not None:
        extensions = (file_extension,)
    else:
        extensions = STRUCTURE_FORMAT_EXTENSIONS.get(structure_format)
        if extensions is None:
            raise typer.BadParameter(f"Unknown structure format: {structure_format}")

    entries = []
    for filename in sorted(os.listdir(folder_path)):
        if any(filename.endswith(ext) for ext in extensions):
            file_path = os.path.join(folder_path, filename)
            name = filename
            for ext in extensions:
                if ext and name.endswith(ext):
                    name = name[:-len(ext)]
                    break
            entries.append((name, file_path, name))

    if not entries:
        raise typer.BadParameter(
            f"No structure files with extensions {extensions} found in {folder_path}"
        )

    return tuple(entries)