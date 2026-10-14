from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Tuple, List, Optional, Literal

StreamSrc = Path | IO
StreamTuple = Tuple[str, StreamSrc, str, str] | Tuple[str, StreamSrc, str] | Tuple[str, str] | Tuple[str, str, str] | Tuple[str, str, str, str]
FileOrStreamTuple = Path | List[StreamTuple]
Devices = int | List[int] | Literal["auto"]

class StructureFormat(str, Enum):
    pdb = "pdb"
    mmcif = "mmcif"
    bciff = "binarycif"

class Granularity(str, Enum):
    chain = "chain"
    assembly = "assembly"

class Accelerator(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    hpu = "hpu"
    auto = "auto"

class Strategy(str, Enum):
    ddp = "ddp"
    auto = "auto"

class SrcLocation(str, Enum):
    file = "file"
    stream = "stream"


class StructureLocation(str, Enum):
    local = "local"
    remote = "remote"


class SrcProteinFrom(str, Enum):
    chain = "chain"
    structure = "structure"


class SrcEsmFrom(str, Enum):
    fasta = "fasta"
    structure = "structure"


class SrcAssemblyFrom(str, Enum):
    assembly = "assembly"
    structure = "structure"


class SrcTensorFrom(str, Enum):
    file = "file"
    structure = "structure"
    parquet = "parquet"

class OutFormat(str, Enum):
    csv = "csv"
    pt = "pt"
    parquet = "parquet"
    json = "json"


class ResEmbeddingFormat(str, Enum):
    pt = "pt"
    csv = "csv"

class LogLevel(str, Enum):
    info = "info"
    warn = "warn"
    debug = "debug"


class IndexType(str, Enum):
    """FAISS index variant used when building an embedding database."""
    auto = "auto"        # IndexFlatIP if N<10k else IndexHNSWFlat (legacy default)
    flat = "flat"        # IndexFlatIP — exact, RAM-resident
    hnsw = "hnsw"        # IndexHNSWFlat — approximate, RAM-resident
    ivf_pq = "ivf_pq"    # OPQ + IVF-PQ with OnDiskInvertedLists — approximate, on-disk


class FormatMode(str, Enum):
    """How a result file is laid out.

    The column *set* is chosen elsewhere (``--format-output`` for a
    sequence-identity search; fixed for the other commands); this only controls
    the file's shape. New layouts are added by defining a member here and a
    matching spec in :mod:`foldmatch.search.output`.
    """
    headless_tsv = "headless_tsv"  # tab-separated, no header row (default)
    tsv = "tsv"                    # tab-separated with a header row


class ScoringChoice(str, Enum):
    """Stage-2 scoring of ``fm-search query sequences``: one scoring per query run, never both (D7, user
    2026-10-13). ``blosum62`` is the default for now; which one becomes the default is decided before the release."""
    blosum62 = "blosum62"    # biotite BLOSUM62 Smith-Waterman, BLAST-style statistics (the existing path)
    esm_codes = "esm-codes"  # ESM3 residue codes, the system named in the database manifest (S1-S16)


DEFAULT_SCORING = ScoringChoice.blosum62


class SignificanceMode(str, Enum):
    """Scale used for Stage-2 bit scores, p-values and E-values."""
    default = "default"  # calibrated constants + finite-size correction; needs gaps 11/1
    sampled = "sampled"  # lambda/K estimated by sampling; any gaps, unreliable magnitudes


@dataclass
class IndexConfig:
    """Tuning knobs for IndexType. All fields are optional; sensible defaults apply."""
    # IVF-PQ
    nlist: int = 16384            # number of IVF cells
    m: int = 64                   # PQ subquantizers (must divide D)
    nbits: int = 8                # bits per PQ subcode
    nprobe: Optional[int] = None  # search-time cells probed; defaults to nlist // 64
    opq_train_size: int = 1_000_000  # training sample size for OPQ + IVF-PQ
    # HNSW
    hnsw_m: int = 32              # HNSW graph connectivity