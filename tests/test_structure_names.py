"""P9.B2: a structure name names one source.

The datasets read a structure once for all the rows that name it (one per chain or assembly): StructureProvider keeps
the last structure read under its name, whether the source is a path or a downloaded stream. Rows that gave one name
to two files or URLs got the first one's structure for both; they are now refused when the rows are loaded."""
import gzip
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from foldmatch.cli import embedding
from foldmatch.dataset import resdiue_assembly_embedding_from_structure, residue_embedding_from_structure
from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.dataset.utils import refuse_shared_names
from foldmatch.errors import RefusalError
from foldmatch.inference import chain_complete_inference
from foldmatch.types.api_types import SrcLocation

_RESOURCES = Path(__file__).parent / "resources"
_CIF = _RESOURCES / "pdb" / "1acb.cif"
_EMBEDDINGS = _RESOURCES / "embeddings"


class SharedNameTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.copy = Path(shutil.copyfile(_CIF, self.tmp / "1acb.cif"))
        self.link = self.tmp / "link.cif"
        os.symlink(_CIF, self.link)
        self.parquet = self.tmp / "residues.parquet"
        embeddings = [torch.load(_EMBEDDINGS / f"1acb.{c}.pt", map_location="cpu") for c in "AB"]
        kind = pa.list_(pa.list_(pa.float32(), embeddings[0].shape[1]))  # one list of D values per residue
        pq.write_table(pa.table({"id": ["1acb.A", "1acb.B"], "embedding": [e.numpy().tolist() for e in embeddings]},
                                schema=pa.schema([("id", pa.string()), ("embedding", kind)])),
                       self.parquet)

    def datasets(self, sources):
        """Each dataset that reads structures, built on rows of the name '1acb' with these sources (two rows each)."""
        three = [("1acb", source, f"1acb-{k}") for k, source in enumerate(sources)]
        chains = [("1acb", source, chain, f"1acb-{k}.{chain}") for k, source in enumerate(sources) for chain in "AB"]
        assemblies = [("1acb", source, "1", f"1acb-{k}-1") for k, source in enumerate(sources)]
        stream = SrcLocation.stream
        return {
            "EsmAssemblyFromStructure": lambda: EsmAssemblyFromStructure(src_stream=three, src_location=stream),
            "EsmProtFromStructure": lambda: EsmProtFromStructure(src_stream=three, src_location=stream),
            "EsmProtFromChain": lambda: EsmProtFromChain(src_stream=chains, src_location=stream),
            "ResidueAssemblyEmbeddingFromTensorFile": lambda: ResidueAssemblyEmbeddingFromTensorFile(
                src_stream=assemblies, res_embedding_location=_EMBEDDINGS, src_location=stream),
            "ResidueAssemblyEmbeddingFromParquet": lambda: ResidueAssemblyEmbeddingFromParquet(
                src_stream=assemblies, parquet_path=self.parquet, src_location=stream),
            "ResidueAssemblyDatasetFromStructure": lambda: ResidueAssemblyDatasetFromStructure(
                src_stream=three, res_embedding_location=_EMBEDDINGS, src_location=stream),
            "ResidueEmbeddingFromStructure": lambda: ResidueEmbeddingFromStructure(
                src_stream=three, res_embedding_location=_EMBEDDINGS, src_location=stream),
        }

    def test_two_files_under_one_name(self):
        for dataset, build in self.datasets([str(_CIF), str(self.copy)]).items():
            with self.subTest(dataset=dataset):
                with self.assertRaises(RefusalError) as ctx:
                    build()
                message = str(ctx.exception)
                self.assertIn(f"'1acb' names {min(str(_CIF), str(self.copy))} and {max(str(_CIF), str(self.copy))}",
                              message)
                self.assertIn("give each file its own name", message)

    def test_one_file_under_one_name(self):
        """The same file on several rows, under two spellings: a symbolic link to it, a hard link to it."""
        hard = self.tmp / "hard.cif"
        os.link(self.copy, hard)
        for spellings in ([str(_CIF), str(self.link)], [str(self.copy), str(hard)]):
            for dataset, build in self.datasets(spellings).items():
                with self.subTest(spellings=spellings, dataset=dataset):
                    build()

    def test_what_is_compared(self):
        """URLs differing only in the case of their scheme and host are one source, other URLs two; a row without a
        name or a source is left to the dataset, as before."""
        def refused(rows):
            try:
                refuse_shared_names(pd.DataFrame(rows, columns=["stream_name", "stream"]), "stream_name", "stream")
            except RefusalError:
                return True
            return False
        self.assertFalse(refused([("x", "HTTPS://Example.ORG/a/1acb.cif"), ("x", "https://example.org/a/1acb.cif")]))
        self.assertTrue(refused([("x", "https://example.org/a/1acb.cif"), ("x", "https://example.org/A/1acb.cif")]))
        self.assertTrue(refused([("x", "https://example.org/1acb.cif"), ("x", str(_CIF))]))
        self.assertFalse(refused([("x", str(_CIF)), ("x", None), (None, str(self.copy)), ("y", str(self.copy))]))

    def test_two_urls_under_one_name(self):
        """Refused before anything is downloaded."""
        urls = ["https://example.org/a/1acb.cif", "https://example.org/b/1acb.cif"]
        never = mock.Mock(side_effect=AssertionError("downloaded"))
        with mock.patch.object(resdiue_assembly_embedding_from_structure, "stringio_from_url", never), \
                mock.patch.object(residue_embedding_from_structure, "stringio_from_url", never):
            for dataset, build in self.datasets(urls).items():
                with self.subTest(dataset=dataset):
                    with self.assertRaises(RefusalError) as ctx:
                        build()
                    self.assertIn(f"'1acb' names {urls[0]} and {urls[1]}", str(ctx.exception))
        never.assert_not_called()

    def test_a_folder_with_two_files_of_one_entry(self):
        """'fm-embedding from-structures chain' on a folder with 1acb.cif and 1acb.cif.gz, which both get the name
        1acb: exit status 1, one 'Error:' line, before ESM3 is loaded."""
        folder = self.tmp / "structures"
        folder.mkdir()
        shutil.copyfile(_CIF, folder / "1acb.cif")
        with gzip.open(folder / "1acb.cif.gz", "wt") as f:
            f.write(_CIF.read_text())
        out = self.tmp / "out"
        out.mkdir()
        argv = ["fm-embedding", "from-structures", "chain", "--src-folder", str(folder), "--output-folder", str(out)]
        stderr = self.enterContext(mock.patch("sys.stderr", new_callable=io.StringIO))
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(chain_complete_inference, "get_residue_model",
                                      side_effect=AssertionError("ESM3 loaded")), \
                    self.assertRaises(SystemExit) as ctx:
                embedding.main()
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        self.assertEqual(ctx.exception.code, 1)
        errors = [line for line in stderr.getvalue().splitlines() if line.startswith("Error: ")]
        self.assertEqual(len(errors), 1, stderr.getvalue())
        resolved = folder.resolve()  # --src-folder is resolved (/var -> /private/var on macOS)
        self.assertIn(f"'1acb' names {resolved / '1acb.cif'} and {resolved / '1acb.cif.gz'}", errors[0])
        self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
