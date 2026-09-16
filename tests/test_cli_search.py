import os
import re
import shutil
import unittest
import tempfile
from pathlib import Path

from foldmatch.search.output import CLUSTER_OUTPUT_FIELDS, DEFAULT_OUTPUT_FIELDS
from foldmatch.types.api_types import StructureFormat, Accelerator, Granularity, FormatMode


class TestCliSearch(unittest.TestCase):
    __test_path = os.path.dirname(__file__)
    __temp_dir = None
    __db_path = None
    __temp_file = None

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Create temporary directory for test outputs
        cls.__temp_dir = Path(tempfile.mkdtemp())
        cls.__db_path = os.path.join(cls.__temp_dir, "test_faiss")
        cls.__temp_file = os.path.join(cls.__temp_dir, "temp_embeddings.pt")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests complete."""
        if cls.__temp_dir and os.path.exists(cls.__temp_dir):
            shutil.rmtree(cls.__temp_dir)

    def test_01_build_database(self):
        """Test building a database from structure files."""
        from foldmatch.cli.search import build_database_from_structures

        # Use the test PDB files
        structure_dir = Path(f"{self.__test_path}/resources/pdb")

        build_database_from_structures(
            structure_folder=structure_dir,
            output_db=self.__db_path,
            tmp_embedding_folder=self.__temp_dir,
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            min_res=10,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        # Verify database files were created
        db_path = Path(self.__db_path)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

    def test_02_query_database(self):
        """Test querying the database with a structure."""
        from foldmatch.cli.search import query_database_from_structure

        # Use one of the test structures as the query
        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")
        output_file = os.path.join(self.__temp_dir, "search_results.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=5,
            threshold=None,
            output_file=output_file,
            min_res=10,
            max_res=None,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_file))

        # Embedding-only search: TSV, no header, query/target/embscore.
        with open(output_file, 'r') as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        self.assertGreater(len(lines), 0)
        for ln in lines:
            cols = ln.split("\t")
            self.assertEqual(len(cols), 3)
            float(cols[2])  # embedding score parses as a number

    def test_03_query_with_threshold(self):
        """Test querying with a similarity score threshold."""
        from foldmatch.cli.search import query_database_from_structure

        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")
        output_file = os.path.join(self.__temp_dir, "search_results_threshold.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id=None,
            top_k=10,
            threshold=0.5,  # Apply threshold
            output_file=output_file,
            min_res=10,
            max_res=None,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_file))

    def test_04_query_specific_chain(self):
        """Test querying with a specific chain ID."""
        from foldmatch.cli.search import query_database_from_structure

        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")
        output_file = os.path.join(self.__temp_dir, "search_results_chain.csv")

        query_database_from_structure(
            db_path=self.__db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id="A",  # Query specific chain
            top_k=5,
            threshold=None,
            output_file=output_file,
            min_res=10,
            max_res=None,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_file))

    def test_05_stats(self):
        """Test getting database statistics."""
        from foldmatch.cli.search import show_statistics

        # This should run without errors
        show_statistics(
            db_path=self.__db_path
        )

    def test_07_faiss_database_class(self):
        """Test FaissEmbeddingDatabase class directly."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        import numpy as np
        import torch

        db_path = Path(os.path.join(self.__temp_dir, "test_chroma_direct"))
        db = FaissEmbeddingDatabase(db_folder=db_path, index_name="test_direct")

        # Create some dummy embeddings
        chain_ids = ["test1:A", "test2:A", "test3:B"]
        embeddings_arr = np.stack(
            [torch.randn(256).numpy() for _ in range(3)]
        ).astype(np.float32)

        # Create database (new batch-iterator API)
        db.create_database(embedding_batches=[(chain_ids, embeddings_arr)])

        # Load database
        db2 = FaissEmbeddingDatabase(db_folder=db_path, index_name="test_direct")
        db2.load_database()

        # Get stats
        stats = db2.get_statistics()
        self.assertEqual(stats['total_embeddings'], 3)

        # Search
        query_embedding = torch.randn(256)
        results_ids, results_distances = db2.search(query_embedding, top_k=2)
        self.assertEqual(len(results_ids), 2)
        self.assertEqual(len(results_distances), 2)

    def test_09_residue_embedding_by_chain_filter(self):
        """Test residue_embedding_by_chain with chain_id parameter."""
        from foldmatch.foldmatch import FoldMatch

        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")

        embedder = FoldMatch(min_res=10)
        import torch
        embedder.load_models(device=torch.device("cpu"))

        # Test getting all chains
        all_chains = embedder.residue_embedding_by_chain(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif
        )
        self.assertGreater(len(all_chains), 1)  # 1acb has multiple chains
        self.assertIn("A", all_chains)

        # Test getting only chain A
        chain_a_only = embedder.residue_embedding_by_chain(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            chain_id="A"
        )
        self.assertEqual(len(chain_a_only), 1)
        self.assertIn("A", chain_a_only)
        self.assertNotIn("B", chain_a_only)

        # Verify the embeddings are the same for chain A
        import torch
        self.assertTrue(torch.equal(all_chains["A"], chain_a_only["A"]))

    def test_10_residue_embedding_by_assembly(self):
        """Test residue_embedding_by_assembly with assembly_id parameter."""
        from foldmatch.foldmatch import FoldMatch

        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")

        embedder = FoldMatch(min_res=10)
        import torch
        embedder.load_models(device=torch.device("cpu"))

        # Test getting asymmetric unit (no assembly_id provided)
        asymmetric_unit = embedder.residue_embedding_by_assembly(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif
        )
        self.assertEqual(len(asymmetric_unit), 1)  # Should have only asymmetric unit
        self.assertIn("0", asymmetric_unit)  # Key "0" represents asymmetric unit
        self.assertGreater(len(asymmetric_unit["0"]), 1)  # 1acb has multiple chains

        # Test getting only assembly 1
        assembly_1_only = embedder.residue_embedding_by_assembly(
            src_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            assembly_id="1"
        )
        self.assertEqual(len(assembly_1_only), 1)
        self.assertIn("1", assembly_1_only)

        # Verify the asymmetric unit and assembly 1 have the same residue embeddings
        import torch
        self.assertTrue(torch.equal(asymmetric_unit["0"], assembly_1_only["1"]))


    def test_11_query_database_with_database(self):
        """Test querying one database against another."""
        from foldmatch.cli.search import query_database_from_database

        output_file = os.path.join(self.__temp_dir, "database_to_database_results.csv")

        query_database_from_database(
            query_db_path=self.__db_path,
            subject_db_path=self.__db_path,
            top_k=3,
            threshold=None,
            output_file=output_file,
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_file))

        with open(output_file, 'r') as f:
            lines = [ln for ln in f if ln.strip()]
        self.assertGreater(len(lines), 0)
        self.assertEqual(len(lines[0].rstrip("\n").split("\t")), 3)

    def test_12_embedding_search_search_by_database(self):
        """Test EmbeddingSearch database-to-database search API."""
        from foldmatch.search.embedding_database import EmbeddingDatabase

        searcher = EmbeddingDatabase(
            db_path=self.__db_path
        )
        searcher.load(
            use_gpu=False
        )
        results = searcher.search_by_database(
            query_db_path=self.__db_path,
            top_k=2
        )

        self.assertGreater(len(results), 0)
        for query_chain, (matching_ids, scores) in results.items():
            self.assertGreater(len(matching_ids), 0)
            self.assertEqual(len(matching_ids), len(scores))

    def test_13_build_database_assembly(self):
        """Test building a database with assembly granularity."""
        from foldmatch.cli.search import build_database_from_structures

        # Use the test PDB files
        structure_dir = Path(f"{self.__test_path}/resources/pdb")
        assembly_db_path = os.path.join(self.__temp_dir, "test_faiss_assembly")

        build_database_from_structures(
            structure_folder=structure_dir,
            output_db=assembly_db_path,
            tmp_embedding_folder=self.__temp_dir,
            structure_format=StructureFormat.mmcif,
            file_extension=".cif",
            min_res=10,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
            granularity=Granularity.assembly
        )

        # Verify database files were created
        db_path = Path(assembly_db_path)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

    def test_14_query_database_assembly(self):
        """Test querying the database with assembly granularity."""
        from foldmatch.cli.search import query_database_from_structure

        # Use the assembly database built in test_13
        assembly_db_path = os.path.join(self.__temp_dir, "test_faiss_assembly")
        query_structure = Path(f"{self.__test_path}/resources/pdb/1acb.cif")
        output_file = os.path.join(self.__temp_dir, "search_results_assembly.csv")

        query_database_from_structure(
            db_path=assembly_db_path,
            query_structure=query_structure,
            structure_format=StructureFormat.mmcif,
            top_k=5,
            threshold=None,
            output_file=output_file,
            min_res=10,
            max_res=None,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
            granularity=Granularity.assembly,
            assembly_id='1'
        )

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_file))

        # Embedding-only search: TSV, no header.
        with open(output_file, 'r') as f:
            lines = [ln for ln in f if ln.strip()]
        self.assertGreater(len(lines), 0)
        self.assertEqual(len(lines[0].rstrip("\n").split("\t")), 3)


    def test_17_build_database_from_embeddings(self):
        """Test building a database from pre-computed embedding .pt files."""
        from foldmatch.cli.search import build_database_from_embeddings

        embedding_dir = Path(f"{self.__test_path}/resources/embeddings")
        output_db = os.path.join(self.__temp_dir, "test_from_embeddings")

        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=output_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        from foldmatch.search.faiss_database import FaissEmbeddingDatabase

        db_path = Path(output_db)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

        db = FaissEmbeddingDatabase(db_folder=db_path.parent, index_name=db_path.name)
        db.load_database()
        self.assertEqual(len(db.chain_ids), 5)

    def test_18_build_database_from_fasta(self):
        """Test building a database from a FASTA file."""
        from foldmatch.cli.search import build_database_from_fasta

        fasta_file = Path(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        output_db = os.path.join(self.__temp_dir, "test_from_fasta")

        build_database_from_fasta(
            fasta_file=fasta_file,
            output_db=output_db,
            tmp_embedding_folder=self.__temp_dir,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        from foldmatch.search.faiss_database import FaissEmbeddingDatabase

        db_path = Path(output_db)
        self.assertTrue((db_path.parent / f"{db_path.name}.index").exists())
        self.assertTrue((db_path.parent / f"{db_path.name}.metadata").exists())

        db = FaissEmbeddingDatabase(db_folder=db_path.parent, index_name=db_path.name)
        db.load_database()
        self.assertEqual(len(db.chain_ids), 2)

    def test_21_query_database_from_embedding(self):
        """Test querying the database with a pre-computed embedding file."""
        from foldmatch.cli.search import build_database_from_embeddings, query_database_from_embedding

        embedding_dir = Path(f"{self.__test_path}/resources/embeddings")
        subject_db = os.path.join(self.__temp_dir, "test_query_from_emb_db")
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=subject_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        # Use one of the pre-computed .pt chain embedding files as the query
        embedding_folder = Path(f"{self.__test_path}/resources/embeddings")
        output_file = os.path.join(self.__temp_dir, "query_from_embedding_results.csv")

        query_database_from_embedding(
            db_path=subject_db,
            embedding_folder=embedding_folder,
            file_extension=".pt",
            top_k=5,
            threshold=None,
            output_file=output_file,
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        # No header: every row is a result, query id in the first column.
        self.assertGreater(len(lines), 0)
        query_ids = {ln.split("\t")[0] for ln in lines}
        self.assertIn("1acb.A", query_ids)

    def test_22_query_database_from_fasta(self):
        """Test querying the database with sequences from a FASTA file."""
        from foldmatch.cli.search import build_database_from_embeddings, query_database_from_fasta

        embedding_dir = Path(f"{self.__test_path}/resources/embeddings")
        subject_db = os.path.join(self.__temp_dir, "test_query_from_fasta_db")
        build_database_from_embeddings(
            embedding_folder=embedding_dir,
            output_db=subject_db,
            file_extension=".pt",
            use_gpu_index=False
        )

        fasta_file = Path(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        output_file = os.path.join(self.__temp_dir, "query_from_fasta_results.csv")

        query_database_from_fasta(
            db_path=subject_db,
            fasta_file=fasta_file,
            tmp_embedding_folder=self.__temp_dir,
            top_k=5,
            threshold=None,
            output_file=output_file,
            accelerator=Accelerator.cpu,
            use_gpu_index=False
        )

        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        # Subject DB was built from embeddings, so it has no sequence store:
        # Stage 2 is skipped and this is the embedding-only 3-column output.
        self.assertGreater(len(lines), 1)
        query_ids = {ln.split("\t")[0] for ln in lines}
        self.assertIn("1acb_E", query_ids)
        self.assertIn("2uzi_A", query_ids)
        for ln in lines:
            self.assertEqual(len(ln.split("\t")), 3)

    def test_23_cluster_database(self):
        """Test cluster_database CLI command on the test fixture FAISS database."""
        from foldmatch.cli.search import cluster_database

        output_file = os.path.join(self.__temp_dir, "clusters.tsv")
        cluster_database(
            db_path=self.__db_path,
            threshold=0.0,
            resolution=1.0,
            output_file=output_file,
            max_neighbors=10,
            min_cluster_size=None,
            use_gpu_index=False,
            seed=42,
        )

        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        # Same clustering with --format-mode tsv gains exactly one header row.
        headed = os.path.join(self.__temp_dir, "clusters_headed.tsv")
        cluster_database(
            db_path=self.__db_path,
            threshold=0.0,
            resolution=1.0,
            output_file=headed,
            max_neighbors=10,
            min_cluster_size=None,
            use_gpu_index=False,
            seed=42,
            format_mode=FormatMode.tsv,
        )
        with open(headed, 'r') as f:
            headed_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        self.assertEqual(headed_lines[0].split("\t"), list(CLUSTER_OUTPUT_FIELDS))

        # TSV, no header: every line is a real chain assignment.
        self.assertGreater(len(lines), 0)
        for ln in lines:
            cols = ln.split("\t")
            self.assertEqual(len(cols), len(CLUSTER_OUTPUT_FIELDS))
            int(cols[1])                      # cluster_id
            self.assertGreater(int(cols[2]), 0)  # cluster_size
        # First column is a chain id from the fixture database, not a header.
        self.assertNotEqual(lines[0].split("\t")[0], "chain_id")

        # cluster_size must agree with the number of rows sharing a cluster_id.
        from collections import Counter
        counts = Counter(ln.split("\t")[1] for ln in lines)
        for ln in lines:
            _chain, cluster_id, size = ln.split("\t")
            self.assertEqual(int(size), counts[cluster_id])

        # The header is the only difference between the two modes.
        self.assertEqual(headed_lines[1:], lines)

    def test_24_similarity_graph(self):
        """Test similarity_graph CLI command exports a non-empty GraphML file."""
        from foldmatch.cli.search import similarity_graph

        output_graphml = os.path.join(self.__temp_dir, "similarity_graph.graphml")
        similarity_graph(
            db_path=self.__db_path,
            threshold=0.0,
            output_file=output_graphml,
            max_neighbors=10,
            use_gpu_index=False,
        )

        self.assertTrue(os.path.exists(output_graphml))
        self.assertGreater(os.path.getsize(output_graphml), 0)

        # GraphML is XML; verify the output is parseable and reflects the
        # fixture database (>0 nodes, with chain-id name attributes preserved).
        import xml.etree.ElementTree as ET
        tree = ET.parse(output_graphml)
        root = tree.getroot()
        # GraphML default namespace
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = root.findall('.//g:node', ns)
        self.assertGreater(len(nodes), 0)

    def test_25_stage2_format_output_tsv(self):
        """Stage-2 sequence-identity search emits mmseqs-style TSV columns.

        A FASTA-built database carries a sequence store, so querying it with
        FASTA sequences triggers Stage-2 alignment; the results are written as
        tab-separated rows (no header) over the requested --format-output fields.
        """
        from foldmatch.cli.search import (
            build_database_from_fasta,
            query_database_from_fasta,
        )

        fasta_file = Path(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        subject_db = os.path.join(self.__temp_dir, "test_stage2_fmt_db")
        build_database_from_fasta(
            fasta_file=fasta_file,
            output_db=subject_db,
            tmp_embedding_folder=self.__temp_dir,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
        )

        # Default format: query,target,embscore + the mmseqs defaults.
        default_tsv = os.path.join(self.__temp_dir, "stage2_default.tsv")
        query_database_from_fasta(
            db_path=subject_db,
            fasta_file=fasta_file,
            tmp_embedding_folder=self.__temp_dir,
            top_k=5,
            threshold=None,
            output_file=default_tsv,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
        )
        with open(default_tsv) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        self.assertGreater(len(lines), 0)
        n_default = len(DEFAULT_OUTPUT_FIELDS)
        for ln in lines:
            cols = ln.split("\t")
            self.assertEqual(len(cols), n_default)
            self.assertNotIn(",", ln)            # tab-separated, not CSV
        # No header row: first token of every line is a real query id.
        query_ids = {ln.split("\t")[0] for ln in lines}
        self.assertTrue(query_ids <= {"1acb_E", "2uzi_A"})
        # Column order is exactly DEFAULT_OUTPUT_FIELDS, so look fields up by
        # name rather than hardcoding indices.
        idx = {name: i for i, name in enumerate(DEFAULT_OUTPUT_FIELDS)}
        self.assertEqual(DEFAULT_OUTPUT_FIELDS[:3], ("query", "target", "embscore"))
        # Each query self-matches at full identity, with an embedding score.
        self_hits = [ln.split("\t") for ln in lines
                     if ln.split("\t")[0] == ln.split("\t")[1]]
        self.assertTrue(self_hits)
        self.assertEqual(self_hits[0][idx["fident"]], "1.000")
        float(self_hits[0][idx["embscore"]])

        # Custom format: a heavy column (cigar) plus qseq/qlen must be honored.
        custom_tsv = os.path.join(self.__temp_dir, "stage2_custom.tsv")
        query_database_from_fasta(
            db_path=subject_db,
            fasta_file=fasta_file,
            tmp_embedding_folder=self.__temp_dir,
            top_k=5,
            threshold=None,
            output_file=custom_tsv,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
            format_output="query,target,pident,alnlen,cigar,qseq,qlen",
        )
        with open(custom_tsv) as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        self.assertTrue(rows)
        for cols in rows:
            self.assertEqual(len(cols), 7)
            _q, _t, _pident, alnlen, cigar, qseq, qlen = cols
            # CIGAR is a run-length M/D/I string summing to the alignment length.
            self.assertRegex(cigar, r"^(\d+[MDI])+$")
            self.assertEqual(len(qseq), int(qlen))
            cigar_len = sum(int(n) for n in re.findall(r"(\d+)[MDI]", cigar))
            self.assertEqual(cigar_len, int(alnlen))

    def test_26_stage2_rejects_unknown_format_field(self):
        """An unknown --format-output field fails with a clear error."""
        from foldmatch.cli.search import (
            build_database_from_fasta,
            query_database_from_fasta,
        )

        fasta_file = Path(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        subject_db = os.path.join(self.__temp_dir, "test_stage2_badfmt_db")
        build_database_from_fasta(
            fasta_file=fasta_file,
            output_db=subject_db,
            tmp_embedding_folder=self.__temp_dir,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
        )
        with self.assertRaises(ValueError) as ctx:
            query_database_from_fasta(
                db_path=subject_db,
                fasta_file=fasta_file,
                tmp_embedding_folder=self.__temp_dir,
                top_k=5,
                threshold=None,
                output_file=os.path.join(self.__temp_dir, "unused.tsv"),
                accelerator=Accelerator.cpu,
                use_gpu_index=False,
                format_output="query,target,not_a_field",
            )
        self.assertIn("not_a_field", str(ctx.exception))

    def test_27_fasta_min_res_n_filters_index_store_and_queries(self):
        """--min-res-n drops short sequences from the index, the sequence store
        and the embedded queries alike.

        test_sequences.fasta holds 1acb_E (245 aa) and 2uzi_A (58 aa), so with
        min_res_n=100 only 1acb_E may survive. The FAISS index and the sequence
        store must hold the same ids, otherwise Stage-2 silently skips hits that
        are "missing from sequence store".
        """
        from foldmatch.cli.search import (
            build_database_from_fasta,
            query_database_from_fasta,
        )
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from foldmatch.search.sequence_store import SequenceStore

        fasta_file = Path(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        subject_db = os.path.join(self.__temp_dir, "test_min_res_n_db")
        build_database_from_fasta(
            fasta_file=fasta_file,
            output_db=subject_db,
            tmp_embedding_folder=self.__temp_dir,
            min_res_n=100,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
        )

        db_path = Path(subject_db)
        db = FaissEmbeddingDatabase(db_folder=db_path.parent, index_name=db_path.name)
        db.load_database()
        self.assertEqual(list(db.chain_ids), ["1acb_E"])
        store = SequenceStore(db_path.parent, db_path.name)
        self.assertEqual(set(store.fetch(["1acb_E", "2uzi_A"])), set(db.chain_ids))

        output_file = os.path.join(self.__temp_dir, "min_res_n_results.tsv")
        query_database_from_fasta(
            db_path=subject_db,
            fasta_file=fasta_file,
            tmp_embedding_folder=self.__temp_dir,
            output_file=output_file,
            min_res_n=100,
            top_k=5,
            threshold=None,
            accelerator=Accelerator.cpu,
            use_gpu_index=False,
        )
        with open(output_file) as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        self.assertTrue(rows)
        self.assertEqual({cols[0] for cols in rows}, {"1acb_E"})


if __name__ == '__main__':
    unittest.main()
