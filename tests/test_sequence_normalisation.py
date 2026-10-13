"""Input normalisation shared by every FASTA path (G8)."""
import hashlib
import json
import string
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS, normalize_sequence, scan_fasta, seq_sha1

_TEST_PATH = Path(__file__).parent
_FIXTURE_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_TOKENIZER_CASES = _TEST_PATH / "resources" / "esm_codes" / "tokenizer_cases_v1.json"


def _fixture_records():
    from foldmatch.utils.fasta import parse_fasta
    return parse_fasta(_FIXTURE_FASTA)


def _messy_fasta(path: Path):
    """The fixture FASTA in lowercase, with whitespace inside the sequence lines."""
    with open(path, "w") as f:
        for name, sequence in _fixture_records():
            messy = sequence.lower()
            lines = [messy[i:i + 30] for i in range(0, len(messy), 30)]
            # Spaces and tabs inside lines, and a CRLF line ending.
            lines = [" ".join([line[:10], line[10:20], line[20:]]).rstrip() + "\t" for line in lines]
            f.write(f">{name} some description\r\n")
            f.write("\n".join(lines) + "\n\n")


class NormalizeSequenceTest(unittest.TestCase):

    def test_uppercases_and_removes_ascii_whitespace(self):
        self.assertEqual(normalize_sequence("acDe"), "ACDE")
        self.assertEqual(normalize_sequence(" A\tC\nD\rE\x0bF\x0c "), "ACDEF")
        self.assertEqual(normalize_sequence(""), "")

    def test_every_other_ascii_character_is_kept_one_for_one(self):
        for char in map(chr, range(128)):
            with self.subTest(char=repr(char)):
                if char in string.whitespace:
                    expected = ""
                elif char in string.ascii_lowercase:
                    expected = char.upper()
                else:
                    expected = char
                self.assertEqual(normalize_sequence(f"{char}"), expected)
                self.assertEqual(normalize_sequence(f"A{char}C"), f"A{expected}C")

    def test_non_ascii_is_refused(self):
        for raw in ("ACDÉ", "AC DE", "ßA"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    normalize_sequence(raw)

    def test_non_ascii_kept_when_allowed(self):
        # Only ASCII letters change: 'ß'.upper() would be 'SS' and shift every later residue.
        self.assertEqual(normalize_sequence("aß cé", allow_non_ascii=True), "AßCé")
        self.assertEqual(normalize_sequence("ac de", allow_non_ascii=True), "ACDE")

    def test_idempotent(self):
        for raw in ("ac de", "XBZUO", "-.|_*", "a<mask>b"):
            once = normalize_sequence(raw)
            self.assertEqual(normalize_sequence(once), once)

    def test_seq_sha1(self):
        self.assertEqual(seq_sha1("ACDE"), hashlib.sha1(b"ACDE").digest())
        self.assertEqual(seq_sha1("ac de\n"), seq_sha1("ACDE"))
        self.assertEqual(len(seq_sha1("A")), 20)
        with self.assertRaises(ValueError):
            seq_sha1("ACÉ")

    def test_scan_counts_normalised_residues(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "x.fasta"
            fasta.write_text(">a desc\nAC DE\n  FG\t\n\n>empty\n\n>b\nmk\n>c\n \n")
            records = list(scan_fasta(fasta))
        self.assertEqual([(name, length) for _offset, name, length in records], [("a", 6), ("b", 2)])
        self.assertEqual(records[0][0], 0)


class TokenizerInvariantTest(unittest.TestCase):
    """F-tok: what ESM3's tokenizer makes of raw and normalised input (no weights needed)."""

    @classmethod
    def setUpClass(cls):
        from esm.tokenization import EsmSequenceTokenizer
        from esm.utils.encoding import tokenize_sequence
        # The tokenizer foldmatch.utils.esm.loaders gives ESM3 (ESM3.encode -> tokenize_sequence).
        tokenizer = EsmSequenceTokenizer()
        cls.tokenize = staticmethod(
            lambda s: tokenize_sequence(s, tokenizer, add_special_tokens=True).tolist()
        )
        with open(_TOKENIZER_CASES) as f:
            cls.cases = json.load(f)["cases"]

    def test_cases(self):
        self.assertGreaterEqual(len(self.cases), 20)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                raw = case["raw"]
                self.assertEqual(self.tokenize(raw), case["tokens_raw"])
                if case["normalize_refuses"]:
                    with self.assertRaises(ValueError):
                        normalize_sequence(raw)
                    continue
                normalized = normalize_sequence(raw)
                self.assertEqual(normalized, case["normalized"])
                tokens = self.tokenize(normalized)
                self.assertEqual(tokens, case["tokens_normalized"])
                # One token per residue of the normalised string, plus BOS and EOS.
                self.assertEqual(len(tokens) - 2, len(normalized))
                self.assertTrue(case["length_preserved"])
                self.assertEqual(sum(c not in STANDARD_AMINO_ACIDS for c in normalized),
                                 case["n_nonstandard"])

    def test_lowercase_and_whitespace_no_longer_tokenise_as_unk(self):
        unk = 3
        raw = "acde fghik"
        self.assertIn(unk, self.tokenize(raw))
        self.assertNotIn(unk, self.tokenize(normalize_sequence(raw)))


class FoldMatchApiNormalisationTest(unittest.TestCase):
    """FoldMatch.sequence_embedding (Python API) embeds the normalised sequence too."""

    class FakeEsm3:
        """Records the sequence ESM3 is asked to encode; no weights."""
        def __init__(self):
            from types import SimpleNamespace
            self.transformer = SimpleNamespace(blocks=[])
            self.encoded = []

        def encode(self, protein):
            self.encoded.append(protein.sequence)
            return protein

        def forward_and_sample(self, tensor, config):
            import torch
            from types import SimpleNamespace
            return SimpleNamespace(per_residue_embedding=torch.zeros(len(tensor.sequence) + 2, 4))

    def _embed(self, sequence, **kwargs):
        from foldmatch.foldmatch import FoldMatch
        fake = self.FakeEsm3()
        with patch("foldmatch.foldmatch._load_res_model", return_value=fake):
            embedding = FoldMatch(**kwargs).sequence_embedding(sequence)
        return fake.encoded, embedding

    def test_lowercase_and_whitespace_reach_esm3_normalised(self):
        encoded, embedding = self._embed("acde fghik\tLM\r\n", min_res=1)
        self.assertEqual(encoded, ["ACDEFGHIKLM"])
        self.assertEqual(tuple(embedding.shape), (13, 4))
        encoded, _ = self._embed(">q some header\nacde fghik\nlm\n", min_res=1)
        self.assertEqual(encoded, ["ACDEFGHIKLM"])

    def test_length_limits_count_normalised_residues(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            self._embed("ACDE FGHIK", min_res=10)  # 10 characters, 9 residues
        encoded, _ = self._embed("ACDE FGHIK LM", max_res=11, min_res=1)  # 13 characters, 11 residues
        self.assertEqual(encoded, ["ACDEFGHIKLM"])


class LowercaseFastaTest(unittest.TestCase):
    """A lowercase FASTA with whitespace inside lines is read as the uppercase fixture."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.messy = self.tmp / "messy.fasta"
        _messy_fasta(self.messy)

    def test_dataset_yields_the_normalised_sequences(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        clean = EsmProtFromFasta(_FIXTURE_FASTA)
        messy = EsmProtFromFasta(self.messy)
        self.assertEqual(len(messy), 2)
        for i in range(2):
            (clean_prot, clean_name), (messy_prot, messy_name) = clean[i], messy[i]
            self.assertEqual(messy_name, clean_name)
            self.assertEqual(messy_prot.sequence, clean_prot.sequence)
        self.assertEqual([len(messy[i][0]) for i in range(2)], [245, 58])

    def test_min_res_n_counts_normalised_residues(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        fasta = self.tmp / "spaced.fasta"
        # 4 residues but 8 characters on the line.
        fasta.write_text(">short\nA C D E\n>long\nACDEFGHIK\n")
        dataset = EsmProtFromFasta(fasta, min_res_n=5)
        self.assertEqual([dataset[i][1] for i in range(len(dataset))], ["long"])

    def test_sequence_store_keeps_the_normalised_sequences(self):
        from foldmatch.search.sequence_store import SequenceStore
        store = SequenceStore(self.tmp, "messy")
        store.create(self.messy)
        self.assertEqual(store.fetch(["1acb_E", "2uzi_A"]), dict(_fixture_records()))
        self.assertEqual(store.meta()["total_residues"], 303)

    def test_query_path_aligns_the_normalised_sequences(self):
        from foldmatch.cli import search
        captured = {}

        def stage2(**kwargs):
            captured.update(kwargs["query_sequences"])

        with patch("foldmatch.search.embedding_computer.chain_predict"), \
                patch("foldmatch.search.embedding_computer.EmbeddingComputer.get_embedding_batches",
                      return_value=iter(())), \
                patch("foldmatch.search.embedding_database.EmbeddingDatabase.load"), \
                patch("foldmatch.search.embedding_database.EmbeddingDatabase.search_by_embeddings",
                      return_value={}), \
                patch("foldmatch.cli.search._resolve_stage2", return_value=True), \
                patch("foldmatch.cli.search._stage2_align_and_report", side_effect=stage2):
            search.query_database_from_fasta(
                db_path=str(self.tmp / "db"), fasta_file=self.messy, tmp_embedding_folder=self.tmp,
                output_file=str(self.tmp / "out.tsv"),
            )
        self.assertEqual(captured, dict(_fixture_records()))

    def test_tokens_match_the_uppercase_fixture(self):
        from esm.tokenization import EsmSequenceTokenizer
        from esm.utils.encoding import tokenize_sequence
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        tokenizer = EsmSequenceTokenizer()
        clean, messy = EsmProtFromFasta(_FIXTURE_FASTA), EsmProtFromFasta(self.messy)
        for i in range(2):
            self.assertTrue(tokenize_sequence(messy[i][0].sequence, tokenizer).equal(
                tokenize_sequence(clean[i][0].sequence, tokenizer)))

    def test_chain_embeddings_match_the_uppercase_fixture(self):
        """Before normalisation the lowercase records were all <unk> and embedded as garbage."""
        import torch
        from foldmatch.inference.chain_complete_inference import predict
        from foldmatch.types.api_types import Accelerator, SrcEsmFrom

        def embed(fasta):
            batches = predict(src_stream=fasta, src_from=SrcEsmFrom.fasta, accelerator=Accelerator.cpu,
                              devices=1, batch_size=2)
            return {name: emb for embs, names in batches for emb, name in zip(embs, names)}

        clean, messy = embed(_FIXTURE_FASTA), embed(self.messy)
        self.assertEqual(sorted(messy), ["1acb_E", "2uzi_A"])
        for name in clean:
            self.assertTrue(torch.equal(messy[name], clean[name]), name)


if __name__ == "__main__":
    unittest.main()
