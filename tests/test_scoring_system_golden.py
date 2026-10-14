"""P4: golden digests of every registered esm-codes scoring system (plan section 8 P4, acceptance).

The registry froze on 2026-10-14, after P0.6, P0.7 and the bundle upload. From here on any change of a registered
system fails this test: a different value is a new system with a new name and its own golden line (D17), never an
edit of an existing entry. ``digest``: every field; ``encoding_digest``: ``ENCODING_FIELDS``; ``canonical_sha256``:
sha256 of ``to_json()``.

The module needs only the standard library and ``foldmatch.search.scoring_system`` (which imports neither numpy nor
torch), so CI runs it in the kernel jobs too, on Linux x86_64, Linux arm64 and macOS arm64: the plan's "digests
identical on Linux and macOS CI".
"""
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from foldmatch.search import scoring_system as ss

GOLDEN = {
    "esm-codes-v1": {
        "digest": "81de063d13a7ef66",
        "encoding_digest": "92220d870638cb57",
        "canonical_sha256": "3c81bbceac34688135262daa92d06544b87d4a20bbdddba3fb354b14e44cfad3",
    },
}

#: Run in a fresh interpreter with every non-stdlib dependency of FoldMatch blocked: the digests need none of them.
_FRESH = """
import sys
for blocked in ("numpy", "torch", "huggingface_hub", "typer", "esm", "faiss", "biotite", "lightning"):
    sys.modules[blocked] = None
import hashlib
from foldmatch.search import scoring_system as ss
for name, s in sorted(ss.REGISTRY.items()):
    print(name, s.digest(), s.encoding_digest(), hashlib.sha256(s.to_json().encode()).hexdigest())
"""


def _env(**extra):
    env = dict(os.environ)
    src = str(Path(ss.__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


class RegistryGoldenTest(unittest.TestCase):

    def test_golden_digests(self):
        self.assertEqual(sorted(ss.REGISTRY), sorted(GOLDEN))
        for name, want in GOLDEN.items():
            s = ss.get(name)
            with self.subTest(system=name):
                self.assertEqual(s.digest(), want["digest"])
                self.assertEqual(s.encoding_digest(), want["encoding_digest"])
                self.assertEqual(hashlib.sha256(s.to_json().encode()).hexdigest(), want["canonical_sha256"])
                self.assertEqual(ss.EsmCodesSystem.from_json(s.to_json()), s)

    def test_digests_are_their_definition(self):
        """16 hex of sha256(tag + newline + canonical JSON): the definition, independent of the methods."""
        for name, want in GOLDEN.items():
            s = ss.get(name)
            d = s.to_dict()
            for tag, obj, key in ((ss.SYSTEM_DIGEST_TAG, d, "digest"),
                                  (ss.ENCODING_DIGEST_TAG, {k: d[k] for k in ss.ENCODING_FIELDS}, "encoding_digest")):
                text = ss.canonical_json(obj)
                self.assertTrue(text.isascii())
                got = hashlib.sha256(tag.encode() + b"\n" + text.encode()).hexdigest()[:16]
                self.assertEqual(got, want[key], f"{name} {key}")

    def test_digests_in_fresh_processes_without_numpy_or_torch(self):
        want = [f"{n} {g['digest']} {g['encoding_digest']} {g['canonical_sha256']}" for n, g in sorted(GOLDEN.items())]
        for seed in ("0", "12345"):
            with self.subTest(PYTHONHASHSEED=seed):
                r = subprocess.run([sys.executable, "-c", _FRESH], env=_env(PYTHONHASHSEED=seed), capture_output=True,
                                   text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertEqual(r.stdout.strip().splitlines(), want)


if __name__ == "__main__":
    unittest.main()
