"""A structure file that cannot be read is refused (RefusalError, a ValueError), not a RuntimeError.

Callers such as the embedding calculator web service map ValueError to a client error (HTTP 400); a RuntimeError
became a server error (HTTP 500) for files that were simply malformed. The fixtures are small mmCIF texts, read from
streams as that service does. ESM3 is never loaded."""
import io
import unittest

from foldmatch.errors import RefusalError
from foldmatch.utils import structure_parser

HEADER = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""
ROWS = [
    "ATOM 1 N N . GLY {asym} 1 1 ? 47.270 37.190 29.570 1.00 0.00 1 GLY {asym} N 1",
    "ATOM 2 C CA . GLY {asym} 1 1 ? 48.280 37.630 29.380 1.00 0.00 1 GLY {asym} CA 1",
    "ATOM 3 C C . GLY {asym} 1 1 ? 47.270 36.080 29.620 1.00 0.00 1 GLY {asym} C 1",
]


def _cif(asym: str) -> io.StringIO:
    return io.StringIO(HEADER + "\n".join(row.format(asym=asym) for row in ROWS) + "\n#\n")


class StructureReadErrorTest(unittest.TestCase):
    def test_readable_file(self):
        structure = structure_parser.get_structure_from_src(_cif("'seg 0 water'"), "mmcif")
        self.assertEqual(structure.array_length(), 3)

    def test_unquoted_value_with_spaces_is_refused(self):
        # A chain ID with spaces written without quotes: each row has more values than the loop has columns
        with self.assertRaises(RefusalError) as refused:
            structure_parser.get_structure_from_src(_cif("seg 0 water"), "mmcif")
        self.assertIsInstance(refused.exception, ValueError)
        self.assertTrue(str(refused.exception).startswith("Error reading structure from "))
        self.assertIsNotNone(refused.exception.__cause__)

    def test_unknown_format_is_refused(self):
        with self.assertRaises(RefusalError):
            structure_parser.get_structure_from_src(_cif("A"), "xyz")

    def test_unreadable_assemblies_are_refused(self):
        # No pdbx_struct_assembly category to list the assemblies from
        with self.assertRaises(RefusalError) as refused:
            structure_parser.get_assemblies(_cif("A"), "mmcif")
        self.assertTrue(str(refused.exception).startswith("Error reading assemblies from "))
        self.assertIsNotNone(refused.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
