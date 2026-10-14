"""P4: esm-codes E-value statistics (S10, S11; ``foldmatch.search.code_statistics``).

F4 (``evalue_grid_v1.json``) holds lambda, lnK, lnE, E, bits and the printed bits from the research table
``ship_final.json`` (md5 c51735f9...) on a grid of query lengths (below the first centre, at each centre, between
centres, above the last), database sizes and S' values; the plan's tolerance is rel <= 1e-12.
"""
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from foldmatch.search import code_statistics as cs

_RES = Path(__file__).parent / "resources" / "esm_codes"
F4 = json.loads((_RES / "evalue_grid_v1.json").read_text())
REL = 1e-12


def f4_table(**rules):
    t = F4["table"]
    return SimpleNamespace(bin_centres=tuple(t["center_len"]), lam=tuple(t["lam"]), lnK=tuple(t["lnK"]),
                           low_m_rule=rules.get("low_m_rule", "clamp"), high_m_rule=rules.get("high_m_rule", "clamp"))


def _rel(got, want):
    return abs(got - want) / abs(want)


class F4GoldenTest(unittest.TestCase):
    """Every F4 row at rel <= 1e-12: lambda, lnK, lnE, E, bits; the printed bits exactly."""

    def test_rows(self):
        table = f4_table()
        self.assertEqual(len(F4["rows"]), 208)
        worst = 0.0
        for r in F4["rows"]:
            with self.subTest(m=r["m"], N=r["N"], S_prime=r["S_prime"]):
                la, lk = cs.lam_lnk(r["m"], table)
                lne = cs.log_evalue(r["S_prime"], r["m"], r["N"], table)
                bits = cs.bit_score(r["S_prime"], r["m"], table)
                for got, want in ((la, r["lam"]), (lk, r["lnK"]), (lne, r["lnE"]), (bits, r["bits"]),
                                  (math.exp(lne), r["E"])):
                    self.assertLessEqual(_rel(got, want), REL)
                    worst = max(worst, _rel(got, want))
                self.assertEqual(cs.render_bits(bits), r["render_bits"])
        self.assertLessEqual(worst, REL)

    def test_exact_at_the_centres(self):
        table = f4_table()
        rows = [r for r in F4["rows"] if r["at_centre"] is not None]
        self.assertEqual(sorted({r["at_centre"] for r in rows}), list(range(6)))
        for r in rows:
            k = r["at_centre"]
            self.assertEqual(r["m"], table.bin_centres[k])
            self.assertEqual(cs.lam_lnk(r["m"], table), (table.lam[k], table.lnK[k]))

    def test_low_clamp(self):
        table = f4_table()
        rows = [r for r in F4["rows"] if r["region"] == "below_first_centre"]
        self.assertEqual(sorted({r["m"] for r in rows}), [10.0, 32.58])
        for r in rows:
            self.assertEqual(cs.lam_lnk(r["m"], table), (table.lam[0], table.lnK[0]))
        self.assertEqual(cs.lam_lnk(1, table), (table.lam[0], table.lnK[0]))

    def test_high_m_clamp(self):
        """D15: above the last centre lambda and lnK keep the last centre's values; F4's uncapped ln-m
        extrapolation (the rejected alternative) is not used."""
        table = f4_table()
        rows = [r for r in F4["rows"] if r["region"] == "above_last_centre"]
        self.assertEqual(sorted({r["m"] for r in rows}), [594.92, 1000.0, 5000.0])
        for r in rows:
            la, lk = cs.lam_lnk(r["m"], table)
            self.assertEqual((la, lk), (table.lam[-1], table.lnK[-1]))
            self.assertNotEqual(la, r["extrapolated"]["lam"])
            self.assertNotEqual(cs.log_evalue(r["S_prime"], r["m"], r["N"], table), r["extrapolated"]["lnE"])
        self.assertEqual(cs.lam_lnk(30000, table), (table.lam[-1], table.lnK[-1]))

    def test_inside_is_linear_in_ln_m(self):
        table = f4_table()
        c, lam = table.bin_centres, table.lam
        m = math.sqrt(c[2] * c[3])                          # half way in ln m
        la, _ = cs.lam_lnk(m, table)
        self.assertLessEqual(_rel(la, 0.5 * (lam[2] + lam[3])), 1e-12)

    def test_e_equals_m_n_2_to_minus_bits(self):
        table = f4_table()
        for m in (10.0, 32.58437573116228, 71.0, 300.0, 594.9175724688861, 966.0, 5000.0):
            for n in (1.0, 303.0, 3.6e5, 8557236.0, 1e9, 1e11):
                for sp in (-5.0, 0.0, 12.3, 80.0, 250.0):
                    with self.subTest(m=m, N=n, S_prime=sp):
                        e = math.exp(cs.log_evalue(sp, m, n, table))
                        self.assertLessEqual(_rel(e, m * n * 2.0 ** -cs.bit_score(sp, m, table)), REL)

    def test_vectorised_equals_scalar(self):
        table = f4_table()
        m = np.array([10.0, 50.0, 100.0, 1000.0])
        sp = np.array([-5.0, 0.0, 12.3, 80.0])
        lne = cs.log_evalue(sp, m, 1e9, table)
        bits = cs.bit_score(sp, m, table)
        for i in range(4):
            self.assertEqual(lne[i], cs.log_evalue(sp[i], m[i], 1e9, table))
            self.assertEqual(bits[i], cs.bit_score(sp[i], m[i], table))
        np.testing.assert_array_equal(cs.render_bits(bits), [cs.render_bits(b) for b in bits])
        lne_q = cs.log_evalue(sp, 300, 1e9, table)          # one query, many targets
        self.assertEqual(lne_q.shape, (4,))
        self.assertIsInstance(cs.log_evalue(1.0, 300, 1e9, table), float)


class RefusalTest(unittest.TestCase):

    def test_unimplemented_rules(self):
        for rules in ({"high_m_rule": "extrapolate"}, {"low_m_rule": "extrapolate"}, {"high_m_rule": "Clamp"}):
            with self.subTest(**rules), self.assertRaisesRegex(ValueError, "not implemented"):
                cs.lam_lnk(100, f4_table(**rules))

    def test_bad_inputs(self):
        table = f4_table()
        for m in (0, -3, float("nan"), float("inf")):
            with self.subTest(m=m), self.assertRaises(ValueError):
                cs.lam_lnk(m, table)
        for n in (0, -1.0, float("nan")):
            with self.subTest(N=n), self.assertRaisesRegex(ValueError, "database residues"):
                cs.log_evalue(1.0, 100, n, table)
        with self.assertRaises(ValueError):
            cs.log_evalue(float("nan"), 100, 1e9, table)
        with self.assertRaises(ValueError):
            cs.bit_score(float("inf"), 100, table)

    def test_bad_table(self):
        bad = f4_table()
        bad.bin_centres = bad.bin_centres[::-1]
        with self.assertRaises(ValueError):
            cs.lam_lnk(100, bad)
        short = f4_table()
        short.lam = short.lam[:5]
        with self.assertRaises(ValueError):
            cs.lam_lnk(100, short)


class RenderBitsTest(unittest.TestCase):

    def test_floor_half_up(self):
        cases = {-2.5: -2, -0.4: 0, -0.5: 0, -0.6: -1, -1.7: -2, 0.0: 0, 0.49: 0, 0.5: 1, 2.5: 3, 18.01: 18,
                 22.698: 23}
        for x, want in cases.items():
            with self.subTest(bits=x):
                got = cs.render_bits(x)
                self.assertEqual(got, want)
                self.assertIsInstance(got, int)
        self.assertNotEqual(cs.render_bits(-1.7), int(-1.7 + 0.5))   # the BLOSUM path's rounding differs here

    def test_array_and_refusal(self):
        np.testing.assert_array_equal(cs.render_bits([-2.5, -0.4, 1.5]), [-2, 0, 2])
        self.assertEqual(cs.render_bits(np.array([1.5])).dtype, np.int64)
        with self.assertRaises(ValueError):
            cs.render_bits(float("nan"))

    def test_range_is_the_same_for_scalars_and_arrays(self):
        """Beyond 2^53 neither path renders (the array path's int64 would saturate at 2^63; P4 review)."""
        self.assertEqual(cs.render_bits(2.0 ** 53), 2 ** 53)
        np.testing.assert_array_equal(cs.render_bits(np.array([-(2.0 ** 53), 2.0 ** 53])), [-(2 ** 53), 2 ** 53])
        for bad in (np.nextafter(2.0 ** 53, np.inf), 1e19, -1e300):
            with self.subTest(bits=bad):
                with self.assertRaisesRegex(ValueError, "2\\^53"):
                    cs.render_bits(bad)
                with self.assertRaisesRegex(ValueError, "2\\^53"):
                    cs.render_bits(np.array([1.0, bad]))


class FormatEvalueTest(unittest.TestCase):

    def test_underflow(self):
        self.assertEqual(cs.format_evalue_from_log(-1000), "5.076E-435")
        self.assertTrue(cs.format_evalue_from_log(-1000).endswith("E-435"))
        self.assertEqual(cs.format_evalue_from_log(-745), "2.822E-324")    # subnormal exp(): formatted from ln E
        self.assertEqual(cs.format_evalue_from_log(-1e6), "3.297E-434295")  # 10^-434294.4819... = 3.29683E-434295

    def test_against_decimal_arithmetic(self):
        from decimal import Decimal, getcontext
        getcontext().prec = 40
        for x in (-709.5, -800.25, -1000.0, -12345.678, -1e6, 750.0, 3000.5):
            t = Decimal(x) / Decimal(10).ln()
            e = math.floor(t)
            want = f"{float(Decimal(10) ** (t - e)):.3f}E{'-' if e < 0 else '+'}{abs(e):02d}"
            self.assertEqual(cs.format_evalue_from_log(x), want, x)

    def test_overflow(self):
        self.assertEqual(cs.format_evalue_from_log(1000), "1.970E+434")
        self.assertEqual(cs.format_evalue_from_log(710), "2.234E+308")

    def test_in_range_equals_the_blosum_format(self):
        rng = np.random.default_rng(20261014)
        for x in [0.0, math.log(1e-3), math.log(12.7), -708.0, 709.0] + rng.uniform(-708, 709, 500).tolist():
            with self.subTest(lnE=x):
                self.assertEqual(cs.format_evalue_from_log(x), f"{math.exp(x):.3E}")

    def test_log10_path_agrees_with_direct_formatting(self):
        rng = np.random.default_rng(7)
        for x in rng.uniform(-700, 700, 2000).tolist():
            self.assertEqual(cs._format_from_log10(x / cs.LN10), f"{math.exp(x):.3E}", x)

    def test_carry_into_the_next_decade(self):
        x = math.log(9.9996) - 500 * math.log(10)
        self.assertEqual(cs.format_evalue_from_log(x), "1.000E-499")
        x = math.log(9.9994) - 500 * math.log(10)
        self.assertEqual(cs.format_evalue_from_log(x), "9.999E-500")

    def test_special_values(self):
        self.assertEqual(cs.format_evalue_from_log(-math.inf), "0.000E+00")
        self.assertEqual(cs.format_evalue_from_log(math.inf), "INF")
        with self.assertRaises(ValueError):
            cs.format_evalue_from_log(float("nan"))


class Q90Test(unittest.TestCase):

    def test_linear_quantile(self):
        self.assertEqual(cs.q90(np.arange(11.0)), 9.0)
        self.assertAlmostEqual(cs.q90([1.0, 2.0, 3.0, 4.0]), 3.7, places=15)
        rng = np.random.default_rng(1)
        s = rng.normal(size=1000)
        self.assertEqual(cs.q90(s), float(np.quantile(s, 0.9)))
        self.assertEqual(cs.q90(s.astype(np.float32)), float(np.quantile(s.astype(np.float32).astype(np.float64),
                                                                          0.9)))
        self.assertEqual(cs.q90(s, quantile=0.5), float(np.median(s)))
        self.assertIsInstance(cs.q90(s), float)

    def test_refusals(self):
        for bad in ([], [[1.0, 2.0]], [1.0, float("nan")]):
            with self.subTest(scores=bad), self.assertRaises(ValueError):
                cs.q90(bad)
        with self.assertRaises(ValueError):
            cs.q90([1.0, 2.0], method="nearest")
        with self.assertRaises(ValueError):
            cs.q90([1.0, 2.0], quantile=1.0)


if __name__ == "__main__":
    unittest.main()
