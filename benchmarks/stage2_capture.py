#!/usr/bin/env python
"""Run ``fm-search ARGS...`` (``foldmatch.cli.search.main``, what the console script calls) and save the exact inputs
of its esm-codes Stage 2 to ``$FM_CAPTURE_DIR`` for ``stage2_esm_codes.py replay`` (plan section 8 P7, profiling
gate on the real pipeline: real Stage-1 candidates, so the real E-filter survivor fraction).

Saved: ``queries.f16`` (a copy of the run's query-encoding memmap, before the CLI deletes it) and ``capture.json``:
its entries (offset, length, seq_sha1), the Stage-1 candidates ``{query: [[ids], [embedding scores]]}`` exactly as
Stage 2 received them, the query sequences, N, the code store's ``.bin`` path, the keyword options of the call (the
output fields as a list), the hits' signature (``stage2_esm_codes.signature``) and the Stage-2 ``stats`` (with the
profile when ``FOLDMATCH_STAGE2_PROFILE=1``); and the wall time of the CLI's phases ``stage2`` and ``output``
(``output.write_aligned_results``). The wrappers pass arguments and results through unchanged. Several calls in
one run (not the CLI's case) get ``capture.json``, ``capture_1.json``, ...
"""
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    out = Path(os.environ["FM_CAPTURE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    import foldmatch.cli.search as cli
    import foldmatch.search.code_alignment as ca
    import foldmatch.search.output as output
    from stage2_esm_codes import signature

    rec = dict(argv=sys.argv[1:], phases={}, calls=0, import_s=time.perf_counter() - T0)
    orig_s2, orig_out = ca.align_candidates_esm_codes, output.write_aligned_results

    def stage2(query_index, prefilter_results, code_store, fetch, prepared, db_residues, pool, **kw):
        n = rec["calls"]
        rec["calls"] += 1
        suffix = "" if n == 0 else f"_{n}"
        shutil.copyfile(query_index.path, out / f"queries{suffix}.f16")
        if kw.get("stats") is None:
            kw["stats"] = {}
        t0 = time.perf_counter()
        res = orig_s2(query_index, prefilter_results, code_store, fetch, prepared, db_residues, pool, **kw)
        dt = time.perf_counter() - t0
        opts = {}
        for k, v in kw.items():
            if k in ("stats", "query_sequences"):
                continue
            if k == "output_needs" and not isinstance(v, (list, tuple)):
                v = dataclasses.asdict(v)
            opts[k] = list(v) if isinstance(v, tuple) else v
        cap = dict(
            rows=query_index.rows, width=query_index.width, system_name=getattr(query_index, "system_name", ""),
            entries={q: [int(e.offset), int(e.length), bytes(e.seq_sha1).hex()] for q, e in query_index.entries.items()},
            prefilter_results={q: [list(v[0]), [float(s) for s in v[1]]] for q, v in prefilter_results.items()},
            query_sequences=dict(kw.get("query_sequences") or {}), db_residues=int(db_residues),
            code_store_bin=str(code_store.bin_path), pool=dict(repr=repr(pool), workers=getattr(pool, "workers", None)),
            options=opts, stage2_s=dt, signature=signature(res), hits=sum(len(v) for v in res.values()),
            stats=kw["stats"])
        (out / f"capture{suffix}.json").write_text(json.dumps(cap, default=str))
        rec["phases"].setdefault("stage2", []).append(dt)
        return res

    def write(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_out(*args, **kwargs)
        finally:
            rec["phases"].setdefault("output", []).append(time.perf_counter() - t0)

    ca.align_candidates_esm_codes, output.write_aligned_results = stage2, write
    sys.argv = ["fm-search"] + sys.argv[1:]
    rc = 0
    try:
        cli.main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        raise
    finally:
        rec["rc"] = rc
        rec["total_s"] = time.perf_counter() - T0
        (out / "cli_phases.json").write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
