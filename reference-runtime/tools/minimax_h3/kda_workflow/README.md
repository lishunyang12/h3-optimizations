# H3 SM120 kernel candidate ledger

This directory is a small, project-local workflow for deciding whether an
experimental MiniMax-H3 kernel deserves production integration.  It borrows
the useful idea of separating hypotheses, measurements, profiler evidence,
and promotion decisions, but the records and wording here are original.

The source of truth is split deliberately:

- `candidates.jsonl` records one decision object per candidate;
- `benchmarks.csv` stores comparable exact-shape measurements;
- `candidate-template.json` is copied when starting a new experiment;
- `candidate.schema.json` documents the record format;
- `check_ledger.py` validates IDs, evidence links, arithmetic, and gates.

Run the checker from the repository root:

```bash
python tools/minimax_h3/kda_workflow/check_ledger.py
```

## Promotion rule

A candidate may be marked `qualified` only when all of the following are true:

1. its declared production shape and precision were tested;
2. the required correctness mode passed (the current lossless lane requires
   bitwise equality);
3. an interleaved benchmark saves at least `0.2 ms/layer` at the candidate's
   exact boundary;
4. an NCU report is attached and its review passes, with no unexplained spill,
   serialization, or occupancy regression;
5. production dispatch is still kept behind a fail-closed feature gate until a
   block/step/E2E A/B passes.

NCU is a *promotion* gate, not a ritual.  A correctness or isolated-speed
failure is recorded as `not_run_early_stop`; profiling that candidate is not
required.  A profiler result alone can never override a failed wall-time gate.

## Starting a candidate

Copy `candidate-template.json`, choose a stable `candidate_id`, add a matching
CSV measurement after every exact-shape run, and append the completed compact
JSON object to `candidates.jsonl`.  Keep raw JSON/NCU artifacts outside this
ledger and reference their paths; do not paste profiler dumps into the JSONL.

