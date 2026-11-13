# Benchmark records

- `rtx4090_herg.json`: original 2026-11-13 measurements behind the README's
  large-scale extrapolations. Use `teacher` and `cp16_distinct` for the table.
  The older `cp16` field timed repeated reuse of one tile; the distinct sweep
  supersedes that measurement. `cp16_distinct.with_d2h_seconds` includes copying
  every float32 score to CPU RAM, but does not retain the complete output table.
- `herg_two_slot_factors.npz`: exact float16 factors used by the distinct
  32,768 × 29,100 matrix sweep; `herg_two_slot_factors.json` records provenance.
  These are learned vectors only, not molecular structures or vendor IDs.
- `demo_rdkit_*.json`: measured small-example CPU preparation and training.
- `demo_herg_*.json`: measured small-example GPU preparation and training using
  the pinned downloaded OpenADMET hERG model in a fresh environment.
- `adapter_smoke.json`: two-molecule encoding/prediction smoke checks for the nine
  pinned repositories, using their local model weights and the pinned adapter.
- `reproduction_*.json`: the fresh installation's dependency versions and
  subsequent runs of the portable GPU benchmark (about 9 minutes for 20 trillion
  scores by kernel-only extrapolation; timing varies under concurrent GPU load).
- `paper_model_summary.csv`: original research metrics across architectures and
  ranks. Filter `architecture=cp`, `rank=16` to inspect the 18 CP16 outputs.
  They come from a different, larger source library and product split than
  the bundled demo. The figure is not a claim about this demo's held-out scores.

All extrapolations use `time = number_of_products * seconds / measured_count`.
The README uses 365.25 days per year. The original GPU was shared with another
active job; timing varies with contention, batching, hardware and library shape.

The original teacher used 4,953 assembled products and batch size 256.
The small demo's products are different. Its teacher benchmarking script
reproduces the timing method, not the original chemical input distribution.
The CP16 matrix fixture reproduces the exact numerical inputs for its kernel.
Neither benchmark includes teacher training, student fitting, cache creation,
product assembly, ranking or disk persistence in the scoring measurements.
