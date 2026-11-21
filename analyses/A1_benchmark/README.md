# A1 — Core benchmark

**Question:** How accurately do DORIAN, HYFA, PMM, TEEBoT and ablations reconstruct hidden tissues in hide-one and accessible-tissue settings?

## Outputs

- frozen aggregate and per-tissue tables: `outputs/tables/`
- figure source data: `outputs/source_data/`
- selected code-generated panels: `outputs/figures/`
- selected submission artifacts: `paper/figures/`, `paper/tables/`, and supplements

## Level A

`02_compute_metrics_from_aligned_arrays.py` calculates publication metrics from explicit, aligned truth/prediction arrays, records and training-derived tissue means using only the released clean metrics module. `02_compute_gene_residual_pearson.py` is the retained gene-table workflow. `03_plot_*` scripts generate individual panels. Multi-panel page layout was completed in Inkscape/Illustrator and is not claimed as script-generated.

## Level B

Historical DORIAN/HYFA summaries and HYFA, PMM and TEEBoT workflows are preserved in `baselines/`. Remote tissue-identity checks are in `remote/`. Their scientific logic has not been rewritten.

**Evidence state:** `frozen_result_pending_provenance`. Formal DORIAN prediction equivalence and the accessible HYFA checkpoint hash remain open in `SCIENTIFIC_GATES.md`.
