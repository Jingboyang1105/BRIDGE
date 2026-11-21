# A3 — Age-association recovery

**Question:** Do reconstructed profiles preserve gene-wise donor-age coefficients under increasing target-tissue masking?

`01_run_all_eligible_formal.py` is the formal 49-tissue workflow. The two `03_plot_*` scripts generate the missingness curve and per-tissue comparison. Tests are copied under `tests/`. Compact formal manifests and summaries are under `outputs/source_data/`; large per-gene NPZ arrays are not duplicated.

**Evidence state:** `rebuildable` from authorized inputs and the formal result directory; selected frozen Figure 4 and Table 2 are also under `paper/`.
