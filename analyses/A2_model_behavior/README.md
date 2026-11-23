# A2 — Model behavior

**Question:** What structure does BRIDGE learn, which tissues are retrieved, and how does the mixer weight its experts?

The active scripts generate the hide-one retrieval-composition heatmap, formal hide-one statistics and mixer diagnostics. The historical embedding workflow requires the original runtime and is preserved under `baselines/`. `outputs/source_data/hide_one_retrieval_manifest.json` identifies the formal 200-epoch checkpoint and hide-one source.

Only hide-one retrieval evidence is citable. Random-known m=5 retrieval analyses are excluded.

**Evidence state:** `frozen_result`. The Figure 3b statistics have been recomputed from the included formal hide-one source and are recorded in `outputs/tables/hide_one_retrieval_statistics.json`; the historical embedding workflow still requires the original runtime.
