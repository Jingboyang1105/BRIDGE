# A5 — Training efficiency and data scaling

**Question:** How quickly do BRIDGE and HYFA reach validation-MAE thresholds, and how does hide-one performance change with training-set size?

Only the corrected hide-one scaling run is included. The superseded run and all random-known m=5 scaling results are excluded.

- `03_plot_epoch_convergence.py`: validation convergence panels.
- `01_run_hide_one_training_size_scaling.py`: historical formal scaling workflow, preserved as run.
- `03_plot_hide_one_training_size_scaling.py`: corrected scaling panels and source data.

**Evidence state:** aggregate figures and tables are `rebuildable`; full retraining is `external_data_required`.
