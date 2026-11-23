# BRIDGE

**BRIDGE (Body-wide Representation via Inter-tissue Decoding Gene Expression)** is a framework for donor-specific and tissue-specific gene-expression imputation.

> **Naming note:** BRIDGE was previously developed under the name DORIAN.
> To preserve backward compatibility and reproducibility, the internal Python
> package, module names, and some code identifiers continue to use `dorian`.

This repository is the compact software and analysis-code release. It contains the core implementation, manuscript analysis scripts, a minimal API example, and focused correctness tests. Restricted GTEx inputs, trained checkpoints, large prediction artifacts, and generated analysis outputs are intentionally excluded.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate dorian
python -m pip install -e .
```

Or install into an existing Python 3.11+ environment:

```bash
python -m pip install -e .
```

To install the additional plotting and scientific-analysis dependencies:

```bash
python -m pip install -e ".[analysis]"
```

## Minimal example

```bash
python -m examples.basic_prediction
```

The example demonstrates the public two-expert combination interface with small synthetic arrays. It is not a pretrained biological prediction.

```python
import numpy as np
from dorian import predict_missing_tissue

prediction = predict_missing_tissue(
    same_donor_prediction=np.array([1.2, 2.0, 3.1]),
    memory_prediction=np.array([1.5, 1.5, 3.2]),
    mixer_weights=np.array([0.6, 0.4]),
)
```

## Reproducing the manuscript model

After preparing the input cache described in [`data/README.md`](data/README.md), the manuscript hide-one model (20,000 genes, seed 42) corresponds to:

```bash
python -m dorian.train_dorian --genes 20000 --seed 42 --m 5   --train_source hide_one_only --epochs 200 --batch 256 --mixer_epochs 50
```

The default command-line values are smaller settings for quick runs and do not reproduce the manuscript model.

## Tests

```bash
python -m pytest -q
```

## Repository layout

```text
dorian/       Core model, training, evaluation, and prediction code
analyses/     Manuscript analysis and figure-generation scripts
examples/     Minimal public-API example
tests/        Focused correctness tests
data/         Data-availability information only
```

See [`analyses/README.md`](analyses/README.md) for the correspondence between analysis folders and manuscript components.

## Data and reproducibility boundary

This repository does not contain GTEx donor-level matrices, credentials, trained checkpoints, or the completed 10.4-GB HDF5 resource. See [`data/README.md`](data/README.md) for the associated reserved Zenodo record and data-access boundary.

Generated manuscript figures, tables, and distributable data resources are maintained separately from this compact code repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The associated manuscript citation should be added when it is publicly available.

## License

An open-source license has not yet been approved by all authors. Until that decision is made, the terms in [`LICENSE`](LICENSE) apply.

