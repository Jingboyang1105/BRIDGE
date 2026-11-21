# DORIAN

DORIAN (**Donor Residual Imputation by Aligned Neighbors**) is a lightweight contrastive-learning framework for donor-specific and tissue-specific gene-expression imputation. It combines a donor-level average expert with a sample-matching expert through an adaptive mixer.

This repository is the compact software release. It contains the core implementation, a minimal API example, and focused correctness tests. Manuscript-specific analyses, figures, restricted GTEx inputs, trained checkpoints, and large prediction artifacts are intentionally excluded.

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
examples/     Minimal public-API example
tests/        Focused correctness tests
data/         Data-availability information only
```

## Data and reproducibility boundary

This repository does not contain GTEx donor-level matrices, credentials, trained checkpoints, or the completed 10.4-GB HDF5 resource. See [`data/README.md`](data/README.md) for the associated reserved Zenodo record and data-access boundary.

The full manuscript analysis archive is maintained separately from this compact software repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The associated manuscript citation should be added when it is publicly available.

## License

An open-source license has not yet been approved by all authors. Until that decision is made, the terms in [`LICENSE`](LICENSE) apply.
