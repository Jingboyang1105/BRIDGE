"""Data loading and high-variance gene-set construction."""

from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
DATA_DIR = REPO_ROOT / "data"
TPM_FILE = DATA_DIR / "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.gct.gz"
VAR_FILE = DATA_DIR / "gene_variances.pkl"


@dataclass(frozen=True)
class GeneSetData:
    expr: np.ndarray
    meta: pd.DataFrame
    tissue_encoder: dict
    gene_set: int
    cache_file: Path


def cache_path(root, genes):
    root = Path(root)
    data_dir = root if root.name == "data" else root / "data"
    return data_dir / f"gtex_phase4_{int(genes)}genes.pkl"


def load_gene_set(genes, root=None, build_if_missing=True):
    root = Path(root) if root is not None else REPO_ROOT
    path = cache_path(root, genes)
    compat = compat_dir(root, genes)
    if compat.exists():
        return load_compat_gene_set(genes, root)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(path)
        build_gene_set_cache(genes, root=root)
    try:
        with path.open("rb") as f:
            loaded = pickle.load(f)
    except ModuleNotFoundError as exc:
        if "numpy._core" in str(exc) and compat.exists():
            return load_compat_gene_set(genes, root)
        raise
    expr, meta, tissue_encoder = loaded[:3]
    return GeneSetData(
        expr=np.asarray(expr, dtype=np.float32),
        meta=meta,
        tissue_encoder=tissue_encoder,
        gene_set=int(genes),
        cache_file=path,
    )


def compat_dir(root, genes):
    root = Path(root)
    data_dir = root if root.name == "data" else root / "data"
    return data_dir / f"gtex_phase4_{int(genes)}genes_compat"


def load_compat_gene_set(genes, root=None):
    root = Path(root) if root is not None else REPO_ROOT
    directory = compat_dir(root, genes)
    expr = np.load(directory / "expr.npy", mmap_mode=None).astype(np.float32)
    meta = pd.read_csv(directory / "meta.csv", index_col=0)
    with (directory / "tissue_encoder.json").open("r", encoding="utf-8") as f:
        tissue_encoder = json.load(f)
    tissue_encoder = {k: int(v) for k, v in tissue_encoder.items()}
    return GeneSetData(expr=expr, meta=meta, tissue_encoder=tissue_encoder, gene_set=int(genes), cache_file=directory)


def write_compat_gene_set(genes, root=None):
    data = load_gene_set(genes, root=root, build_if_missing=False)
    directory = compat_dir(root or REPO_ROOT, genes)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "expr.npy", data.expr.astype(np.float32))
    data.meta.to_csv(directory / "meta.csv")
    with (directory / "tissue_encoder.json").open("w", encoding="utf-8") as f:
        json.dump(data.tissue_encoder, f, indent=2, sort_keys=True)
    return directory


def selected_gene_ids(genes, variance_file=VAR_FILE):
    with Path(variance_file).open("rb") as f:
        variances = pickle.load(f)
    return set(sorted(variances, key=variances.get, reverse=True)[: int(genes)])


def load_raw_tpm_gene_set(genes, root=None, tpm_file=None, variance_file=None, build_if_missing=True):
    """Load raw TPM values for the same high-variance genes used by log caches.

    The phase4 caches store log1p(TPM). HYFA baselines should instead receive
    the selected genes on the raw TPM scale while keeping the exact same samples
    and donor split protocol.
    """
    root = Path(root) if root is not None else REPO_ROOT
    data = load_gene_set(genes, root=root, build_if_missing=build_if_missing)
    tpm_file = Path(tpm_file) if tpm_file is not None else (root / "data" / TPM_FILE.name)
    variance_file = Path(variance_file) if variance_file is not None else (root / "data" / VAR_FILE.name)
    selected = selected_gene_ids(genes, variance_file=variance_file)
    sample_ids = list(data.meta.index)
    keep = {"Name", "Description"} | set(sample_ids)
    opener = gzip.open if str(tpm_file).endswith(".gz") else open

    rows = []
    gene_ids = []
    with opener(tpm_file, "rt") as f:
        f.readline()
        f.readline()
        for chunk in pd.read_csv(f, sep="\t", chunksize=200, low_memory=False, usecols=lambda c: c in keep):
            chunk.set_index("Name", inplace=True)
            chunk.drop(columns=["Description"], errors="ignore", inplace=True)
            sel = chunk.index.isin(selected)
            if sel.any():
                selected_chunk = chunk.loc[sel, sample_ids]
                gene_ids.extend(selected_chunk.index.tolist())
                rows.append(selected_chunk.to_numpy(dtype=np.float32, copy=True))

    if not rows:
        raise ValueError(f"No selected genes were found in {tpm_file}")
    raw = np.concatenate(rows, axis=0).T.astype(np.float32, copy=False)
    return GeneSetData(
        expr=raw,
        meta=data.meta,
        tissue_encoder=data.tissue_encoder,
        gene_set=int(genes),
        cache_file=tpm_file,
    ), gene_ids


def build_gene_set_cache(genes, root=None):
    """Build missing high-variance cache by delegating to phase4 dataset code.

    The legacy phase4 loader already streams the v11 GCT and attaches metadata.
    This wrapper keeps the new project interface clean while reusing the proven
    cache builder.
    """
    root = Path(root) if root is not None else REPO_ROOT
    if root != REPO_ROOT:
        raise ValueError("Building caches is supported only from the repository root data layout.")
    import importlib.util

    builder = REPO_ROOT / "phase4" / "dataset.py"
    if not builder.exists():
        raise FileNotFoundError(
            f"No preprocessed cache found at {compat_dir(root, genes)}. "
            "The raw-GCT cache builder is not part of this release; prepare the "
            "cache as described in data/README.md."
        )
    spec = importlib.util.spec_from_file_location("phase4_dataset", builder)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load_all_metadata(cache=True, n_genes=int(genes))


