from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cff"}

FROZEN_IDENTIFIERS = {
    "dorian_gtex_v11_gene_annotations.csv",
    "dorian_all_eligible_predictions.npz",
    "dorian_hide_one_per_tissue_metrics.csv",
    "dorian_validation_curve.csv",
    "hide_one_training_size_scaling_dorian_curve.csv",
    "dorian_full_data_seed42.pt",
    "imputed_dorian.npy",
    "dorian_gtex_v11_completed_resource.h5",
    "dorian_records_20000genes_seed42",
    "03_How_DORIAN_Works",
    "imputed_same_donor_expert.npy",
    '"imputed/dorian"',
    '"imputed/same_donor_expert"',
    '"imputed/same_donor_weight"',
    '"imputed/memory_expert"',
    '"method"] = "DORIAN"',
    '"dorian":',
    '"DORIAN_accessible":',
}

FORBIDDEN = re.compile(
    r"DORIAN|\bfrom\s+dorian\b|\bimport\s+dorian\b|"
    r"same[-_ ]donor expert|memory[-_ ]retrieval expert|memory expert",
    re.IGNORECASE,
)


def test_public_tree_uses_bridge_naming_except_frozen_artifact_contracts():
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path == Path(__file__):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not FORBIDDEN.search(line):
                continue
            if any(identifier in line for identifier in FROZEN_IDENTIFIERS):
                continue
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    assert not failures, "Old public naming remains:\n" + "\n".join(failures)


def test_obsolete_package_and_module_paths_are_absent():
    forbidden_paths = [
        ROOT / "dorian",
        ROOT / "bridge" / "dorian_model.py",
        ROOT / "bridge" / "train_dorian.py",
        ROOT / "bridge" / "evaluate_dorian.py",
        ROOT / "bridge" / "same_donor_expert.py",
        ROOT / "bridge" / "memory_retrieval_expert.py",
    ]
    assert not [str(path.relative_to(ROOT)) for path in forbidden_paths if path.exists()]
