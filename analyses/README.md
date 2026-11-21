# Manuscript analysis code

This directory contains the compact analysis-code release associated with the DORIAN manuscript. The folders follow the scientific workflow rather than the manuscript figure numbering, which may change during revision.

| Folder | Analysis |
|---|---|
| `S0_cohort` | GTEx cohort summaries, donor split demographics, and missingness |
| `A1_benchmark` | Hide-one and accessible-tissue benchmarks, competing methods, and performance plots |
| `A2_model_behavior` | Sample-matching retrieval statistics, retrieval heatmaps, and adaptive-mixer behavior |
| `A3_age` | Recovery of gene-wise age associations under simulated missingness |
| `A4_hallmark` | Hallmark gene-set recovery and enrichment plots |
| `A5_efficiency_scaling` | Epoch convergence and training-donor scaling analyses |
| `A6_tabula` | Documentation for the formal within-Tabula Sapiens evaluation |
| `A7_resource` | Completed-resource construction, audit, and production QC |

## Data boundary

The scripts document the analyses but do not redistribute controlled or large inputs, trained checkpoints, generated prediction matrices, or manuscript outputs. The completed GTEx Analysis V11 resource is available through the associated Zenodo record described in [`../data/README.md`](../data/README.md). Original GTEx data remain subject to GTEx data-access terms.

## Environment-specific scripts

Personal cluster launch, status, and file-transfer helpers are intentionally excluded because they contain institution-specific hosts, usernames, and filesystem paths. The scientific computation and plotting scripts are retained. Some analyses therefore require the user to provide equivalent input files and paths in their own environment.

Each subdirectory includes a README describing its inputs, outputs, protocol, and evidence boundary. Generated `outputs/` directories are not tracked in this compact repository.
