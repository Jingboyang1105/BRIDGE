from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .engine import run_tissue
from .inputs import align_matrix
from .protocol import sha256_file

PROJECT=Path(__file__).resolve().parents[3]
EXPERIMENT=Path(__file__).resolve().parents[1]

def bundle(path,key,ids):
    z=np.load(path,allow_pickle=True); source=z["record_ids"].astype(str)
    return align_matrix(z["y_true"],source,ids),align_matrix(z[key],source,ids)

def main() -> int:
    lock=EXPERIMENT/"results/all_eligible_protocol_lock"; raw=EXPERIMENT/"results/all_eligible_raw_predictions"; out=EXPERIMENT/"results/all_eligible_formal_seed42"
    if not (lock/"LOCKED").exists() or not (raw/"COMPLETE").exists(): raise RuntimeError("locked inputs and downloads required")
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False)
    records=pd.read_csv(lock/"all_eligible_records.csv"); demo=pd.read_csv(lock/"all_eligible_demographics.csv"); tissues=pd.read_csv(lock/"all_eligible_tissues.csv")
    ids=records.record_id.astype(str).tolist()
    truth,dorian=bundle(raw/"dorian_all_eligible_predictions.npz","pred_two_expert_cml_gate",ids)
    htruth,hyfa=bundle(raw/"hyfa_all_eligible_no_demographics_predictions.npz","pred_HYFA_no_demographics",ids)
    if not np.array_equal(truth,htruth): raise ValueError("neural truth mismatch")
    p=PROJECT/"results/main_pmm_v11_seed42/formal_olga/hide_one"; pids=pd.read_csv(p/"records.csv").record_id.astype(str).tolist()
    ptruth=align_matrix(np.load(p/"truth.npy",mmap_mode="r"),pids,ids); pmm=align_matrix(np.load(p/"predictions.npy",mmap_mode="r"),pids,ids)
    if not np.allclose(truth,ptruth,atol=1e-6,rtol=0): raise ValueError("PMM truth mismatch")
    if not all(np.isfinite(x).all() for x in [truth,dorian,hyfa,pmm]): raise ValueError("non-finite formal inputs")
    summaries=[]; masks=[]; stats=out/"gene_statistics"; stats.mkdir()
    names=tissues.set_index("target_tissue").target_tissue_name.to_dict()
    for tissue in tissues.target_tissue.astype(int):
        take=records.target_tissue.to_numpy(int)==tissue
        summary,arrays,tmasks=run_tissue(tissue,str(names[tissue]),truth[take],{"DORIAN":dorian[take],"HYFA":hyfa[take],"PMM":pmm[take]},demo.loc[take].reset_index(drop=True),[.2,.4,.6,.8],5,42)
        summaries.append(summary); masks.append(tmasks); np.savez_compressed(stats/f"tissue_{tissue}_age_statistics.npz",**arrays)
    summary=pd.concat(summaries,ignore_index=True); mask=pd.concat(masks,ignore_index=True)
    summary.to_csv(out/"age_recovery_endpoints.csv",index=False); mask.to_csv(out/"mask_manifest.csv",index=False)
    per_tissue=summary.groupby(["tissue","tissue_name","arm"],as_index=False)[["beta_pearson","beta_slope","beta_mae","sign_agreement","top_100_recall","top_500_recall","top_1000_recall"]].mean()
    per_tissue.to_csv(out/"age_recovery_per_tissue_summary.csv",index=False)
    manifest={"records":len(records),"genes":20000,"tissues":len(tissues),"endpoint_rows":len(summary),"mask_rows":len(mask),"expected_endpoint_rows":49*4*5*5,
              "input_sha256":{"records":sha256_file(lock/"all_eligible_records.csv"),"dorian":sha256_file(raw/"dorian_all_eligible_predictions.npz"),"hyfa":sha256_file(raw/"hyfa_all_eligible_no_demographics_predictions.npz"),"pmm":sha256_file(p/"predictions.npy")}}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); (out/"COMPLETE_PRIMARY").write_text("49-tissue primary complete\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
