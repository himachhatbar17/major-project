#!/usr/bin/env python
"""Stage 5: CASIA-B sequence-level metadata preparation. No training or source edits."""
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description="Prepare sequence-level CASIA-B metadata; no training.")
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--output-dir", default="Stage 5")
    a=p.parse_args()
    start=time.time(); s2=Path(a.stage2_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    logs=[]
    def log(x): print(x); logs.append(x)
    log("="*78); log("STAGE 5 — CASIA-B SEQUENCE-LEVEL DATA PREPARATION"); log("="*78)
    log(f"Stage 2 dir: {s2}"); log(f"Output dir: {out}"); log("Training: DISABLED | Source images modified: NO")
    mp=s2/"stage2_manifest_validated.csv"
    if not s2.is_dir(): raise NotADirectoryError(f"Stage 2 directory not found: {s2}")
    if not mp.is_file(): raise FileNotFoundError(f"Validated manifest not found: {mp}")
    df=pd.read_csv(mp)
    req=["subject_id","view","condition","sequence","sequence_path","frame_count","first_frame","last_frame","split"]
    missing=[c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    checks={
      "manifest_rows_13592":len(df)==13592,
      "no_missing_required_values":not df[req].isna().any().any(),
      "positive_frame_counts":bool((pd.to_numeric(df.frame_count,errors="coerce")>0).all()),
      "no_duplicate_sequence_keys":not df.duplicated(["subject_id","view","sequence"]).any(),
      "one_split_per_subject":bool(df[["subject_id","split"]].drop_duplicates().groupby("subject_id").split.nunique().le(1).all())
    }
    log("\n[1/4] Metadata checks")
    for k,v in checks.items(): log(f"{'PASS' if v else 'FAIL'}: {k} = {v}")
    seq=df[req].copy()
    seq["subject_id"]=seq.subject_id.astype(str).str.zfill(3)
    seq["view"]=seq.view.astype(str).str.zfill(3)
    seq["frame_count"]=pd.to_numeric(seq.frame_count,errors="coerce").astype("Int64")
    seq=seq.sort_values(["split","subject_id","condition","sequence","view"],kind="stable").reset_index(drop=True)
    seq_path=out/"stage5_sequence_manifest.csv"; seq.to_csv(seq_path,index=False)
    log(f"\n[2/4] Sequence manifest: {len(seq)} rows"); log(f"Saved: {seq_path}")
    dist=(seq.groupby(["split","condition"])["frame_count"].agg(sequence_count="count",min_frames="min",median_frames="median",mean_frames="mean",max_frames="max",std_frames="std").reset_index())
    dist_path=out/"stage5_frame_count_distribution.csv"; dist.to_csv(dist_path,index=False)
    log("\n[3/4] Frame-count distribution"); log(dist.to_string(index=False)); log(f"Saved: {dist_path}")
    subjects=seq[["subject_id","split"]].drop_duplicates()
    split_counts=subjects.split.value_counts().to_dict()
    expected={"train":64,"val":10,"test":50}
    split_ok=all(int(split_counts.get(k,0))==v for k,v in expected.items())
    disjoint=bool(subjects.groupby("subject_id").split.nunique().le(1).all())
    report={"stage":5,"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"manifest_rows":len(df),"sequence_records":len(seq),"checks":checks,"subjects_by_split":{str(k):int(v) for k,v in split_counts.items()},"split_counts_pass":split_ok,"subject_disjoint_pass":disjoint,"frame_count_distribution":dist.to_dict(orient="records"),"sequence_length_policy":"Not fixed; inspect frame-count distribution before choosing","source_images_modified":False,"training_performed":False,"elapsed_seconds":round(time.time()-start,2)}
    rp=out/"stage5_sequence_preparation_report.json"; rp.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    ok=all(checks.values()) and split_ok and disjoint
    log("\n[4/4] Final status"); log(f"Subjects by split: {split_counts}"); log(f"Split counts pass: {split_ok}"); log(f"Subject-disjoint: {disjoint}"); log(f"Report saved: {rp}"); log(f"Status: {'PASS' if ok else 'REVIEW'}"); log("No training; source images unchanged.")
    (out/"stage5_console_log.txt").write_text("\n".join(logs)+"\n",encoding="utf-8")
    if not ok: sys.exit(2)
if __name__=="__main__":
    try: main()
    except Exception as e: print(f"\nSTAGE 5 FAILED\n{type(e).__name__}: {e}"); sys.exit(1)
