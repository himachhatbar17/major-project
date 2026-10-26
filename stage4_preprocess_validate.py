#!/usr/bin/env python
"""Stage 4: conservative preprocessing specification and deterministic loading smoke test. No training."""
import argparse, hashlib, json, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

VIEWS={"000","018","036","054","072","090","108","126","144","162","180"}
CONDS={"normal","bag","coat"}

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument("--stage2-dir",required=True)
    p.add_argument("--stage3-report",required=True)
    p.add_argument("--output-dir",default="Stage 4")
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--smoke-test-frames",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

def load_array(path):
    # Preserve intensity values after grayscale conversion; no threshold/crop/resize.
    with Image.open(path) as im:
        return np.asarray(im.convert("L"),dtype=np.uint8)

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    a=args_parse(); start=time.time()
    s2=Path(a.stage2_dir).resolve(); s3=Path(a.stage3_report).resolve(); out=Path(a.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True); logs=[]
    def log(s): print(s); logs.append(s)
    log("="*80); log("STAGE 4 — PREPROCESSING SPECIFICATION AND PIPELINE VALIDATION"); log("="*80)
    log(f"Stage 2 dir: {s2}"); log(f"Stage 3 report: {s3}"); log(f"Output dir: {out}")
    log("Training: DISABLED | Source images modified: NO")
    mp=s2/"stage2_manifest_validated.csv"
    if not s2.is_dir(): raise NotADirectoryError(f"Stage 2 directory not found: {s2}")
    if not mp.is_file(): raise FileNotFoundError(f"Validated manifest not found: {mp}")
    if not s3.is_file(): raise FileNotFoundError(f"Stage 3 report not found: {s3}")
    df=pd.read_csv(mp); qc=json.loads(s3.read_text(encoding="utf-8"))
    required=["subject_id","view","condition","sequence","sequence_path","frame_count","first_frame","last_frame","split"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required manifest columns: {missing}")
    checks={
      "manifest_rows_13592":len(df)==13592,
      "expected_conditions_present":CONDS.issubset(set(df.condition.astype(str))),
      "expected_views_present":VIEWS.issubset(set(df.view.astype(str).str.zfill(3))),
      "no_missing_required_values":not df[required].isna().any().any(),
      "positive_frame_counts":bool((pd.to_numeric(df.frame_count,errors="coerce")>0).all()),
      "stage3_checked_1117083_frames":qc.get("images_checked")==1117083,
      "stage3_zero_failed_records":qc.get("failed_or_missing_image_records")==0,
      "stage3_zero_missing_sequence_dirs":qc.get("missing_sequence_directories")==0,
      "stage3_zero_sequences_without_pngs":qc.get("sequences_without_png_frames")==0,
    }
    log("\n[1/4] Input checks")
    for k,v in checks.items(): log(f"{'PASS' if v else 'REVIEW'}: {k} = {v}")
    uniq=df[["subject_id","split"]].drop_duplicates()
    split_counts=uniq.split.value_counts().to_dict()
    split_ok=all(int(split_counts.get(k,0))==v for k,v in {"train":64,"val":10,"test":50}.items())
    disjoint=bool(uniq.groupby("subject_id").split.nunique().le(1).all())
    log("\n[2/4] Split contract"); log(f"Unique subjects by split: {split_counts}")
    log(f"Split counts correct: {split_ok}"); log(f"Subject-disjoint: {disjoint}")
    smoke_ok=None
    if a.smoke_test:
        paths=list(dict.fromkeys(str(x) for x in pd.concat([df.first_frame,df.last_frame]).dropna()))
        rng=random.Random(a.seed); n=min(max(1,a.smoke_test_frames),len(paths))
        chosen=rng.sample(paths,n) if n<len(paths) else paths
        rows=[]
        root=s2.parent
        log(f"\n[3/4] Deterministic load smoke test: {len(chosen)} frames")
        for i,raw in enumerate(chosen,1):
            p=Path(raw); p=p if p.is_absolute() else (root/p).resolve()
            r={"path":str(p),"exists":p.is_file(),"read_ok":False,"shape":"","dtype":"","min":None,"max":None,"unique_values":None,"sha256_first":"","sha256_second":"","deterministic":False,"error":""}
            try:
                x=load_array(p); y=load_array(p); hx,hy=digest(x),digest(y)
                r.update(read_ok=True,shape="x".join(map(str,x.shape)),dtype=str(x.dtype),min=int(x.min()),max=int(x.max()),unique_values=int(np.unique(x).size),sha256_first=hx,sha256_second=hy,deterministic=hx==hy)
            except Exception as e: r["error"]=f"{type(e).__name__}: {e}"
            rows.append(r)
            if i%10==0 or i==len(chosen): log(f"Progress: {i}/{len(chosen)}")
        sm=pd.DataFrame(rows); sm.to_csv(out/"stage4_smoke_test.csv",index=False)
        smoke_ok=bool(len(sm)>0 and sm.read_ok.all() and sm.deterministic.all())
        log(f"Smoke test saved: {out/'stage4_smoke_test.csv'}"); log(f"Smoke test passed: {smoke_ok}")
    else: log("\n[3/4] Smoke test skipped; run with --smoke-test to execute.")
    spec={
      "stage":4,"title":"Preprocessing specification and pipeline validation",
      "created_at":datetime.now().astimezone().isoformat(timespec="seconds"),
      "input_manifest":str(mp),"stage3_report":str(s3),"source_data_modified":False,"training_performed":False,
      "split_policy":{"subject_disjoint":True,"expected_subject_counts":{"train":64,"val":10,"test":50},"split_recreated":False},
      "input_contract":{"source_format":"PNG","observed_dimensions":"64x64 (from Stage 3 report)","load_as":"8-bit grayscale uint8 array","thresholding":"none","cropping":"none","resizing":"none","normalization":"not applied in this stage; specify and version before training","augmentation":"none","frame_order":"preserve numeric filename/frame order; never shuffle within a sequence","sequence_sampling":"must be fixed before comparative experiments"},
      "input_checks":checks,"split_counts":{str(k):int(v) for k,v in split_counts.items()},
      "split_counts_pass":bool(split_ok),"subject_disjoint_pass":disjoint,"smoke_test_requested":a.smoke_test,"smoke_test_pass":smoke_ok,
      "elapsed_seconds":round(time.time()-start,2),
      "notes":["This stage does not train or extract embeddings.","Grayscale conversion preserves intensity levels as uint8 but may convert a source mode to grayscale.","No thresholding, cropping, resizing, normalization, or augmentation is applied here.","Stage 3 readability checks do not prove semantic silhouette correctness."]
    }
    sp=out/"stage4_preprocessing_spec.json"; sp.write_text(json.dumps(spec,indent=2),encoding="utf-8")
    status="PASS" if all(checks.values()) and split_ok and disjoint and smoke_ok is not False else "REVIEW"
    report={"status":status,"checks":checks,"split_counts":spec["split_counts"],"split_counts_pass":bool(split_ok),"subject_disjoint_pass":disjoint,"smoke_test_pass":smoke_ok,"stage3_images_checked":qc.get("images_checked"),"stage3_failed_records":qc.get("failed_or_missing_image_records"),"spec_path":str(sp),"training_performed":False,"source_data_modified":False}
    rp=out/"stage4_validation_report.json"; rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    log("\n[4/4] Artifacts"); log(f"Preprocessing spec: {sp}"); log(f"Validation report: {rp}"); log(f"Status: {status}")
    log("No model training was performed. Source images were not modified.")
    (out/"stage4_console_log.txt").write_text("\n".join(logs)+"\n",encoding="utf-8")
    if status!="PASS": sys.exit(2)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"\nSTAGE 4 FAILED\n{type(e).__name__}: {e}"); sys.exit(1)
