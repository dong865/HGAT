"""Audit every local pt/pth file for scheme-3 compatibility."""
import argparse,json
from pathlib import Path
import torch

def classify(path):
    try: size=path.stat().st_size
    except OSError as e:
        return {"path":str(path.absolute()),"size_bytes":None,
                "category":"incompatible","reason":f"missing_or_broken_link: {e}"}
    row={"path":str(path.resolve()),"size_bytes":size}
    try: x=torch.load(path,map_location="cpu",weights_only=False)
    except Exception as e:
        return {**row,"category":"incompatible","reason":f"load_error: {type(e).__name__}: {e}"}
    if not isinstance(x,dict): return {**row,"category":"incompatible","reason":f"payload_type={type(x).__name__}"}
    keys=sorted(map(str,x.keys())); arch=x.get("architecture_version"); schema=x.get("schema")
    row.update(architecture=arch,schema=schema,keys=keys,hidden_dim=x.get("hidden_dim"))
    if arch=="normalized_demand_student_v3" and "demand_head" in x and x.get("num_pursuers")==4:
        row.update(category="directly_usable",reason="scheme-3 4v2 distilled demand checkpoint")
    elif {"st_hgat","relation_head","actor"}.issubset(x):
        row.update(category="warm_start_only",reason="policy backbone/actor available; lacks scheme-3 demand")
    elif schema in {"coalition_value_geometric_proxy_v1","coalition_value_scale_aware_teacher_v2"} or "teacher" in x:
        row.update(category="offline_only",reason="value teacher/proxy is excluded from online scheme-3 runtime")
    elif "st_hgat" in x or "encoder_state_dict" in x or "model_state_dict" in x:
        row.update(category="warm_start_only",reason="partial encoder/model weights require migration")
    else: row.update(category="incompatible",reason="missing scheme-3 demand and required policy modules")
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument("roots",nargs="+");p.add_argument("--output",required=True);a=p.parse_args()
    paths=[]
    for root in a.roots:
        q=Path(root);paths += list(q.rglob("*.pt"))+list(q.rglob("*.pth"))
    rows=[classify(x) for x in sorted(set(paths))]
    counts={k:sum(r["category"]==k for r in rows) for k in ("directly_usable","warm_start_only","offline_only","incompatible")}
    out={"total":len(rows),"counts":counts,"models":rows}
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps({"total":len(rows),"counts":counts},indent=2))
if __name__=="__main__":main()
