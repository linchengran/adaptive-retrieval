# -*- coding: utf-8 -*-
"""四域基线/回归对比：读 experiments/*/skill_index + testset，输出指标与稳态延迟。

用法：
    python _bench.py before          # 生成 experiments/_opt/before.json
    python _bench.py after           # 生成 experiments/_opt/after.json
    python _bench.py diff            # 打印 before vs after 对比表
"""
import ast
import json
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
sys.path.insert(0, str(SKILL_DIR / "scripts"))
import retrieval_core as rc  # noqa: E402

DOMAINS = ["m5_dureader", "m7_cmrc2018", "m7_t2retrieval", "m7_cmedqa"]
OUT_DIR = ROOT / "experiments" / "_opt"


def normalize(dom):
    p = ROOT / "experiments" / dom / "testset.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    out = []
    for it in items:
        rel = it.get("relevant_chunks") or it.get("relevant") or []
        if isinstance(rel, str):
            rel = ast.literal_eval(rel)
        out.append({"query": it.get("query") or it["question"], "relevant": list(rel)})
    return out


def run(tag, domains=None):
    rc.ensure_utf8_stdout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for dom in (domains or DOMAINS):
        idx = ROOT / "experiments" / dom / "skill_index"
        items = normalize(dom)
        tmp = Path(tempfile.mkdtemp()) / "ts.json"
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        res = rc.eval_index(str(idx), str(tmp), profiles=("fast", "auto", "best"),
                            k_values=(1, 5), with_baseline=True, quiet=True)
        eval_s = time.time() - t0

        # 稳态延迟：预热 2 条后逐条计时，剔除 model_load
        ret = rc.Retriever(str(idx))
        for q in [it["query"] for it in items[:2]]:
            ret.retrieve(q, k=5, profile="auto")
        lat = {}
        for prof in ("fast", "auto", "best"):
            ts = []
            for it in items:
                a = time.perf_counter()
                ret.retrieve(it["query"], k=5, profile=prof)
                ts.append((time.perf_counter() - a) * 1000)
            ts.sort()
            lat[prof] = {"p50": round(ts[len(ts) // 2], 1),
                         "p95": round(ts[int(len(ts) * 0.95)], 1),
                         "mean": round(sum(ts) / len(ts), 1)}

        report[dom] = {"corpus": res["corpus"], "profiles": res["profiles"],
                       "latency": lat, "eval_seconds": round(eval_s, 1)}
        print(f"[{tag}] {dom:18s} " + "  ".join(
            f"{p}: h1={res['profiles'][p]['hit@1']:.3f} h5={res['profiles'][p]['hit@5']:.3f} "
            f"n5={res['profiles'][p]['ndcg@5']:.3f}" for p in ("fast", "auto", "best", "dense")))
        print(f"[{tag}] {dom:18s} latency " + "  ".join(
            f"{p}={lat[p]['p50']}ms" for p in ("fast", "auto", "best")))

    out = OUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out}")


def diff():
    b = json.loads((OUT_DIR / "before.json").read_text(encoding="utf-8"))
    a = json.loads((OUT_DIR / "after.json").read_text(encoding="utf-8"))
    cols = ["hit@1", "hit@5", "ndcg@5", "mrr"]
    print(f"{'域':16s} {'档':5s} " + " ".join(f"{c:>9s}" for c in cols) + "   Δhit@1   Δndcg@5  p50ms")
    for dom in b:
        if dom not in a:
            continue
        for prof in ("fast", "auto", "best", "dense"):
            if prof not in b[dom]["profiles"] or prof not in a[dom]["profiles"]:
                continue
            before, after = b[dom]["profiles"][prof], a[dom]["profiles"][prof]
            d1 = after["hit@1"] - before["hit@1"]
            dn = after["ndcg@5"] - before["ndcg@5"]
            flag = "  <<<" if (d1 + dn) > 0.004 else ("  vvv" if (d1 + dn) < -0.004 else "")
            lat = a[dom]["latency"].get(prof, {}).get("p50", "-")
            print(f"{dom:16s} {prof:5s} " + " ".join(
                f"{before[c]:9.3f}" for c in cols)
                + f"   {d1:+6.3f}   {dn:+6.3f}   {lat}{flag}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "before"
    if cmd == "diff":
        diff()
    else:
        run(cmd, sys.argv[2:] or None)
