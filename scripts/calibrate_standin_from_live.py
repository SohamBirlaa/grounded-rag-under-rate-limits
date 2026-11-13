"""Fit the stand-in's parameters from REAL probe data (data/calibration/live_probe_<model>.json).

The brief requires a stand-in for loads the free tier cannot take, calibrated from real
measurements. This writes config/calibration_live.json; it never invents numbers: every value
is a fit or percentile of live rows, and the fit residual is reported.
"""

import json
import sys

import numpy as np

from _live_common import ROOT

model = sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b-instant"
probe = json.loads((ROOT / "data/calibration" / f"live_probe_{model.replace('/', '_')}.json").read_text())
rows = [r for r in probe["phases"]["latency"]["rows"] if r["ok"] and r["ttft_ms"] and r["prompt_tokens"]]
x = np.array([r["prompt_tokens"] for r in rows], float)
y = np.array([r["ttft_ms"] for r in rows], float)
slope, base = np.polyfit(x, y, 1)
resid = y - (base + slope * x)
itl = [r["itl_ms"] for r in rows if r.get("itl_ms")]
conc = probe["phases"].get("concurrency", {})
c1 = (conc.get("1") or conc.get(1) or {}).get("ttft_p50")
c8 = (conc.get("8") or conc.get(8) or {}).get("ttft_p50")
out = {
    "source": f"data/calibration/live_probe_{model}.json", "n_rows": len(rows),
    "base_ttft_ms": round(float(base), 1), "prompt_token_slope_ms": round(float(slope), 4),
    "noise_sigma_ms": round(float(resid.std()), 1), "ttft_p95_residual_ms": round(float(np.percentile(np.abs(resid), 95)), 1),
    "itl_ms_p50": round(float(np.median(itl)), 2) if itl else None,
    "concurrency_ttft_ratio_8_vs_1": round(c8 / c1, 2) if c1 and c8 else None,
    "token_ratio": probe["phases"].get("ratio", {}).get("token_ratio_mean"),
}
(ROOT / "config/calibration_live.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
