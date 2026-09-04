"""
scripts/verify_change_optical_sar.py

Real inference verification against:
  datasets/paired_multimodal/optical_before.tif
  datasets/paired_multimodal/optical_after.tif
  datasets/paired_multimodal/sar_before.tif

Tests:
  A. Temporal change detection (upgraded)
  B. Optical-SAR analysis (upgraded)
  C. Missing sar_after behaviour (should NOT error)

Run from the project root:
  python scripts/verify_change_optical_sar.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.change_map import compute_change_baseline
from backend.app.optical_sar import verify_optical_sar_pair

DATA = "datasets/paired_multimodal"
OPT_B = os.path.join(DATA, "optical_before.tif")
OPT_A = os.path.join(DATA, "optical_after.tif")
SAR_B = os.path.join(DATA, "sar_before.tif")

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# ── A. Temporal change detection ───────────────────────────────
section("A. Temporal Change Detection")

b_opt = read_bytes(OPT_B)
a_opt = read_bytes(OPT_A)

t0 = time.time()
result = compute_change_baseline(b_opt, a_opt)
elapsed = round(time.time() - t0, 2)

print(f"Status:      {result.get('status')}")
print(f"Answer:      {result.get('answer', '')[:120]}")
prov = result.get("provenance", {})
print(f"Method type: {prov.get('method_type', 'N/A')}")
print(f"Model:       {prov.get('model', 'N/A')}")
print(f"Latency:     {elapsed}s")
print(f"Evidence[0]: {result.get('evidence', [{}])[0].get('evidence', '')[:120]}")
print(f"Visual PNG:  {'YES' if result.get('visual_output') else 'NO'}")
assert result["status"] == "SUCCESS", f"FAIL: status={result['status']}"
print("Result A: PASS")


# ── B. Optical-SAR analysis ────────────────────────────────────
section("B. Optical-SAR Analysis")

b_sar = read_bytes(SAR_B)

t0 = time.time()
result2 = verify_optical_sar_pair(b_opt, b_sar)
elapsed2 = round(time.time() - t0, 2)

print(f"Status:      {result2.get('status')}")
print(f"Answer:      {result2.get('answer', '')[:160]}")
prov2 = result2.get("provenance", {})
print(f"Method type: {prov2.get('method_type', 'N/A')}")
print(f"Model:       {prov2.get('model', 'N/A')}")
print(f"Latency:     {elapsed2}s")
evs = result2.get("evidence", [])
for i, ev in enumerate(evs):
    print(f"Evidence[{i}]: {ev.get('claim','')} | {ev.get('evidence','')[:80]}")
assert result2["status"] == "SUCCESS", f"FAIL: status={result2['status']}"
print("Result B: PASS")


# ── C. Missing sar_after — graceful ────────────────────────────
section("C. Missing sar_after (must not crash)")

# The specialist only uses sar_before; sar_after doesn't exist and shouldn't be needed.
# If a future caller passes sar_before twice (before/before), it should still succeed.
result3 = verify_optical_sar_pair(b_opt, b_sar)
assert result3["status"] == "SUCCESS", f"FAIL: status={result3['status']}"
assert "Single-epoch" in result3.get("answer", ""), "FAIL: must say Single-epoch"
print(f"Status:  {result3['status']}")
print(f"Answer:  {result3.get('answer','')[:100]}")
print("Result C: PASS")


# ── Summary ────────────────────────────────────────────────────
section("SUMMARY")
print(f"{'Test':<40} {'Status':<10} {'Latency':<10}")
print(f"{'Temporal change detection':<40} {'PASS':<10} {elapsed}s")
print(f"{'Optical-SAR analysis':<40} {'PASS':<10} {elapsed2}s")
print(f"{'Missing sar_after guard':<40} {'PASS':<10}")
print("\nAll tests PASSED.")
