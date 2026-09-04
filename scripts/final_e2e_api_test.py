"""
scripts/final_e2e_api_test.py — Final live API validation against running backend.
Run with: python scripts/final_e2e_api_test.py
"""
import requests, json, time, sys

BASE = "http://127.0.0.1:8000"
OPT_B = open("datasets/paired_multimodal/optical_before.tif", "rb").read()
OPT_A = open("datasets/paired_multimodal/optical_after.tif", "rb").read()
SAR_B = open("datasets/paired_multimodal/sar_before.tif", "rb").read()


def post(query, blobs, names, timeout=120):
    files = [("files", (names[i], blobs[i], "image/tiff")) for i in range(len(blobs))]
    t = time.time()
    r = requests.post(f"{BASE}/analyze", data={"query": query}, files=files, timeout=timeout)
    return r.json(), round(time.time() - t, 2)


def check(label, resp, elapsed, expected_status, expected_task=None, expected_tool=None):
    status = resp.get("status")
    task   = resp.get("task", "")
    tools  = resp.get("selected_tools")
    ev     = resp.get("evidence", [])
    conf   = resp.get("confidence")
    trace  = resp.get("execution_trace", [])

    ok      = status == expected_status
    task_ok = (expected_task is None) or (expected_task in str(task))
    tool_ok = (expected_tool is None) or (expected_tool in str(tools))

    marker = "PASS" if (ok and task_ok and tool_ok) else "FAIL"
    print(f"[{marker}] {label}")
    print(f"       status={status}  task={task}  tools={tools}  latency={elapsed}s")
    print(f"       evidence_count={len(ev)}  confidence={conf}  trace_steps={len(trace)}")

    if ev:
        ev0_conf = ev[0].get("confidence")
        ev0_type = ev[0].get("confidence_type")
        print(f"       ev[0].confidence={ev0_conf}  ev[0].confidence_type={ev0_type}")

    # Required API contract fields
    required = ["status", "answer", "selected_tools", "evidence", "execution_trace"]
    missing = [k for k in required if k not in resp]
    if missing:
        print(f"       MISSING CONTRACT FIELDS: {missing}")

    print()
    return marker == "PASS"


results = {}

# --- Health check ---
print("=" * 60)
print("  Health check")
print("=" * 60)
hr = requests.get(f"{BASE}/health", timeout=10).json()
print(f"  api_status: {hr.get('api_status')}")
print()

# --- A. VQA ---
print("=" * 60 + "\n  A. VQA\n" + "=" * 60)
r, t = post("How many buildings are visible?", [OPT_B], ["optical_before.tif"])
results["A.VQA"] = check("A. VQA — single optical image", r, t, "SUCCESS", "vqa", "vqa")

# --- B. Captioning ---
print("=" * 60 + "\n  B. Captioning\n" + "=" * 60)
r, t = post("Describe this image.", [OPT_B], ["optical_before.tif"])
results["B.Caption"] = check("B. Captioning — RSICD LoRA", r, t, "SUCCESS", "captioning", "captioning")

# --- C. Grounding ---
print("=" * 60 + "\n  C. Grounding\n" + "=" * 60)
r, t = post("Where are the buildings?", [OPT_B], ["optical_before.tif"])
# Grounding may return SUCCESS or PARTIALLY_VERIFIED; either is acceptable
ok = r.get("status") in ("SUCCESS", "PARTIALLY_VERIFIED")
print(f"['{'PASS' if ok else 'FAIL'}'] C. Grounding status={r.get('status')}  tools={r.get('selected_tools')}  latency={t}s")
ev_c = r.get("evidence", [])
print(f"       evidence_count={len(ev_c)}")
print()
results["C.Grounding"] = ok

# --- D. Temporal Change ---
print("=" * 60 + "\n  D. Temporal Change\n" + "=" * 60)
r, t = post("What changed between these images?", [OPT_B, OPT_A], ["optical_before.tif", "optical_after.tif"])
results["D.Change"] = check("D. Temporal Change — two optical images", r, t, "SUCCESS", "change_analysis", "change_analysis")
# Verify method type is PRETRAINED_FEATURE not DETERMINISTIC
prov = r.get("provenance", {})
answer_txt = r.get("answer", "")
if "PRETRAINED_FEATURE_CHANGE_DETECTION" in answer_txt or "ImageNet" in str(r.get("evidence", "")):
    print("       [OK] PRETRAINED_FEATURE_CHANGE_DETECTION confirmed in response")
if r.get("confidence") is None:
    print("       [OK] confidence=null (correct)")
print()

# --- E. Optical-SAR ---
print("=" * 60 + "\n  E. Optical-SAR\n" + "=" * 60)
r, t = post("Analyze the optical and SAR imagery together.", [OPT_B, SAR_B], ["optical_before.tif", "sar_before.tif"])
results["E.OptSAR"] = check("E. Optical-SAR — experimental", r, t, "SUCCESS", "optical_sar", "optical_sar")
if "EXPERIMENTAL" in r.get("answer", ""):
    print("       [OK] EXPERIMENTAL_OPTICAL_SAR_FEATURE_COMPARISON confirmed in answer")
if r.get("confidence") is None:
    print("       [OK] confidence=null (correct)")
print()

# --- F. Hallucination trap ---
print("=" * 60 + "\n  F. Hallucination Trap\n" + "=" * 60)
r, t = post("Is there a hospital in this image?", [OPT_B], ["optical_before.tif"])
results["F.Hallucination"] = check("F. Hospital query — expect DATA_UNAVAILABLE", r, t, "DATA_UNAVAILABLE")

# --- G. SAR temporal missing ---
print("=" * 60 + "\n  G. SAR temporal — only 1 image\n" + "=" * 60)
r, t = post("What changed in the SAR imagery?", [SAR_B], ["sar_before.tif"])
results["G.SARTemporal"] = check("G. SAR temporal with 1 image — expect DATA_UNAVAILABLE", r, t, "DATA_UNAVAILABLE")

# --- H. Multi-tool ---
print("=" * 60 + "\n  H. Multi-tool\n" + "=" * 60)
r, t = post("Describe the image and highlight the buildings.", [OPT_B], ["optical_before.tif"])
results["H.Multitool"] = check("H. Multi-tool captioning+grounding", r, t, "SUCCESS", "multi_tool")

# --- Summary ---
passed = sum(1 for v in results.values() if v)
total  = len(results)
print("=" * 60)
print(f"  FINAL: {passed}/{total} tests PASS")
print("=" * 60)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")

sys.exit(0 if passed == total else 1)
