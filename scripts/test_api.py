from __future__ import annotations

import json
import sys
import requests

BASE = "http://127.0.0.1:8000"

def post(path: str, payload: dict):
    url = f"{BASE}{path}"
    r = requests.post(url, json=payload, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body

def main() -> int:
    ok = True

    # 1) Tutor evaluate: Spanish answer expected
    status, body = post("/tutor/evaluate", {
        "lesson_id": 1,
        "item_id": 1,
        "learner_text": "Me llamo Joel y estudio informática.",
        "expected_lang": "ES",
        "allow_mixed": False
    })
    print("\n[1] /tutor/evaluate (ES expected)")
    print("Status:", status)
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    if status != 200:
        ok = False
    else:
        # minimal assertions (adjust keys to your actual response schema)
        for k in ("expected_lang", "detected_lang"):
            if k not in body:
                print(f"Missing key: {k}")
                ok = False

    # 2) Tutor evaluate: wrong-language check (EN expected, Spanish given)
    status, body = post("/tutor/evaluate", {
        "lesson_id": 1,
        "item_id": 1,
        "learner_text": "Me llamo Joel y estudio informática.",
        "expected_lang": "EN",
        "allow_mixed": False
    })
    print("\n[2] /tutor/evaluate (EN expected, ES text)")
    print("Status:", status)
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    if status != 200:
        ok = False

    # 3) Language detection endpoint (if you have it)
    status, body = post("/classify/language", {"text": "Me llamo Joel y estudio informática."})
    print("\n[3] /classify/language")
    print("Status:", status)
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    # allow 404 if you didn’t add the route
    if status not in (200, 404):
        ok = False

    # 4) NER extraction endpoint (if you have it)
    status, body = post("/extract/ner", {"lang": "ES", "text": "Me llamo Joel y estudio informática."})
    print("\n[4] /extract/ner")
    print("Status:", status)
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    if status not in (200, 404):
        ok = False

    # 5) Semantic scoring endpoint (if you have it)
    status, body = post("/semantic/score", {
        "lesson_id": 1,
        "item_id": 1,
        "learner_es": "Me llamo Joel y estudio informática."
    })
    print("\n[5] /semantic/score")
    print("Status:", status)
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    if status not in (200, 404):
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
