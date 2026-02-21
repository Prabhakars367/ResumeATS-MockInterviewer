import json, os, sys

p = "emotion_analysis_log.json"
print("PWD:", os.getcwd())
print("Exists:", os.path.exists(p))

if not os.path.exists(p):
    sys.exit(0)

try:
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    print("JSON load: OK")
    print("Type:", type(j))
    if isinstance(j, dict):
        print("Top keys:", list(j.keys())[:50])
    if isinstance(j, list):
        print("List length:", len(j))
    preview = json.dumps(j if isinstance(j, (dict, list)) else str(j))[:1000]
    print("\nPreview (first 1000 chars):\n", preview)
except Exception as e:
    print("JSON load error:", repr(e))
    raw = open(p,'r',encoding='utf-8',errors='replace').read()[:800]
    print("\nRaw preview (first 800 chars):\n", raw)
