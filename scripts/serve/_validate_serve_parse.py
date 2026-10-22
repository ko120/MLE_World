"""Parse one /v1/chat/completions reply and verdict it. Reads JSON on stdin.

A serve is PASS only if it produced non-empty content AND finished on "stop".
finish_reason "length" with empty content means the reasoning trace ate the
whole budget -- raise max_tokens, do not condemn the serve.
"""
import sys, os, json

raw = sys.stdin.read()
sn, endpoint = os.environ.get("SN", "?"), os.environ.get("ENDPOINT", "?")
try:
    d = json.loads(raw)
    ch = d["choices"][0]
    msg = ch["message"]
    content = (msg.get("content") or "").strip()
    think = msg.get("reasoning") or msg.get("reasoning_content") or ""
    fin = ch.get("finish_reason")
    usage = d.get("usage") or {}
    ok = bool(content) and fin == "stop"
    verdict = "PASS" if ok else "SUSPECT"
    print("  %-7s %s @ %s" % (verdict, sn, endpoint))
    print("    content=%r  finish=%s  completion_tokens=%s  reasoning_chars=%d"
          % (content[:60], fin, usage.get("completion_tokens"), len(think)))
    sys.exit(0 if ok else 2)
except Exception:
    print("  FAIL    %s @ %s -- no parseable completion" % (sn, endpoint))
    print("    raw: %s" % raw[:300])
    sys.exit(1)
