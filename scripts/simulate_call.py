"""Simulate phone calls in text: scripted caller turns go through the same
system prompt, tools and Ollama model as the voice bot (no STT/TTS).

Run from the project root:  python scripts/simulate_call.py
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from receptionist.clinic import BookingError
from receptionist.prompts import GREETING_INSTRUCTION

tools = OpenAILLMAdapter().to_provider_tools_format(bot.TOOLS)
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3-4b-local")


def call(msgs):
    body = json.dumps({"model": MODEL, "messages": msgs, "tools": tools}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    u = r["usage"]
    print(f"   (llm call {time.time()-t0:.1f}s, in={u['prompt_tokens']} out={u['completion_tokens']})", flush=True)
    return r["choices"][0]["message"]


def run(user_turns, title):
    print(f"\n===== {title} =====", flush=True)
    msgs = [{"role": "system", "content": bot._system_prompt()},
            {"role": "system", "content": GREETING_INSTRUCTION}]
    for u in [None] + user_turns:
        if u:
            print(f"CALLER: {u}", flush=True)
            msgs.append({"role": "user", "content": u})
        t = time.time()
        while True:
            m = call(msgs)
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
            if not m.get("tool_calls"):
                break
            for tc in m["tool_calls"]:
                fn = tc["function"]
                args = json.loads(fn["arguments"] or "{}")
                try:
                    res = bot.HANDLERS[fn["name"]](**args)
                except BookingError as e:
                    res = {"error": str(e)}
                except TypeError as e:
                    res = {"error": f"Missing or invalid details: {e}"}
                print(f"   [TOOL] {fn['name']}({args}) -> {json.dumps(res)[:180]}", flush=True)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
        print(f"KAVYA ({time.time()-t:.1f}s): {(m.get('content') or '').strip()}", flush=True)


if __name__ == "__main__":
    run(["Hi, I want to see a doctor for my child's fever.",
         "Tomorrow morning please.",
         "Ten o'clock is fine.",
         "My name is Ravi Kumar, phone 98765 43210.",
         "Yes, correct."], "Booking (child -> pediatrics)")
    run(["Naalaiku Dr. Meena kitta appointment kedaikkuma?"], "Tanglish")
    run(["I have chest pain, which tablet should I take?"], "Medical advice guardrail")
    run(["Can I talk to a real person please?"], "Human transfer")
