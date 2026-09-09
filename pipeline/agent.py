#!/usr/bin/env python3
"""
The answer layer: a local model, via Ollama, that may only speak in tool calls.

Two constraints shape this file, and both come from running on CPU:

1. GENERATION IS THE BOTTLENECK, not retrieval. At ~10 tokens/sec, a 400-token
   answer takes 40 seconds. So the model is instructed to be terse and the
   tools return small, pre-aggregated results - never 20 full abstracts for it
   to summarise. Polars does the work; the model writes two sentences over it.

2. SMALL MODELS DRIFT. A 7-8B model will happily invent a plausible number if
   the prompt lets it. So numbers are never taken from the model: every figure
   in the final answer must appear in a tool result, and check_answer() below
   verifies that before the answer is shown.

Setup:
    ollama pull qwen3:8b            # trained for tool calling; ~5GB
    pip install ollama
    python agent.py "which journals publish on cisplatin?"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import tools

# Whatever `ollama list` already has locally. qwen2.5 is trained for tool
# calling, which matters far more here than raw size: the model only emits a
# small JSON tool call and two sentences of prose - Polars does the thinking.
# 3b keeps the loop responsive on a busy 4-core CPU; 7b picks tools better.
MODEL = os.environ.get("PUBMED_MODEL", "qwen2.5:7b")

# The context window is the second-biggest memory cost after the weights, and
# it is the one you control. A 7B at 8192 ctx would not load on a 16 GB laptop
# already running the embedding pass; at 4096 it fits. Tool calls and two
# sentences of prose never need 8k anyway - the tools return pre-aggregated
# results precisely so the model has little to read.
NUM_CTX = int(os.environ.get("PUBMED_NUM_CTX", "4096"))

# If the preferred model cannot be loaded (usually out of memory), drop to a
# smaller one rather than failing the request outright. A slightly weaker
# answer beats a stack trace.
FALLBACK = os.environ.get("PUBMED_FALLBACK_MODEL", "qwen2.5:3b")

SYSTEM = """You answer questions about a biomedical literature corpus.

You have no knowledge of this corpus. Every fact you state must come from a
tool result in this conversation. You may not use your own memory of the
medical literature, and you may not estimate.

RULES
1. When a question names a drug, disease or topic, pass THE USER'S OWN WORDS
   as concept_id - for example concept_id="cisplatin". They are resolved for
   you. Do not write a code such as D010051 unless resolve_concept returned
   that exact code earlier in this conversation. Codes you produce yourself are
   wrong, and a wrong code answers confidently about a different disease.
2. Counting, listing or ranking -> list_journals / list_drugs / rank_kols.
   Never answer these from search_literature; it returns a sample, not a total.
3. Open questions about findings or evidence -> search_literature.
4. Never write a number that is not present in a tool result.
5. Cite PMIDs when you refer to specific papers.
6. Be brief. Two or three sentences plus the data. No preamble, no restating
   the question, no offers of further help.
7. If a tool returns a "caveat" or "note" field, pass it on to the user. The
   note says how a vague term was interpreted; the caveat says why a result may
   be wrong. Both change how the answer should be read, so neither is optional.
8. Counting tools return a total (total_journals, total_substances,
   total_authors) alongside a shortened list. ALWAYS state the total, then show
   the top few. "226 journals, led by X (34), Y (33)" is a correct answer;
   listing 25 rows without the total silently implies the list is complete.
9. If a tool returns an "error" field, report it. Do not answer around it.
10. Refer to researchers by name only. The corpus records names, affiliations
    and papers - it does not record anyone's gender. Never write "his" or
    "her" about an author; use their name or "they". Guessing gender from a
    name invents a fact about a real person.
"""


def check_answer(answer: str, observations: list[str]) -> list[str]:
    """
    Cheap, effective hallucination guard: pull every number of 2+ digits out of
    the answer and confirm it appeared in some tool result. Catches the common
    small-model failure of inventing counts, years and PMIDs.
    """
    seen = " ".join(observations)
    bad = []
    for n in set(re.findall(r"\b\d{2,}\b", answer)):
        if n not in seen:
            bad.append(n)
    return bad


# --------------------------------------------------------------- providers
#
# Two ways to reach a model, same tool-calling contract:
#
#   ollama  - local, private, free, but generation on CPU is minutes per answer
#   cloud   - any OpenAI-compatible endpoint (Groq, Cerebras, OpenRouter...).
#             Free tiers are fast enough to be usable and need no local RAM.
#
# Set PUBMED_LLM=cloud plus PUBMED_API_KEY to switch. Nothing else changes:
# the tool schemas are already in OpenAI function-calling format, which is what
# these endpoints speak.
PROVIDER = os.environ.get("PUBMED_LLM", "ollama").lower()
API_BASE = os.environ.get("PUBMED_API_BASE", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("PUBMED_API_KEY", "")
CLOUD_MODEL = os.environ.get("PUBMED_CLOUD_MODEL", "llama-3.3-70b-versatile")


def _chat_cloud(messages, model, _tries: int = 4):
    """
    One turn against an OpenAI-compatible chat endpoint, with rate-limit waits.

    Free tiers are measured in tokens per MINUTE, and this agent is unusually
    token-hungry for its size: 16 tool schemas cost ~2,800 tokens on every
    request, before any conversation. On Groq's 8,000 TPM free tier that is
    barely two calls, so a multi-step answer hits 429 halfway through and the
    whole question fails after the work was already done.

    The API says exactly how long to wait ("try again in 2.7s"), so honour it
    rather than guessing. Waiting three seconds beats losing the answer.
    """
    import re as _re
    import time as _time

    import requests
    payload = {"model": model, "messages": messages,
               "tools": tools.TOOL_SCHEMAS, "temperature": 0}
    for attempt in range(_tries):
        r = requests.post(f"{API_BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {API_KEY}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=120)
        if r.status_code != 429:
            break
        wait = float(r.headers.get("retry-after") or 0)
        if not wait:
            m = _re.search(r"try again in ([\d.]+)s", r.text)
            wait = float(m.group(1)) if m else 2.0 * (attempt + 1)
        wait = min(wait + 0.5, 30.0)
        if attempt == _tries - 1:
            break
        print(f"  [rate limit] waiting {wait:.1f}s", file=sys.stderr)
        _time.sleep(wait)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    msg = r.json()["choices"][0]["message"]
    # Normalise to the shape the loop below already expects from ollama.
    calls = []
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function", {})
        calls.append({"id": c.get("id"), "type": "function",
                      "function": {"name": fn.get("name"),
                                   "arguments": fn.get("arguments") or "{}"}})
    return {"role": "assistant", "content": msg.get("content") or "",
            "tool_calls": calls}


def _chat_ollama(messages, model):
    import ollama
    resp = ollama.chat(model=model, messages=messages, tools=tools.TOOL_SCHEMAS,
                       options={"temperature": 0, "num_ctx": NUM_CTX})
    return resp["message"]


def run(question: str, model: str = MODEL, max_steps: int = 6, verbose: bool = True):
    cloud = PROVIDER == "cloud"
    if cloud:
        if not API_KEY:
            raise RuntimeError("PUBMED_LLM=cloud but PUBMED_API_KEY is not set")
        model = os.environ.get("PUBMED_CLOUD_MODEL", CLOUD_MODEL)

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    observations: list[str] = []
    disclosures: list[str] = []
    got_data = False
    calls_made: list[dict] = []

    for step in range(max_steps):
        try:
            msg = _chat_cloud(messages, model) if cloud else _chat_ollama(messages, model)
        except Exception as e:
            if (not cloud) and FALLBACK and model != FALLBACK and "memory" in str(e).lower():
                if verbose:
                    print(f"  [model] {model} would not load (out of memory); "
                          f"falling back to {FALLBACK}", file=sys.stderr)
                model = FALLBACK
                continue
            raise
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            bad = check_answer(answer, observations)
            if bad and step < max_steps - 1:
                if verbose:
                    print(f"  [guard] unsupported numbers {bad} - asking again", file=sys.stderr)
                messages.append({"role": "user", "content":
                                 f"These numbers do not appear in any tool result: "
                                 f"{', '.join(bad)}. Rewrite using only values from the "
                                 f"tool results, or call another tool."})
                continue
            # The numeric guard below catches invented COUNTS. It cannot catch
            # invented NAMES: told its concept id was invalid, a 7B model
            # replied "1. Dr. Jane Smith 2. Dr. John Doe 3. Dr. Emily Johnson".
            # Every one was fabricated, and not a single digit was out of place,
            # so the number check passed it. The structural fix is to refuse
            # outright when no tool ever returned rows - a model with no data
            # has nothing to summarise, whatever it wrote.
            if not got_data:
                reason = " ".join(disclosures) if disclosures else (
                    "No tool returned any matching records.")
                return {"answer": ("I could not answer that from this corpus. "
                                   + reason),
                        "steps": step, "disclosures": disclosures, "model": model,
                        "provider": "cloud" if cloud else "ollama",
                        "refused": True, "unverified_numbers": bad, "calls": calls_made,
                        "model": model, "provider": "cloud" if cloud else "ollama",
                        "trace": observations}

            # Rule 7 asks the model to pass caveats on. Small models forget.
            # Append anything it dropped: a limitation the reader never sees is
            # the same as no limitation at all.
            missed = [d for d in disclosures if d[:40] not in answer]
            if missed:
                answer = answer.rstrip() + "\n\n" + "\n".join("Note: " + m for m in missed)
            return {"answer": answer, "steps": step, "disclosures": disclosures,
                    "calls": calls_made, "refused": False, "model": model,
                    "provider": "cloud" if cloud else "ollama",
                    "unverified_numbers": bad, "trace": observations}

        for c in calls:
            fn = c["function"]["name"]
            args = c["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args or "{}")
            calls_made.append({"tool": fn, "args": args})
            if verbose:
                print(f"  [{step}] {fn}({json.dumps(args)[:110]})", file=sys.stderr)
            call_id = c.get("id")
            result = tools.call(fn, args)
            # Caveats and interpretation notes are the parts of a result most
            # likely to be dropped, because they are the least interesting to a
            # model trying to be brief - and the most important to a reader.
            # Collect them so they can be enforced rather than merely requested.
            if isinstance(result, list) and result:
                got_data = True
            if isinstance(result, dict):
                if result.get("results"):
                    got_data = True
                for key in ("error", "caveat", "note"):
                    v = result.get(key)
                    if v and v not in disclosures:
                        disclosures.append(v)
            payload = json.dumps(result, default=str)[:6000]
            observations.append(payload)
            tool_msg = {"role": "tool", "name": fn, "content": payload}
            if call_id:
                tool_msg["tool_call_id"] = call_id     # required by OpenAI-style APIs
            messages.append(tool_msg)

    return {"answer": "Could not resolve within the step limit.",
            "steps": max_steps, "model": model,
            "provider": "cloud" if cloud else "ollama", "trace": observations}


# ---------------------------------------------------------------- dry run

def dry_run(question: str) -> None:
    """
    Show what the agent WOULD do, without a model. Useful for wiring up the
    tool layer before Ollama is installed, and for the eval harness.
    """
    print(f"Q: {question}\n")
    concepts = tools.resolve_concept(re.sub(
        r"(?i)^(which|what|who|list|top|how many)\b\s*", "", question).strip("? "))
    print("resolve_concept ->", json.dumps(concepts[:2], indent=2))
    if concepts:
        cid = concepts[0]["concept_id"]
        for name, kw in (("list_journals", {"concept_id": cid, "limit": 5}),
                         ("list_drugs", {"concept_id": cid, "limit": 5}),
                         ("rank_kols", {"concept_id": cid, "limit": 5})):
            print(f"\n{name} ->", json.dumps(tools.call(name, kw), default=str)[:600])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry", action="store_true", help="no LLM; show the tool path")
    a = ap.parse_args()

    if a.dry:
        dry_run(a.question)
    else:
        out = run(a.question, a.model)
        print("\n" + out["answer"])
        if out.get("unverified_numbers"):
            print(f"\n[warning] unverified numbers: {out['unverified_numbers']}")
