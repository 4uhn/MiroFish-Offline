#!/usr/bin/env python3
"""
MiroFish LLM Readiness Tests
=============================
Quick smoke tests to verify tool calling, JSON generation, and token usage
before running a full simulation. Supports both Ollama (local) and Groq (cloud).

Usage:
    # Test current .env provider (default)
    python test_llm_readiness.py

    # Force a specific provider
    python test_llm_readiness.py --provider ollama
    python test_llm_readiness.py --provider groq

    # Override model
    python test_llm_readiness.py --provider ollama --model qwen3:8b
    python test_llm_readiness.py --provider groq --model llama-3.3-70b-versatile
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Add parent dirs so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

# Load .env from project root
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

from openai import OpenAI


# ── Token tracker ──────────────────────────────────────────────

@dataclass
class TokenTracker:
    """Accumulates token usage across all API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    call_log: list = field(default_factory=list)

    def record(self, name: str, usage, elapsed_ms: float):
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0
        self.calls += 1
        self.call_log.append({
            "test": name,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "elapsed_ms": round(elapsed_ms),
        })

    def summary(self) -> str:
        lines = [
            "",
            "=" * 65,
            "TOKEN USAGE SUMMARY",
            "=" * 65,
            f"  API calls:         {self.calls}",
            f"  Prompt tokens:     {self.prompt_tokens:,}",
            f"  Completion tokens: {self.completion_tokens:,}",
            f"  Total tokens:      {self.total_tokens:,}",
            "",
            f"  {'Test':<35} {'In':>7} {'Out':>7} {'Total':>7} {'Time':>8}",
            f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*8}",
        ]
        for c in self.call_log:
            lines.append(
                f"  {c['test']:<35} {c['prompt_tokens']:>7,} {c['completion_tokens']:>7,} "
                f"{c['total_tokens']:>7,} {c['elapsed_ms']:>7,}ms"
            )
        lines.append("")

        # Estimate full simulation token usage
        # A 15-agent, 25-round sim makes ~375 LLM calls for actions + ~50 for pipeline stages
        # Each action call uses ~500 tokens (prompt + completion)
        # Pipeline calls (ontology, profiles, config, report) use ~2000-4000 each
        if self.total_tokens > 0 and self.calls > 0:
            avg_per_call = self.total_tokens / self.calls
            sim_action_calls = 15 * 25  # agents * rounds
            sim_pipeline_calls = 30     # ontology + profiles + config + report sections
            est_action_tokens = int(sim_action_calls * avg_per_call)
            est_pipeline_tokens = int(sim_pipeline_calls * avg_per_call * 3)  # pipeline calls are larger
            est_total = est_action_tokens + est_pipeline_tokens

            lines.append("  FULL SIMULATION ESTIMATE (15 agents, 25 rounds):")
            lines.append(f"    Agent action calls (~{sim_action_calls}):  ~{est_action_tokens:,} tokens")
            lines.append(f"    Pipeline calls (~{sim_pipeline_calls}):     ~{est_pipeline_tokens:,} tokens")
            lines.append(f"    Estimated total:            ~{est_total:,} tokens")
            lines.append("")
            lines.append("  GROQ FREE TIER (14,400 tokens/day):")
            if est_total <= 14_400:
                lines.append(f"    Fits in free tier")
            else:
                sims_per_day = 14_400 / max(est_total, 1)
                lines.append(f"    ~{sims_per_day:.2f} full sims/day ({est_total:,} tokens/sim)")
                lines.append(f"    Consider: reduce agents or rounds, or upgrade Groq plan")

        lines.append("=" * 65)
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────

def make_client(provider: str, model: str, api_key: Optional[str] = None):
    """Create an OpenAI-compatible client for the given provider."""
    if provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            print("ERROR: GROQ_API_KEY not set. Add it to .env or pass --api-key")
            sys.exit(1)
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=60), model
    else:
        key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(api_key=key, base_url=base, timeout=120), model


def get_extra_body(provider: str) -> dict:
    """Return extra_body kwargs for Ollama (think:false, num_ctx)."""
    if provider == "ollama":
        return {"think": False, "options": {"num_ctx": 8192}}
    return {}


def clean_content(text: str) -> str:
    """Strip thinking tags from model output."""
    if not text:
        return ""
    return re.sub(r'<think>[\s\S]*?</think>', '', text).strip()


def timed_call(client, tracker: TokenTracker, test_name: str, **kwargs):
    """Make a chat completion call, track tokens, return response."""
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed = (time.time() - t0) * 1000
    tracker.record(test_name, resp.usage, elapsed)
    return resp, elapsed


# ── Tests ──────────────────────────────────────────────────────

def test_basic_chat(client, model, provider, tracker):
    """Test 1: Basic chat — can the model respond at all?"""
    print("\n[1/5] Basic chat completion...", end=" ", flush=True)
    resp, ms = timed_call(
        client, tracker, "basic_chat",
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: HELLO MIROFISH"}],
        temperature=0, max_tokens=100,
        extra_body=get_extra_body(provider),
    )
    content = clean_content(resp.choices[0].message.content or "")
    ok = "MIROFISH" in content.upper()
    print(f"{'PASS' if ok else 'FAIL'} ({ms:.0f}ms) -- {content[:60]}")
    return ok


def test_tool_calling(client, model, provider, tracker):
    """Test 2: Tool calling — the critical OASIS requirement."""
    print("[2/5] Tool calling (OASIS agent actions)...", end=" ", flush=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_post",
                "description": "Create a social media post",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The post content"},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "like_post",
                "description": "Like an existing post",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "post_id": {"type": "integer", "description": "ID of the post to like"},
                    },
                    "required": ["post_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "do_nothing",
                "description": "Choose to take no action this round",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    resp, ms = timed_call(
        client, tracker, "tool_calling",
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a public health official on social media. "
                    "You must use one of the provided tools to take an action. "
                    "Write a concerned post about a meningitis outbreak in Kent."
                ),
            },
            {"role": "user", "content": "It's your turn. Choose an action."},
        ],
        tools=tools,
        temperature=0.7, max_tokens=1000,
        extra_body=get_extra_body(provider),
    )

    msg = resp.choices[0].message
    has_tool_calls = msg.tool_calls is not None and len(msg.tool_calls) > 0

    if has_tool_calls:
        tc = msg.tool_calls[0]
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = tc.function.arguments
        preview = json.dumps(fn_args)[:80]
        print(f"PASS ({ms:.0f}ms) -- {fn_name}({preview})")
        return True
    else:
        content = clean_content(msg.content or "")[:100]
        print(f"FAIL ({ms:.0f}ms) -- no tool call, got: {content}")
        return False


def test_json_generation(client, model, provider, tracker):
    """Test 3: JSON output — used by ontology/config/profile generators."""
    print("[3/5] JSON generation (ontology/config)...", end=" ", flush=True)

    resp, ms = timed_call(
        client, tracker, "json_generation",
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a knowledge graph expert. Return ONLY valid JSON, no other text.",
            },
            {
                "role": "user",
                "content": (
                    'Generate 3 entity types for a meningitis outbreak simulation. '
                    'Return JSON: {"entity_types": [{"name": "TypeName", "description": "..."}]}'
                ),
            },
        ],
        temperature=0.3, max_tokens=500,
        extra_body=get_extra_body(provider),
    )

    content = clean_content(resp.choices[0].message.content or "")
    content = re.sub(r'^```(?:json)?\s*\n?', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\n?```\s*$', '', content).strip()

    try:
        data = json.loads(content)
        has_types = "entity_types" in data and len(data["entity_types"]) >= 2
        names = [t.get("name", "?") for t in data.get("entity_types", [])[:4]]
        print(f"{'PASS' if has_types else 'FAIL'} ({ms:.0f}ms) -- {len(data.get('entity_types', []))} types: {names}")
        return has_types
    except json.JSONDecodeError:
        print(f"FAIL ({ms:.0f}ms) -- invalid JSON: {content[:80]}")
        return False


def test_persona_generation(client, model, provider, tracker):
    """Test 4: Agent persona — tests creative output + JSON structure."""
    print("[4/5] Persona generation (agent profiles)...", end=" ", flush=True)

    resp, ms = timed_call(
        client, tracker, "persona_generation",
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You create social media agent personas. Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": (
                    'Create a persona for "Dr. Sarah Chen", a public health official in Kent '
                    'responding to a meningitis outbreak. JSON format: '
                    '{"name": "...", "age": <int 25-65>, "bio": "<max 50 words>", '
                    '"mbti": "<4 letters>", "posting_style": "..."}'
                ),
            },
        ],
        temperature=0.7, max_tokens=800,
        extra_body=get_extra_body(provider),
    )

    content = clean_content(resp.choices[0].message.content or "")
    content = re.sub(r'^```(?:json)?\s*\n?', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\n?```\s*$', '', content).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract JSON object via regex (handles truncation/extra text)
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                print(f"FAIL ({ms:.0f}ms) -- invalid JSON: {content[:80]}")
                return False
        else:
            print(f"FAIL ({ms:.0f}ms) -- no JSON found: {content[:80]}")
            return False

    has_fields = all(k in data for k in ["name", "age", "bio"])
    age_ok = isinstance(data.get("age"), int) and 20 <= data["age"] <= 80
    bio_short = len(data.get("bio", "").split()) <= 80
    ok = has_fields and age_ok and bio_short
    print(f"{'PASS' if ok else 'FAIL'} ({ms:.0f}ms) -- age={data.get('age')}, "
          f"mbti={data.get('mbti', '?')}, bio={len(data.get('bio', '').split())}w")
    return ok


def test_multi_turn_tool_calling(client, model, provider, tracker):
    """Test 5: Multi-turn ReACT loop — simulates the report agent."""
    print("[5/5] Multi-turn ReACT loop (report agent)...", end=" ", flush=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "quick_search",
                "description": "Search simulation data for a specific topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are writing a report about a meningitis simulation. "
                "You MUST call the quick_search tool first to gather data, then write findings. "
                "After receiving tool results, write: Final Answer: <your analysis>"
            ),
        },
        {"role": "user", "content": "Analyze public sentiment about the meningitis outbreak."},
    ]

    extra = get_extra_body(provider)

    # Turn 1: expect tool call
    resp1, ms1 = timed_call(
        client, tracker, "react_turn1_tool",
        model=model, messages=messages,
        tools=tools, temperature=0.5, max_tokens=500,
        extra_body=extra,
    )

    msg1 = resp1.choices[0].message
    if not (msg1.tool_calls and len(msg1.tool_calls) > 0):
        content = clean_content(msg1.content or "")[:80]
        print(f"FAIL ({ms1:.0f}ms) -- turn 1 no tool call, got: {content}")
        return False

    tc = msg1.tool_calls[0]

    # Simulate tool result
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps({
            "results": [
                {"agent": "Dr_Smith", "post": "Concerned about the outbreak spread in Canterbury schools"},
                {"agent": "Parent_Jane", "post": "Why aren't they closing the schools?! My kids are at risk"},
                {"agent": "NHS_Kent", "post": "Vaccination clinics now open at 5 locations across Kent"},
            ]
        }),
    })

    # Turn 2: expect final answer using the tool results
    resp2, ms2 = timed_call(
        client, tracker, "react_turn2_answer",
        model=model, messages=messages,
        tools=tools, temperature=0.5, max_tokens=800,
        extra_body=extra,
    )

    msg2 = resp2.choices[0].message
    content2 = clean_content(msg2.content or "")
    has_substance = len(content2) > 50

    total_ms = ms1 + ms2
    print(f"{'PASS' if has_substance else 'FAIL'} ({total_ms:.0f}ms) -- "
          f"tool={tc.function.name}, answer={len(content2)} chars")
    return has_substance


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiroFish LLM readiness tests")
    parser.add_argument("--provider", choices=["ollama", "groq"], default=None,
                        help="LLM provider (default: from .env)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--api-key", default=None, help="API key (for groq)")
    args = parser.parse_args()

    provider = args.provider or os.environ.get("LLM_PROVIDER", "ollama")
    if args.model:
        model = args.model
    elif provider == "groq":
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        model = os.environ.get("LLM_MODEL_NAME", "qwen3:8b")

    print("=" * 65)
    print("MiroFish LLM Readiness Tests")
    print("=" * 65)
    print(f"  Provider:  {provider}")
    print(f"  Model:     {model}")
    if provider == "ollama":
        print(f"  Think:     disabled (extra_body think=false)")
    if provider == "groq":
        has_key = bool(args.api_key or os.environ.get("GROQ_API_KEY"))
        print(f"  API Key:   {'set' if has_key else 'MISSING'}")
    print("=" * 65)

    client, model = make_client(provider, model, args.api_key)
    tracker = TokenTracker()

    results = {}
    tests = [
        ("basic_chat", test_basic_chat),
        ("tool_calling", test_tool_calling),
        ("json_generation", test_json_generation),
        ("persona_generation", test_persona_generation),
        ("multi_turn_react", test_multi_turn_tool_calling),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn(client, model, provider, tracker)
        except Exception as e:
            print(f"ERROR -- {type(e).__name__}: {e}")
            results[name] = False

    # Summary
    print(tracker.summary())

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    critical = results.get("tool_calling", False)
    if not critical:
        print("\n!! CRITICAL: Tool calling FAILED -- OASIS agent actions will not work.")
        print("   The simulation will produce zero organic activity (same as gemma3 bug).")
        if provider == "ollama":
            print(f"   Try: python test_llm_readiness.py --provider groq")
        else:
            print(f"   Try a different model: --model llama-3.3-70b-versatile")
    elif passed == total:
        print("\nAll tests passed -- ready for simulation!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nTool calling works, but failed: {', '.join(failed)}")
        print("Simulation should function but quality may vary.")

    sys.exit(0 if critical else 1)


if __name__ == "__main__":
    main()
