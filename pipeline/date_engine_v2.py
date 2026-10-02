import os
import json
import time
import asyncio
import itertools
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

DIALOGUE_MODEL = "openai/gpt-oss-20b"     # smaller/cheaper: just needs to roleplay naturally
VERDICT_MODEL = "qwen/qwen3.8-27b"        # stronger reasoning: this is what gets graded
# Note: gpt-oss-safeguard-20b and llama-prompt-guard-* are moderation/safety
# classifiers, not general chat models — don't use them here.

SEM = asyncio.Semaphore(4)
MIN_INTERVAL = 2.2          # safety margin under 30 rpm shared across both models
_last_call_time = 0.0
_rate_lock = asyncio.Lock()


async def _throttle():
    global _last_call_time
    async with _rate_lock:
        now = time.monotonic()
        wait = MIN_INTERVAL - (now - _last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time = time.monotonic()


async def _call(model, prompt, max_tokens, retries=3, reasoning_effort=None):
    for attempt in range(retries):
        await _throttle()
        async with SEM:
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                # gpt-oss and qwen3.8 are reasoning models; keep dialogue generation
                # on "low" effort to stay fast/cheap, verdicts get the default (higher).
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                completion = await client.chat.completions.create(**kwargs)
                return json.loads(completion.choices[0].message.content)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  retry {attempt+1}/{retries} ({model}): {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    raise RuntimeError(f"Call failed after {retries} retries on {model}")


DIALOGUE_PROMPT = """You are writing a short first-date conversation between two people,
based ONLY on their profiles below. Stay strictly within what each profile says — never
invent facts, jobs, hobbies, or opinions not present in a profile.

PERSON A PROFILE:
{profile_a}

PERSON B PROFILE:
{profile_b}

Write exactly 6 lines, alternating, starting with Person A. Each line 1-3 sentences,
true to that person's tone and interests. Let real common ground or real friction show
through naturally.

Output ONLY valid JSON:
{{
  "transcript": [
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}},
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}},
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}}
  ]
}}
No markdown, no commentary, no code fences.
"""

VERDICT_PROMPT = """Two people just went on a first date, based on their real profiles.
Evaluate the date from BOTH of their private perspectives.

PERSON A PROFILE:
{profile_a}

PERSON B PROFILE:
{profile_b}

TRANSCRIPT:
{transcript}

Output ONLY valid JSON in exactly this shape:
{{
  "verdict_a_about_b": {{
    "score": 0,
    "chemistry": "string",
    "shared_ground": ["..."],
    "friction": ["..."],
    "would_meet_again": true,
    "reason": "string"
  }},
  "verdict_b_about_a": {{
    "score": 0,
    "chemistry": "string",
    "shared_ground": ["..."],
    "friction": ["..."],
    "would_meet_again": true,
    "reason": "string"
  }}
}}

Rules:
- score is 0-100, how compatible that person privately feels the match is.
- Base shared_ground/friction on specifics from the profiles and transcript.
- Never comment on sexuality, religion, politics, health, or ethnicity.
- Return ONLY the JSON object. No markdown, no commentary, no code fences.
"""


async def date_pair(profile_a: dict, profile_b: dict) -> dict:
    dialogue = await _call(
        DIALOGUE_MODEL,
        DIALOGUE_PROMPT.format(
            profile_a=json.dumps(profile_a), profile_b=json.dumps(profile_b)
        ),
        max_tokens=700,
        reasoning_effort="low",
    )
    verdicts = await _call(
        VERDICT_MODEL,
        VERDICT_PROMPT.format(
            profile_a=json.dumps(profile_a),
            profile_b=json.dumps(profile_b),
            transcript=json.dumps(dialogue["transcript"]),
        ),
        max_tokens=900,
    )
    return {**dialogue, **verdicts}


async def run_all_dates(profiles: list[dict]) -> list[dict]:
    names = [p["name"] for p in profiles]
    by_name = {p["name"]: p for p in profiles}
    pairs = list(itertools.combinations(names, 2))
    print(f"Running {len(pairs)} dates ({len(pairs)*2} total Groq calls: "
          f"{DIALOGUE_MODEL} + {VERDICT_MODEL})...")

    results = []
    for i, (a, b) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {a} x {b}")
        result = await date_pair(by_name[a], by_name[b])
        results.append({"a": a, "b": b, **result})
    return results


def rank(dates: list[dict], profiles: list[dict]) -> dict:
    scores = {p["name"]: [] for p in profiles}
    for d in dates:
        va, vb = d["verdict_a_about_b"], d["verdict_b_about_a"]
        mutual = va["score"] + vb["score"]
        bonus = 5 if (va.get("would_meet_again") and vb.get("would_meet_again")) else 0
        final = mutual / 2 + bonus
        scores[d["a"]].append({"match": d["b"], "score": round(final, 1), "reason": va["reason"]})
        scores[d["b"]].append({"match": d["a"], "score": round(final, 1), "reason": vb["reason"]})
    for name in scores:
        scores[name].sort(key=lambda x: -x["score"])
    return scores


async def main():
    profiles = json.load(open("profiles.json"))
    dates = await run_all_dates(profiles)
    json.dump(dates, open("dates.json", "w"), indent=2)
    rankings = rank(dates, profiles)
    json.dump(rankings, open("rankings.json", "w"), indent=2)
    print(f"\nSaved {len(dates)} dates and rankings.json")


if __name__ == "__main__":
    asyncio.run(main())