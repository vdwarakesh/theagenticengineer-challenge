import os
import json
import time
import asyncio
import itertools
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "qwen/qwen3.8-27b"  # confirmed free tier: 30 rpm / 1000 rpd

# Respect the 30 req/min free-tier cap. 1 call/pair, so throttle concurrency.
SEM = asyncio.Semaphore(4)          # max concurrent in-flight calls
MIN_INTERVAL = 2.2                   # seconds between call starts (~27/min, safety margin)
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


DATE_PROMPT = """You are simulating a short first date between two people, based ONLY on
their profiles below. Each profile was built from that person's real LinkedIn and
Instagram. Stay strictly within what each profile says — never invent facts, jobs,
hobbies, or opinions not present in a profile.

PERSON A PROFILE:
{profile_a}

PERSON B PROFILE:
{profile_b}

Write a natural, in-character first-date conversation between them: exactly 6 lines
total, alternating, starting with Person A. Each line should be 1-3 sentences, true to
that person's tone, interests, and needs as described in their profile. Let them find
real common ground or real friction grounded in the profiles — don't force positivity.

After the dialogue, output each person's own private, honest evaluation of the date,
as if each of them is reporting back privately. Each verdict is written from that
person's own perspective about the OTHER person.

Output ONLY valid JSON in exactly this shape, no other text:
{{
  "transcript": [
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}},
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}},
    {{"speaker": "A", "text": "..."}},
    {{"speaker": "B", "text": "..."}}
  ],
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
- Base every shared_ground/friction item on specifics from the two profiles.
- Do not let either person comment on the other's sexuality, religion, politics, health,
  or ethnicity, even if speculating.
- Return ONLY the JSON object. No markdown, no commentary, no code fences.
"""


async def date_pair(profile_a: dict, profile_b: dict, retries: int = 3) -> dict:
    prompt = DATE_PROMPT.format(
        profile_a=json.dumps(profile_a),
        profile_b=json.dumps(profile_b),
    )
    for attempt in range(retries):
        await _throttle()
        async with SEM:
            try:
                completion = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    response_format={"type": "json_object"},
                    max_tokens=1800,
                )
                raw = completion.choices[0].message.content
                return json.loads(raw)
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  retry {attempt+1}/{retries} for "
                      f"{profile_a.get('name')} x {profile_b.get('name')}: {e} "
                      f"(waiting {wait}s)")
                await asyncio.sleep(wait)
    raise RuntimeError(f"Date failed after {retries} retries: "
                        f"{profile_a.get('name')} x {profile_b.get('name')}")


async def run_all_dates(profiles: list[dict]) -> list[dict]:
    names = [p["name"] for p in profiles]
    by_name = {p["name"]: p for p in profiles}
    pairs = list(itertools.combinations(names, 2))
    print(f"Running {len(pairs)} dates for {len(names)} people "
          f"(1 call each, throttled to ~{60/MIN_INTERVAL:.0f} req/min)...")

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
        scores[d["a"]].append({
            "match": d["b"], "score": round(final, 1), "reason": va["reason"]
        })
        scores[d["b"]].append({
            "match": d["a"], "score": round(final, 1), "reason": vb["reason"]
        })
    for name in scores:
        scores[name].sort(key=lambda x: -x["score"])
    return scores


async def main():
    profiles = json.load(open("profiles.json"))   # list of Stage 1 outputs
    dates = await run_all_dates(profiles)
    json.dump(dates, open("dates.json", "w"), indent=2)

    rankings = rank(dates, profiles)
    json.dump(rankings, open("rankings.json", "w"), indent=2)

    print(f"\nSaved {len(dates)} dates to dates.json")
    print("Saved rankings to rankings.json")


if __name__ == "__main__":
    asyncio.run(main())