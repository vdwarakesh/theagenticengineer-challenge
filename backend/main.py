import os
import json
import uuid
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apify_client import ApifyClientAsync
from groq import AsyncGroq
from dotenv import load_dotenv
from trim import trim_linkedin, trim_instagram

load_dotenv()
apify = ApifyClientAsync(os.environ["APIFY_API_KEY"])
groq_async = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

ANALYZE_MODEL = "qwen/qwen3.8-27b"
DIALOGUE_MODEL = "openai/gpt-oss-20b"
VERDICT_MODEL = "qwen/qwen3.8-27b"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

JOBS: dict[str, dict] = {}

SEM = asyncio.Semaphore(4)
MIN_INTERVAL = 2.2
_last_call_time = 0.0
_rate_lock = asyncio.Lock()


async def _throttle():
    global _last_call_time
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        wait = MIN_INTERVAL - (now - _last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time = asyncio.get_event_loop().time()


async def _call(model, prompt, max_tokens, reasoning_effort=None, retries=3):
    last_err = None
    for attempt in range(retries):
        await _throttle()
        async with SEM:
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                completion = await groq_async.chat.completions.create(**kwargs)
                return json.loads(completion.choices[0].message.content)
            except Exception as e:
                last_err = e
                await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Call failed after {retries} retries: {last_err}")


# ---------- Scrape ----------
async def scrape(linkedin_url: str, instagram_url: str):
    username = instagram_url.rstrip("/").split("/")[-1]
    ig_run, li_run = await asyncio.gather(
        apify.actor("apify/instagram-profile-scraper").call(run_input={"usernames": [username]}),
        apify.actor("harvestapi/linkedin-profile-scraper").call(run_input={"urls": [linkedin_url]}),
    )
    ig_items = (await apify.dataset(ig_run.default_dataset_id).list_items()).items
    li_items = (await apify.dataset(li_run.default_dataset_id).list_items()).items
    if not ig_items or not li_items:
        raise ValueError(f"Could not scrape {linkedin_url} / {instagram_url}. Check both are public.")
    ig = ig_items[0]
    if ig.get("private"):
        raise ValueError(f"Instagram {instagram_url} is private. Only public profiles work.")
    return li_items[0], ig


PROFILE_PROMPT = """You are analyzing a real person using ONLY the data below. Do not invent facts not supported by the data.

LINKEDIN DATA:
{linkedin}

INSTAGRAM DATA:
{instagram}

Note: LinkedIn "followed_topics" reflects what the person follows professionally — a WEAK
signal, not a personal hobby unless Instagram also supports it.

Output ONLY valid JSON:
{{
  "name": "string",
  "summary": "2 sentences about who this person seems to be",
  "hobbies": ["..."],
  "interests": ["..."],
  "career_focus": "one sentence",
  "values": ["..."],
  "needs": ["..."],
  "personality_traits": ["3-5 traits"],
  "evidence": [
    {{"claim": "...", "source": "linkedin|instagram", "detail": "..."}}
  ]
}}

Rules:
- Every "needs", "values", "personality_traits", "interests", "hobbies" item must trace to "evidence".
- Never guess sexuality, religion, politics, health, or ethnicity.
- If data is thin, say so in "summary" instead of inventing detail.
- Return ONLY the JSON object.
"""


async def analyze(li_raw, ig_raw):
    li_t = trim_linkedin(li_raw)
    ig_t = trim_instagram(ig_raw)
    profile = await _call(
        ANALYZE_MODEL,
        PROFILE_PROMPT.format(linkedin=json.dumps(li_t), instagram=json.dumps(ig_t)),
        max_tokens=1500,
    )
    profile["location"] = li_t.get("location")
    profile["linkedin_url"] = li_raw.get("linkedinUrl")
    profile["instagram_url"] = f"https://www.instagram.com/{ig_raw.get('username')}"
    return profile


DIALOGUE_PROMPT = """Write a short first-date conversation between two people, based ONLY
on their profiles. Never invent facts not present in a profile.

PERSON A: {profile_a}
PERSON B: {profile_b}

Exactly 6 lines, alternating, starting with A. 1-3 sentences each, true to tone/interests.
Output ONLY: {{"transcript": [{{"speaker":"A","text":"..."}}, ...]}}
"""

VERDICT_PROMPT = """Two people went on a first date based on their real profiles. Evaluate
from both private perspectives.

PERSON A: {profile_a}
PERSON B: {profile_b}
TRANSCRIPT: {transcript}

Output ONLY valid JSON:
{{
  "verdict_a_about_b": {{"score":0,"chemistry":"","shared_ground":[],"friction":[],"would_meet_again":true,"reason":""}},
  "verdict_b_about_a": {{"score":0,"chemistry":"","shared_ground":[],"friction":[],"would_meet_again":true,"reason":""}}
}}
Never comment on sexuality, religion, politics, health, or ethnicity.
"""


async def date_pair(a: dict, b: dict) -> dict:
    dialogue = await _call(
        DIALOGUE_MODEL,
        DIALOGUE_PROMPT.format(profile_a=json.dumps(a), profile_b=json.dumps(b)),
        max_tokens=700, reasoning_effort="low",
    )
    verdicts = await _call(
        VERDICT_MODEL,
        VERDICT_PROMPT.format(
            profile_a=json.dumps(a), profile_b=json.dumps(b),
            transcript=json.dumps(dialogue["transcript"]),
        ),
        max_tokens=900,
    )
    same_city = bool(a.get("location") and a.get("location") == b.get("location"))
    va, vb = verdicts["verdict_a_about_b"], verdicts["verdict_b_about_a"]
    bonus = 5 if (va.get("would_meet_again") and vb.get("would_meet_again")) else 0
    bonus += 5 if same_city else 0
    final_score = round((va["score"] + vb["score"]) / 2 + bonus, 1)
    return {**dialogue, **verdicts, "same_city": same_city, "final_score": final_score}


async def run_compatibility_job(job_id, li_a_url, ig_a_url, li_b_url, ig_b_url):
    try:
        JOBS[job_id]["status"] = "scraping"
        (li_a_raw, ig_a_raw), (li_b_raw, ig_b_raw) = await asyncio.gather(
            scrape(li_a_url, ig_a_url),
            scrape(li_b_url, ig_b_url),
        )

        JOBS[job_id]["status"] = "analyzing"
        profile_a, profile_b = await asyncio.gather(
            analyze(li_a_raw, ig_a_raw),
            analyze(li_b_raw, ig_b_raw),
        )
        JOBS[job_id]["profile_a"] = profile_a
        JOBS[job_id]["profile_b"] = profile_b

        JOBS[job_id]["status"] = "dating"
        result = await date_pair(profile_a, profile_b)
        JOBS[job_id]["result"] = result

        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


class CompatibilityIn(BaseModel):
    linkedin_url_a: str
    instagram_url_a: str
    linkedin_url_b: str
    instagram_url_b: str


@app.post("/api/compatibility")
async def check_compatibility(body: CompatibilityIn):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued"}
    asyncio.create_task(run_compatibility_job(
        job_id, body.linkedin_url_a, body.instagram_url_a,
        body.linkedin_url_b, body.instagram_url_b,
    ))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})