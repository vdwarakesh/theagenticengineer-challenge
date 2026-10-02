import os
import json
from groq import Groq
from dotenv import load_dotenv
from trim import trim_linkedin, trim_instagram

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "qwen/qwen3.8-27b"  # free tier, verify at console.groq.com/docs/models

PROFILE_PROMPT = """You are analyzing a real person using ONLY the data below. Do not invent facts not supported by the data.

LINKEDIN DATA:
{linkedin}

INSTAGRAM DATA:
{instagram}

Note: LinkedIn "followed_topics" reflects what the person follows professionally
(groups, newsletters) — it is a WEAK signal and should not be treated as a personal
hobby unless Instagram content also supports it.

Output ONLY valid JSON, no other text, in exactly this shape:
{{
  "name": "string",
  "summary": "2 sentences about who this person seems to be",
  "hobbies": ["list", "of", "hobbies", "grounded", "in", "instagram", "content"],
  "interests": ["topics", "they", "post", "or", "talk", "about"],
  "career_focus": "one sentence from linkedin",
  "values": ["inferred", "values", "e.g.", "family", "ambition", "creativity"],
  "needs": ["what", "they", "might", "look", "for", "in", "a", "partner"],
  "personality_traits": ["3-5", "traits"],
  "evidence": [
    {{"claim": "loves hiking", "source": "instagram", "detail": "posted hiking photos, caption mentions trail"}},
    {{"claim": "works in philanthropy", "source": "linkedin", "detail": "headline: Chair, Gates Foundation"}}
  ]
}}

Rules:
- Every item in "needs", "values", "personality_traits" must be traceable to the "evidence" list.
- Do not guess sexuality, religion, politics, health, or ethnicity.
- If data is thin, say so in "summary" rather than inventing detail.
- Return ONLY the JSON object. No markdown, no explanation, no code fences.
"""


def analyze_profile(linkedin_trimmed, instagram_trimmed):
    prompt = PROFILE_PROMPT.format(
        linkedin=json.dumps(linkedin_trimmed),
        instagram=json.dumps(instagram_trimmed),
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},  # forces valid JSON, Groq supports this
        max_tokens=1200,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    li_raw = json.load(open("./raw_linkedin.json"))[0]
    ig_raw = json.load(open("./raw_instagram.json"))[0]

    li_trimmed = trim_linkedin(li_raw)
    ig_trimmed = trim_instagram(ig_raw)

    print("=== TRIMMED LINKEDIN ===")
    print(json.dumps(li_trimmed, indent=2))
    print("\n=== TRIMMED INSTAGRAM ===")
    print(json.dumps(ig_trimmed, indent=2))

    print("\n=== CALLING GROQ (llama-3.1-8b-instant) ===")
    raw_response = analyze_profile(li_trimmed, ig_trimmed)
    print(raw_response)

    try:
        parsed = json.loads(raw_response)
        print("\n=== PARSED OK ===")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print("\n=== JSON PARSE FAILED ===")
        print(e)