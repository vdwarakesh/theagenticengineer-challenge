import os
import sys
import json
from groq import Groq
from dotenv import load_dotenv
from trim import trim_linkedin, trim_instagram

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.8-27b"

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
- Every item in "needs", "values", "personality_traits", "interests", "hobbies" must be
  traceable to the "evidence" list.
- Do not guess sexuality, religion, politics, health, or ethnicity.
- If data is thin, say so in "summary" rather than inventing detail.
- Return ONLY the JSON object. No markdown, no explanation, no code fences.
"""


def analyze(linkedin_trimmed, instagram_trimmed):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROFILE_PROMPT.format(
                linkedin=json.dumps(linkedin_trimmed),
                instagram=json.dumps(instagram_trimmed),
            ),
        }],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=1500,
    )
    return json.loads(completion.choices[0].message.content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_person.py <slug>")
        print("Expects raw/<slug>_linkedin.json and raw/<slug>_instagram.json to exist.")
        sys.exit(1)
    slug = sys.argv[1]

    li_raw = json.load(open(f"raw/{slug}_linkedin.json"))[0]
    ig_raw = json.load(open(f"raw/{slug}_instagram.json"))[0]

    li_trimmed = trim_linkedin(li_raw)
    ig_trimmed = trim_instagram(ig_raw)

    profile = analyze(li_trimmed, ig_trimmed)

    os.makedirs("profiles", exist_ok=True)
    out_path = f"profiles/{slug}.json"
    json.dump(profile, open(out_path, "w"), indent=2)
    print(f"Saved profile -> {out_path}")
    print(json.dumps(profile, indent=2))