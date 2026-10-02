import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from apify_client import ApifyClientAsync

load_dotenv()
apify = ApifyClientAsync(os.environ["APIFY_API_KEY"])


async def run_instagram(ig_url: str):
    username = ig_url.rstrip("/").split("/")[-1]
    run = await apify.actor("apify/instagram-profile-scraper").call(
        run_input={"usernames": [username]}
    )
    items = (await apify.dataset(run.default_dataset_id).list_items()).items
    return items


async def run_linkedin(li_url: str):
    run = await apify.actor("harvestapi/linkedin-profile-scraper").call(
        run_input={"urls": [li_url]}
    )
    items = (await apify.dataset(run.default_dataset_id).list_items()).items
    return items


async def main(slug: str, linkedin_url: str, instagram_url: str):
    os.makedirs("raw", exist_ok=True)
    ig_data, li_data = await asyncio.gather(
        run_instagram(instagram_url),
        run_linkedin(linkedin_url),
    )

    ig_path = f"raw/{slug}_instagram.json"
    li_path = f"raw/{slug}_linkedin.json"
    json.dump(ig_data, open(ig_path, "w"), indent=2)
    json.dump(li_data, open(li_path, "w"), indent=2)

    print(f"Instagram items: {len(ig_data)} -> {ig_path}")
    print(f"LinkedIn items: {len(li_data)} -> {li_path}")

    if ig_data and ig_data[0].get("private"):
        print("WARNING: this Instagram account is PRIVATE. Pick a different person.")
    if ig_data and ig_data[0].get("postsCount", 0) == 0:
        print("WARNING: this Instagram account has 0 posts. Profile will be very thin.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scrape_person.py <slug> <linkedin_url> <instagram_url>")
        print('Example: python scrape_person.py jane_doe '
              '"https://www.linkedin.com/in/jane-doe/" '
              '"https://www.instagram.com/jane.doe/"')
        sys.exit(1)
    _, slug, li_url, ig_url = sys.argv
    asyncio.run(main(slug, li_url, ig_url))