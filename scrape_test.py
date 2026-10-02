# scrape_test.py
import os, json, asyncio
from dotenv import load_dotenv
from apify_client import ApifyClientAsync

load_dotenv()
apify = ApifyClientAsync(os.environ["APIFY_API_KEY"])

LINKEDIN_URL = "https://www.linkedin.com/in/williamhgates/"
INSTAGRAM_URL = "https://www.instagram.com/thisisbillgates"

async def run_instagram(ig_url: str):
    username = ig_url.rstrip("/").split("/")[-1]
    run = await apify.actor("apify/instagram-profile-scraper").call(
        run_input={"usernames": [username]}
    )
    dataset_id = run.default_dataset_id  # was run["defaultDatasetId"]
    items = (await apify.dataset(dataset_id).list_items()).items
    return items

async def run_linkedin(li_url: str):
    run = await apify.actor("harvestapi/linkedin-profile-scraper").call(
        run_input={"urls": [li_url]}
    )
    dataset_id = run.default_dataset_id
    items = (await apify.dataset(dataset_id).list_items()).items
    return items

async def main():
    ig_data, li_data = await asyncio.gather(
        run_instagram(INSTAGRAM_URL),
        run_linkedin(LINKEDIN_URL),
    )

    with open("raw_instagram.json", "w") as f:
        json.dump(ig_data, f, indent=2)

    with open("raw_linkedin.json", "w") as f:
        json.dump(li_data, f, indent=2)

    print("Instagram items:", len(ig_data))
    print("LinkedIn items:", len(li_data))
    print("Saved to raw_instagram.json and raw_linkedin.json")

if __name__ == "__main__":
    asyncio.run(main())