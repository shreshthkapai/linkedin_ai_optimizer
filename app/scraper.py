import os
from dotenv import load_dotenv
from apify_client import ApifyClient
from typing import Dict, Optional, Any

load_dotenv()

def scrape_profile(profile_url: str) -> Optional[Dict[str, Any]]:
    """Scrape raw LinkedIn profile data using Apify."""
    try:
        apify_token = os.getenv("APIFY_API_TOKEN")
        if not apify_token:
            raise ValueError("APIFY_API_TOKEN not found in .env")
        
        li_at_cookie = os.getenv("LI_AT_COOKIE")
        run_input = {
            "url": profile_url,
            "cookie": [{"name": "li_at", "value": li_at_cookie}] if li_at_cookie else []
        }
        
        client = ApifyClient(apify_token)
        print(f"Running actor with input: {run_input}")
        run = client.actor("pratikdani/linkedin-people-profile-scraper").call(run_input=run_input)
        print(f"Run status: {run['status']}")
        print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{run['defaultDatasetId']}")
        
        dataset_id = run["defaultDatasetId"]
        for item in client.dataset(dataset_id).iterate_items():
            return dict(item)  # Return raw dictionary
        print("No data returned from dataset")
        return None
    except Exception as e:
        print(f"Error scraping profile: {e}")
        return None