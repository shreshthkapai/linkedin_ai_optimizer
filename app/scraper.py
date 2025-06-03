import os
import re
from apify_client import ApifyClient
from typing import Dict, Optional, Any

# Load secrets safely (Streamlit Cloud or local .env fallback)
try:
    import streamlit as st
    APIFY_API_TOKEN = st.secrets["APIFY_API_TOKEN"]
    LI_AT_COOKIE = st.secrets["LI_AT_COOKIE"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
    LI_AT_COOKIE = os.getenv("LI_AT_COOKIE")


def scrape_profile(profile_url: str) -> Optional[Dict[str, Any]]:
    """Scrape raw LinkedIn profile data using Apify and validate critical fields."""
    try:
        print("⚡️ scrape_profile() called with:", profile_url)

        if not APIFY_API_TOKEN:
            raise ValueError("APIFY_API_TOKEN is missing!")

        run_input = {
            "url": profile_url,
            "cookie": [{"name": "li_at", "value": LI_AT_COOKIE}] if LI_AT_COOKIE else []
        }

        client = ApifyClient(APIFY_API_TOKEN)
        print(f"🚀 Running Apify actor with input: {run_input}")
        run = client.actor("pratikdani/linkedin-people-profile-scraper").call(run_input=run_input)

        if "defaultDatasetId" not in run:
            raise RuntimeError("Apify run did not return a dataset ID!")

        print(f"✅ Apify run status: {run['status']}")
        print(f"📊 View dataset: https://console.apify.com/storage/datasets/{run['defaultDatasetId']}")

        dataset_id = run["defaultDatasetId"]
        for item in client.dataset(dataset_id).iterate_items():
            profile_data = dict(item)
            validated = _validate_profile_data(profile_data)
            print("✅ Scraped profile data (preview):", {k: validated.get(k) for k in ["name", "headline", "skills"]})
            return validated

        print("⚠️ No items returned from Apify dataset.")
        return None

    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return None


def _validate_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean scraped profile data to ensure critical fields are present."""
    cleaned_data = profile_data.copy()
    required_fields = ['name', 'about', 'experience', 'education', 'skills']

    for field in required_fields:
        if field not in cleaned_data or not cleaned_data[field]:
            cleaned_data[field] = [] if field in ['experience', 'education', 'skills'] else ""

    if not cleaned_data['skills']:
        cleaned_data['skills'] = _infer_skills(cleaned_data)

    if not isinstance(cleaned_data['experience'], list):
        cleaned_data['experience'] = []
    if not isinstance(cleaned_data['education'], list):
        cleaned_data['education'] = []

    cleaned_data['experience'] = [
        exp for exp in cleaned_data['experience']
        if isinstance(exp, dict) and exp.get('title') and exp.get('company')
    ]

    return cleaned_data


def _infer_skills(profile_data: Dict[str, Any]) -> list:
    """Infer skills from 'about' and 'activity' sections if skills list is missing."""
    skills = []
    text_sources = [profile_data.get('about', '')]
    if profile_data.get('activity'):
        text_sources.extend([act.get('title', '') for act in profile_data['activity']])

    skill_keywords = [
        'machine learning', 'deep learning', 'artificial intelligence', 'ai',
        'python', 'data science', 'software engineering', 'programming',
        'data analysis', 'cloud computing', 'llm', 'generative ai',
        'tensorflow', 'pytorch', 'sql', 'java', 'javascript', 'aws',
        'azure', 'gcp', 'docker', 'kubernetes', 'rlhf', 'reinforcement learning',
        'natural language processing', 'computer vision', 'big data', 'spark',
        'hadoop', 'git', 'agile', 'scrum', 'devops', 'ci/cd'
    ]

    for text in text_sources:
        if not text:
            continue
        for skill in skill_keywords:
            if re.search(rf'\b{skill}\b', text, re.IGNORECASE) and skill.title() not in skills:
                skills.append(skill.title())

    return skills
