import os
from dotenv import load_dotenv
from apify_client import ApifyClient
from typing import Dict, Optional, Any
import re

load_dotenv()

def scrape_profile(profile_url: str) -> Optional[Dict[str, Any]]:
    """Scrape raw LinkedIn profile data using Apify and validate critical fields."""
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
            profile_data = dict(item)
            # Validate and clean critical fields
            profile_data = _validate_profile_data(profile_data)
            return profile_data
        print("No data returned from dataset")
        return None
    except Exception as e:
        print(f"Error scraping profile: {e}")
        return None

def _validate_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean scraped profile data to ensure critical fields are present."""
    cleaned_data = profile_data.copy()
    
    # Ensure critical fields exist with defaults
    required_fields = ['name', 'about', 'experience', 'education', 'skills']
    for field in required_fields:
        if field not in cleaned_data or not cleaned_data[field]:
            cleaned_data[field] = [] if field in ['experience', 'education', 'skills'] else ""
    
    # Infer skills from 'about' or 'activity' if skills list is empty
    if not cleaned_data['skills']:
        inferred_skills = _infer_skills(cleaned_data)
        cleaned_data['skills'] = inferred_skills
    
    # Ensure experience and education are lists
    if not isinstance(cleaned_data['experience'], list):
        cleaned_data['experience'] = []
    if not isinstance(cleaned_data['education'], list):
        cleaned_data['education'] = []
    
    # Clean up malformed experience entries
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
    
    # Expanded skill keywords for broader inference
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
            if re.search(rf'\b{skill}\b', text, re.IGNORECASE) and skill not in skills:
                skills.append(skill.title())
    
    return skills 
