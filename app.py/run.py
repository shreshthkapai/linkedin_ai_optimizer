#!/usr/bin/env python3
import sys
import os
import subprocess
from dotenv import load_dotenv

def main():
    """
    Loads required environment variables and launches the Streamlit app.

    Ensures that all necessary API keys are set in the .env file before starting
    the application. Exits with an error if any required variables are missing.
    """
    # Load environment variables from a .env file located in the same directory as this script
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
    required_vars = ['HUGGING_FACE_API_KEY', 'APIFY_API_TOKEN']
    # Identify any missing required environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        sys.exit(1)
    # Start the Streamlit application on port 8501, accessible on all network interfaces
    subprocess.run([
        "streamlit", "run", "app/main.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()
