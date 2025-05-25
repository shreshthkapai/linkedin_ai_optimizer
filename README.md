# LinkedIn AI Optimizer 💼
An AI-powered chat system that helps users optimize their LinkedIn profiles, analyze job fit, and receive personalized career guidance. Built with Streamlit, LangGraph, and powered by HuggingFace AI models.

## ✨ Features
1) Interactive Chat Interface: Input your LinkedIn profile URL and chat with AI for personalized feedback.
2) Profile Analysis: Comprehensive evaluation of LinkedIn sections (About, Experience, Skills) with actionable improvements.
3) Job Fit Analysis: Compare your profile against target roles with match scores and gap analysis.
4) Content Enhancement: AI-generated rewrites of profile sections for better impact and keyword optimization.
5) Skill Gap Analysis: Identify missing skills and get learning recommendations with specific resources.
6) Persistent Memory: ChatGPT-like conversation memory with intelligent truncation (maintains last 10 conversation turns).
7) Session Management: Context retention across chat sessions using LangGraph checkpointers.

## 🌐 Live Demo
Try it live: https://linkedinaioptimizer-c3l9qgsowlfphztd7f6wyd.streamlit.app/

## 🚀 Quick Start
### Prerequisites
1) Python 3.11+
2) Apify API token (free tier available)
3) Gemma API key (free)

### Installation
1) Clone the repository
```sh
git clone https://github.com/your-username/linkedin-ai-optimizer.git
cd linkedin-ai-optimizer
```
2) Install dependencies
```sh
pip install -r requirements.txt
```
3) Set up environment variables- Create app/.env file
```sh
APIFY_API_TOKEN=your_apify_token_here
GEMINI_FACE_API_KEY=your_gemini_token_here
LI_AT_COOKIE=your_linkedin_cookie_here (optional)
```
4) Run the application
```sh
streamlit run app/main.py
```
Or use the helper script:
```sh
python app/run.py
```
The app will be available at http://localhost:8501

## 🎯 How to Use
Enter LinkedIn URL: Paste your LinkedIn profile URL in the input field
Start Chatting: Ask questions about your profile, career guidance, or job fit analysis
Get AI Insights: Receive personalized recommendations and actionable feedback

## Example Questions
"Analyze my LinkedIn profile and suggest improvements."
"How well does my profile match a Software Engineer role?"
"Rewrite my About section for better impact."
"What skills am I missing for a Data Scientist position?"

## 🏗️ Architecture
1) Multi-Agent System: LangGraph-powered agents for specialized tasks (analysis, content generation, job matching)
2) Memory Management: Session-based context retention for personalized conversations
3) Profile Scraping: Apify integration for LinkedIn data extraction
4) AI Processing: Gemini-1.5 Flash model for intelligent responses

## For Streamlit Cloud Deployment
Set secrets in .streamlit/secrets.toml:
```sh
APIFY_API_TOKEN = "your_token"
GEMINI_FACE_API_KEY = "your_token"
LI_AT_COOKIE = "your_cookie"
```

## Environment Variables
1) APIFY_API_TOKEN: Required for LinkedIn profile scraping
2) GEMINI_API_KEY: Required for AI model access
3) LI_AT_COOKIE: Optional, improves scraping reliability

## 📁 Project Structure
```
app/
├── main.py              # Streamlit application entry point
├── chat_handler.py      # Chat orchestration and workflow management
├── agents.py            # Multi-agent system with specialized AI agents
├── scraper.py           # LinkedIn profile data extraction
├── prompts.py           # AI prompt templates for different tasks
└── .env                 # Environment variables (create this)
```
## 🛠️ Tech Stack
1) Frontend: Streamlit
2) AI Framework: LangGraph, Gemini API
3) Data Scraping: Apify LinkedIn Scraper
4) AI Model: Gemini-1.5-flash
5) Memory: LangGraph MemorySaver

## 📋 Requirements
See requirements.txt for the complete dependency list. Key packages:
1) streamlit
2) langgraph
3) apify-client
4) Gemini API
5) python-dotenv

## 🔗 APIs Used
1) Apify LinkedIn Scraper: Profile data extraction
2) GEMINI API: AI model hosting and inference
   
