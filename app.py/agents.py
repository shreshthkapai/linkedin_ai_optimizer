from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional, List, Dict
import google.generativeai as genai
import os
import json
import re
from prompts import (
    profile_analysis_prompt,
    job_fit_prompt,
    content_enhancement_prompt,
    skill_gap_prompt,
)
from scraper import scrape_profile

class AgentState(TypedDict):
    """Typed dictionary representing the state passed between agents in the workflow."""
    profile_url: str
    profile_data: Optional[dict]
    user_query: str
    job_role: Optional[str]
    analysis_result: Optional[str]
    session_id: str
    chat_history: List[Dict[str, str]]
    next_node: Optional[str]

def truncate_chat_history(chat_history: List[Dict[str, str]], max_turns=15) -> List[Dict[str, str]]:
    """Truncate chat history to the most recent `max_turns` user/assistant exchanges for context window management."""
    if len(chat_history) <= max_turns * 2:
        return chat_history
    return chat_history[-max_turns * 2:]

def call_llm_api(messages: List[Dict[str, str]]) -> str:
    """
    Send a conversation history to the Gemini LLM API and return the assistant's reply.
    Raises if the API key is missing.
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Combine all chat messages into a single prompt for the LLM
        combined_prompt = ""
        for msg in messages:
            role_label = msg['role'].upper()
            combined_prompt += f"{role_label}: {msg['content']}\n\n"

        # Add a final instruction to ensure LLM knows when to respond
        combined_prompt += "ASSISTANT: "

        response = model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.4,
                top_p=0.9,
            )
        )

        if response.text:
            result = response.text.strip()
        else:
            result = "No valid response received."

        # Return as-is for more natural LLM flow
        return result

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."

def scrape_agent(state: AgentState) -> AgentState:
    """
    Fetch LinkedIn profile data if not already present in the state.
    """
    if not state.get("profile_data"):
        profile_data = scrape_profile(state["profile_url"])
        state["profile_data"] = profile_data
    return state

def profile_analysis_agent(state: AgentState) -> AgentState:
    """Run the profile analysis prompt using the agent."""
    return _run_agent_with_prompt(state, profile_analysis_prompt, "analysis")

def job_fit_agent(state: AgentState) -> AgentState:
    """Run the job fit prompt using the agent."""
    return _run_agent_with_prompt(state, job_fit_prompt, "job_fit")

def content_enhancement_agent(state: AgentState) -> AgentState:
    """Run the content enhancement prompt using the agent."""
    return _run_agent_with_prompt(state, content_enhancement_prompt, "content")

def skill_gap_agent(state: AgentState) -> AgentState:
    """Run the skill gap prompt using the agent."""
    return _run_agent_with_prompt(state, skill_gap_prompt, "skills")

def _run_agent_with_prompt(state: AgentState, prompt_template: str, agent_type: str) -> AgentState:
    """
    General-purpose agent runner which:
    - Formats profile data,
    - Builds a contextual prompt,
    - Calls the LLM,
    - Validates and stores the result.
    """
    if not state.get("profile_data"):
        state["analysis_result"] = "Profile data missing. Cannot proceed."
        return state

    try:
        profile_summary = _format_profile_data(state["profile_data"])

        # Adjust prompt based on agent type and question specificity
        contextual_prompt = _create_contextual_prompt(
            prompt_template, 
            profile_summary, 
            state["user_query"], 
            state.get("job_role", ""),
            agent_type
        )

        system_msg = {"role": "system", "content": contextual_prompt}
        history = truncate_chat_history(state.get("chat_history", []))
        messages = [system_msg] + history + [{"role": "user", "content": state["user_query"]}]

        result = call_llm_api(messages)
        validated_result = _validate_response(result)

        updated_history = history + [
            {"role": "user", "content": state["user_query"]},
            {"role": "assistant", "content": validated_result}
        ]

        state["chat_history"] = updated_history
        state["analysis_result"] = validated_result
        return state

    except Exception as e:
        state["analysis_result"] = f"Error during AI processing: {str(e)}"
        return state

def _create_contextual_prompt(prompt_template: str, profile_data: str, user_query: str, job_role: str, agent_type: str) -> str:
    """
    Build a detailed, contextual prompt for the LLM based on agent type and the user's query.
    """
    base_context = f""" ... """  # [Keep your existing string]
    # [Keep the rest of your logic unchanged, but leave this comment as a marker for context construction.]
    return base_context

def _format_profile_data(profile_data: dict) -> str:
    """
    Summarize the profile data dictionary into a readable string for prompt context.
    Truncates long fields and limits experience/skills/education for brevity.
    """
    if not profile_data:
        return "Profile data not available"

    summary_parts = []

    if profile_data.get("name"):
        summary_parts.append(f"Name: {profile_data['name']}")
    if profile_data.get("headline"):
        summary_parts.append(f"Headline: {profile_data['headline']}")
    if profile_data.get("summary"):
        summary_parts.append(f"Summary: {profile_data['summary'][:500]}...")  # Truncate long summaries

    if profile_data.get("experience"):
        summary_parts.append("Recent Experience:")
        for exp in profile_data["experience"][:4]:
            if isinstance(exp, dict):
                title = exp.get("title", "")
                company = exp.get("company", "")
                duration = exp.get("duration", "")
                if title and company:
                    summary_parts.append(f"  - {title} at {company} {f'({duration})' if duration else ''}")

    if profile_data.get("skills"):
        skills_list = profile_data["skills"]
        if isinstance(skills_list, list) and skills_list:
            if isinstance(skills_list[0], dict):
                skills_names = [skill.get("name", "") for skill in skills_list[:15]]
            else:
                skills_names = skills_list[:15]
            summary_parts.append(f"Skills: {', '.join(str(s) for s in skills_names if s)}")

    if profile_data.get("education"):
        education_list = profile_data["education"]
        if isinstance(education_list, list) and education_list:
            summary_parts.append("Education:")
            for edu in education_list[:2]:
                if isinstance(edu, dict):
                    school = edu.get("school", "")
                    degree = edu.get("degree", "")
                    if school:
                        summary_parts.append(f"  - {degree} from {school}" if degree else f"  - {school}")

    return "\n".join(summary_parts) if summary_parts else "Limited profile information available"

def _validate_response(response: str) -> str:
    """
    Checks if the LLM's response is appropriate and complete.
    Adds a summary of match percentages if found in job fit responses.
    """
    if not response or not isinstance(response, str):
        return "Sorry, I couldn't generate a proper response. Could you try rephrasing your question?"

    if len(response.strip()) < 30:
        return "Could you provide more details about what you'd like to know? I'd be happy to give you a more comprehensive answer."

    # For job fit, parse and highlight match percentages if present
    if "match" in response.lower() or "role" in response.lower():
        match_scores = re.findall(r"\b\d{1,3}%\b", response)
        if match_scores:
            response += f"\n\n📊 **Match Summary:** {', '.join(match_scores)}"

    return response.strip()

def route_agent(state: AgentState) -> AgentState:
    """
    Route the user query to the appropriate agent node based on detected intent.
    """
    try:
        user_query = state['user_query'].lower()

        # Route based on keywords, fallback to profile analysis if ambiguous
        if any(phrase in user_query for phrase in ['job', 'roles', 'position', 'career', 'suited', 'fit', 'work as', 'good for']):
            state["next_node"] = "job_fit"
        elif any(phrase in user_query for phrase in ['improve', 'enhance', 'better', 'rewrite', 'content', 'headline', 'summary', 'description']):
            state["next_node"] = "content_enhancement"
        elif any(phrase in user_query for phrase in ['skill', 'learn', 'gap', 'missing', 'development', 'course', 'training']):
            state["next_node"] = "skill_gap"
        elif any(phrase in user_query for phrase in ['analyze', 'review', 'feedback', 'thoughts', 'look', 'profile']):
            state["next_node"] = "profile_analysis"
        else:
            state["next_node"] = "profile_analysis"

        return state
    except Exception as e:
        print(f"Routing error: {e}")
        state["next_node"] = "profile_analysis"
        return state
