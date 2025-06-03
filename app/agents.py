from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional, List, Dict
import google.generativeai as genai
import os
import json
import re
from datetime import datetime
import time
from prompts import (
    profile_analysis_prompt,
    job_fit_prompt,
    content_enhancement_prompt,
    skill_gap_prompt,
    conversation_summary_prompt,
)
from scraper import scrape_profile

class UnifiedContext(TypedDict):
    """
    Unified context object that bundles all persistent session data.
    
    Combines profile data, conversation history, and user preferences
    into a single coherent context to prevent information loss.
    """
    profile_data: Optional[dict]
    chat_history: List[Dict[str, str]]
    conversation_summary: str
    user_profile: str
    key_insights: List[str]
    session_metadata: Dict[str, any]
    experience_level: str
    total_experience_years: float

class AgentState(TypedDict):
    """
    State object that flows through the agent workflow.
    
    Contains all necessary data for processing LinkedIn profiles and 
    maintaining conversation context across different agent types.
    """
    profile_url: str
    unified_context: UnifiedContext
    user_query: str
    job_role: Optional[str]
    analysis_result: Optional[str]
    session_id: str
    next_node: Optional[str]

def create_unified_context(profile_data: Optional[dict] = None, chat_history: List[Dict[str, str]] = None) -> UnifiedContext:
    """
    Create a new unified context object with all session data bundled together.
    
    Args:
        profile_data: Raw LinkedIn profile data
        chat_history: Previous conversation messages
        
    Returns:
        Unified context object with formatted profile and key insights
    """
    total_years = _calculate_total_experience(profile_data or {})
    experience_level = _determine_experience_level(profile_data or {}, total_years)
    
    context = UnifiedContext(
        profile_data=profile_data or {},
        chat_history=chat_history or [],
        conversation_summary="",
        user_profile=_format_profile_data(profile_data or {}, total_years, experience_level),
        key_insights=[],
        session_metadata={},
        experience_level=experience_level,
        total_experience_years=total_years
    )
    
    if profile_data:
        context["key_insights"] = _extract_key_insights(profile_data, total_years, experience_level)
    
    return context

def _calculate_total_experience(profile_data: dict) -> float:
    """
    Calculate total years of experience across all jobs, handling overlaps and edge cases.
    
    Args:
        profile_data: Raw profile data dictionary
        
    Returns:
        Total experience in years as a float
    """
    if not profile_data.get('experience'):
        return 0.0
    
    periods = []
    current_year = datetime.now().year
    
    for exp in profile_data['experience']:
        if not isinstance(exp, dict):
            continue
        start = exp.get('start_date', '').strip()
        end = exp.get('end_date', 'Present').strip() if exp.get('end_date') else 'Present'
        
        try:
            # Handle "Present" end date
            end = datetime.now().strftime('%b %Y') if end == 'Present' else end
            
            # Parse dates
            start_date = _parse_date(start)
            end_date = _parse_date(end)
            
            # Handle incomplete dates
            if not start_date and start.isdigit():  # e.g., "2023"
                start_date = datetime(int(start), 1, 1)
                end_date = datetime(current_year, 12, 31) if end == 'Present' else (datetime(int(end), 12, 31) if end.isdigit() else None)
            elif not start_date or not end_date:
                # Assume 1 year for jobs with missing or unparsable dates
                start_date = start_date or datetime(current_year - 1, 1, 1)
                end_date = end_date or datetime(current_year, 1, 1)
            
            if start_date and end_date and start_date <= end_date:
                periods.append((start_date, end_date))
        except (ValueError, TypeError):
            # Fallback: assume 1 year for malformed entries
            periods.append((datetime(current_year - 1, 1, 1), datetime(current_year, 1, 1)))
    
    if not periods:
        return 0.0
    
    # Sort periods by start date
    periods.sort(key=lambda x: x[0])
    
    # Merge overlapping periods
    merged = []
    current_start, current_end = periods[0]
    for start, end in periods[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    
    # Calculate total duration
    total_days = sum((end - start).days for start, end in merged)
    return round(total_days / 365.25, 1)  # Convert to years, rounded to 1 decimal

def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse various date formats into a datetime object.
    
    Args:
        date_str: Date string (e.g., "Jul 2024", "2023")
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        if len(date_str.split()) == 2:  # e.g., "Jul 2024"
            return datetime.strptime(date_str, '%b %Y')
        elif date_str.isdigit():  # e.g., "2023"
            return datetime(int(date_str), 1, 1)
        elif '-' in date_str:  # e.g., "Jul 2023 - Dec 2023"
            end_date = date_str.split('-')[-1].strip()
            return datetime.strptime(end_date, '%b %Y') if len(end_date.split()) == 2 else datetime(int(end_date), 12, 31)
        return None
    except ValueError:
        return None

def _determine_experience_level(profile_data: dict, total_experience_years: float) -> str:
    """
    Determine experience level using LLM based on job titles, duration, and context.
    
    Args:
        profile_data: Raw profile data dictionary
        total_experience_years: Calculated total experience in years
        
    Returns:
        Experience level as determined by the LLM
    """
    try:
        titles = [exp.get('title', '').lower() for exp in profile_data.get('experience', []) if isinstance(exp, dict)]
        about = profile_data.get('about', profile_data.get('summary', ''))
        education = [edu.get('degree', '') + ' at ' + edu.get('title', '') for edu in profile_data.get('education', []) if isinstance(edu, dict)]
        certifications = [cert.get('title', '') for cert in profile_data.get('certifications', []) if isinstance(cert, dict)]
        
        prompt = f"""
        You are a career advisor. Based on the data below, classify the user's experience level as Junior, Mid-level, Senior, or Executive. Consider job titles, total experience of {total_experience_years} years, education, certifications, and profile context. Return only the level (e.g., 'Junior').

        Job Titles: {', '.join(titles) or 'None'}
        Total Experience: {total_experience_years} years
        About: {about[:150] + '...' if len(about) > 150 else about}
        Education: {', '.join(education) or 'None'}
        Certifications: {', '.join(certifications) or 'None'}
        """
        
        messages = [{"role": "system", "content": prompt}]
        level = call_llm_api(messages).strip()
        
        # Validate LLM response
        valid_levels = ['Junior', 'Mid-level', 'Senior', 'Executive']
        return level if level in valid_levels else 'Junior'  # Default to Junior if invalid
    except Exception as e:
        print(f"Experience level determination error: {e}")
        return 'Junior'  # Fallback

def _extract_key_insights(profile_data: dict, total_experience_years: float, experience_level: str) -> List[str]:
    """
    Extract key insights from profile data for persistent memory.
    
    Args:
        profile_data: Raw profile data dictionary
        total_experience_years: Calculated total experience in years
        experience_level: Determined experience level
        
    Returns:
        List of key insights to remember across conversations
    """
    insights = []
    
    if profile_data.get("name"):
        insights.append(f"Name: {profile_data['name']}")
    
    if profile_data.get("headline"):
        insights.append(f"Current role: {profile_data['headline']}")
    
    if profile_data.get("experience") and len(profile_data["experience"]) > 0:
        recent_exp = profile_data["experience"][0]
        if isinstance(recent_exp, dict):
            title = recent_exp.get("title", "")
            company = recent_exp.get("company", "")
            if title and company:
                insights.append(f"Latest position: {title} at {company}")
    
    if profile_data.get("skills"):
        skills_count = len(profile_data["skills"]) if isinstance(profile_data["skills"], list) else 0
        skills_list = ", ".join(profile_data["skills"][:5]) if isinstance(profile_data["skills"], list) else "None listed"
        insights.append(f"Skills: {skills_list} ({skills_count} total)")
    
    if profile_data.get("education"):
        recent_edu = profile_data["education"][0] if profile_data["education"] else {}
        if isinstance(recent_edu, dict) and recent_edu.get("title"):
            insights.append(f"Education: {recent_edu['degree']} from {recent_edu['title']}")
    
    if profile_data.get("certifications"):
        certs = [cert.get('title', '') for cert in profile_data["certifications"] if isinstance(cert, dict)][:2]
        if certs:
            insights.append(f"Certifications: {', '.join(certs)}")
    
    insights.append(f"Total Experience: {total_experience_years} years")
    insights.append(f"Experience Level: {experience_level}")
    
    return insights

def manage_conversation_memory(chat_history: List[Dict[str, str]], current_summary: str = "", max_recent_turns: int = 8) -> tuple[List[Dict[str, str]], str]:
    """
    Intelligent memory management that summarizes old conversations while keeping recent ones.
    
    Instead of simple truncation, creates summaries of older conversations to preserve
    important context while preventing token overflow. Maintains recent conversations
    for immediate context continuity.
    
    Args:
        chat_history: Full conversation history
        current_summary: Existing conversation summary
        max_recent_turns: Number of recent conversation turns to keep in full
        
    Returns:
        Tuple of (recent_messages, updated_summary)
    """
    if len(chat_history) <= max_recent_turns * 2:
        return chat_history, current_summary
    
    messages_to_summarize = chat_history[:-max_recent_turns * 2]
    recent_messages = chat_history[-max_recent_turns * 2:]
    
    if messages_to_summarize:
        new_summary = _generate_conversation_summary(messages_to_summarize, current_summary)
        return recent_messages, new_summary
    
    return recent_messages, current_summary

def _generate_conversation_summary(messages: List[Dict[str, str]], existing_summary: str = "") -> str:
    """
    Generate intelligent summary of conversation history using LLM.
    
    Creates concise summaries that preserve important context including
    user goals, previous advice given, and key discussion points.
    
    Args:
        messages: Messages to summarize
        existing_summary: Previous summary to build upon
        
    Returns:
        Intelligent conversation summary
    """
    try:
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        summary_prompt = conversation_summary_prompt.format(
            existing_summary=existing_summary,
            conversation_text=conversation_text
        )
        
        messages_for_api = [{"role": "system", "content": summary_prompt}]
        summary = call_llm_api(messages_for_api)
        
        return summary if summary else existing_summary
        
    except Exception as e:
        print(f"Summary generation error: {e}")
        return existing_summary

def call_llm_api(messages: List[Dict[str, str]], max_retries: int = 3, initial_delay: float = 1.0) -> str:
    """
    Interface to Google's Gemini API for generating responses with retry logic.
    
    Combines message history into a single prompt and calls the Gemini model
    with optimized parameters for conversational responses.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        Generated response text from the model
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            combined_prompt = ""
            for msg in messages:
                role_label = msg['role'].upper()
                combined_prompt += f"{role_label}: {msg['content']}\n\n"
            
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
                return response.text.strip()
            else:
                return "No valid response received."

        except Exception as e:
            print(f"Gemini API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
            else:
                return "I apologize, but I'm having trouble processing your request right now. Please try again."

def scrape_agent(state: AgentState) -> AgentState:
    """
    Entry point agent that scrapes LinkedIn profile data if not already cached.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with unified context containing profile data
    """
    if not state["unified_context"]["profile_data"]:
        profile_data = scrape_profile(state["profile_url"])
        if profile_data:
            existing_chat_history = state["unified_context"]["chat_history"]
            existing_summary = state["unified_context"]["conversation_summary"]
            
            new_context = create_unified_context(profile_data, existing_chat_history)
            new_context["conversation_summary"] = existing_summary
            
            state["unified_context"] = new_context
    
    return state

def profile_analysis_agent(state: AgentState) -> AgentState:
    """Agent specialized in comprehensive LinkedIn profile analysis."""
    return _run_agent_with_prompt(state, profile_analysis_prompt, "analysis")

def job_fit_agent(state: AgentState) -> AgentState:
    """Agent specialized in job role matching and career recommendations."""
    return _run_agent_with_prompt(state, job_fit_prompt, "job_fit")

def content_enhancement_agent(state: AgentState) -> AgentState:
    """Agent specialized in improving LinkedIn profile content and messaging."""
    return _run_agent_with_prompt(state, content_enhancement_prompt, "content")

def skill_gap_agent(state: AgentState) -> AgentState:
    """Agent specialized in identifying skill gaps and learning recommendations."""
    return _run_agent_with_prompt(state, skill_gap_prompt, "skills")

def _run_agent_with_prompt(state: AgentState, prompt_template: str, agent_type: str) -> AgentState:
    """
    Core execution function for all specialized agents.
    
    Handles prompt creation, API calls, and state management for any agent type.
    Uses contextual prompting to adapt responses based on user query specificity.
    
    Args:
        state: Current agent state
        prompt_template: Base prompt template for the agent
        agent_type: Type identifier for specialized handling
        
    Returns:
        Updated state with analysis results and unified context
    """
    context = state["unified_context"]
    
    if not context["profile_data"]:
        state["analysis_result"] = "Profile data missing. Cannot proceed."
        return state

    try:
        recent_history, updated_summary = manage_conversation_memory(
            context["chat_history"], 
            context["conversation_summary"]
        )
        
        contextual_prompt = _create_contextual_prompt(
            prompt_template, 
            context["user_profile"], 
            state["user_query"], 
            state.get("job_role", ""),
            agent_type,
            context["key_insights"],
            updated_summary,
            context["total_experience_years"]
        )

        system_msg = {"role": "system", "content": contextual_prompt}
        messages = [system_msg] + recent_history + [{"role": "user", "content": state["user_query"]}]

        result = call_llm_api(messages)
        validated_result = _validate_response(result)

        updated_history = context["chat_history"] + [
            {"role": "user", "content": state["user_query"]},
            {"role": "assistant", "content": validated_result}
        ]

        state["unified_context"]["chat_history"] = updated_history
        state["unified_context"]["conversation_summary"] = updated_summary
        if state["unified_context"]["profile_data"]:
            state["unified_context"]["key_insights"] = _extract_key_insights(
                state["unified_context"]["profile_data"],
                state["unified_context"]["total_experience_years"],
                state["unified_context"]["experience_level"]
            )
        
        state["analysis_result"] = validated_result
        return state

    except Exception as e:
        state["analysis_result"] = f"Error during AI processing: {str(e)}"
        return state

def _create_contextual_prompt(prompt_template: str, profile_data: str, user_query: str, job_role: str, agent_type: str, key_insights: List[str], conversation_summary: str = "", total_experience_years: float = 0.0) -> str:
    """
    Generate context-aware prompts based on user query complexity and agent type.
    
    Adapts the response style from simple Q&A to detailed analysis based on
    the user's question specificity and the agent's specialization.
    
    Args:
        prompt_template: Base template for the agent
        profile_data: Formatted profile information
        user_query: User's specific question
        job_role: Extracted job role if any
        agent_type: Agent specialization identifier
        key_insights: Important profile insights to remember
        conversation_summary: Summary of previous conversations
        total_experience_years: Total calculated experience in years
        
    Returns:
        Contextually optimized prompt string
    """
    base_context = f"""
You are a helpful LinkedIn career advisor. Answer the user's question naturally and conversationally.

PROFILE DATA:
{profile_data}

KEY PROFILE INSIGHTS TO REMEMBER:
{chr(10).join(f"• {insight}" for insight in key_insights)}

PREVIOUS CONVERSATION SUMMARY:
{conversation_summary if conversation_summary else "This is the start of our conversation."}

IMPORTANT GUIDELINES:
- Answer the user's specific question directly
- Be conversational and natural, not overly structured
- Only provide the level of detail the user is asking for
- If they ask a simple question, give a simple answer
- Use bullet points sparingly and only when truly helpful
- Don't always follow rigid templates - adapt to the conversation
- Remember the key insights above throughout our conversation
- Build on our previous conversations when relevant
- Always refer to the user by their name if available in the profile data
- Tailor recommendations to the user's total experience of {total_experience_years} years
"""

    if agent_type == "job_fit":
        if any(word in user_query.lower() for word in ['quick', 'what', 'which', 'best']):
            base_context += """
- For job role questions: Focus on 3-5 specific job titles that match well
- ALWAYS include match percentages (e.g., "82% match") for each role
- Include brief reasons why each role fits and justify the percentage
- Only mention gaps/improvements if specifically asked
- Ensure recommendations align with the user's total experience
"""
        else:
            base_context += prompt_template.replace("{profile_data}", "").replace("{user_query}", "").replace("{job_role}", "")
    
    elif agent_type == "content":
        if any(word in user_query.lower() for word in ['headline', 'summary', 'specific']):
            base_context += """
- Focus on the specific section they're asking about
- Provide concrete examples and rewrites
- Keep suggestions actionable and specific
"""
        else:
            base_context += prompt_template.replace("{profile_data}", "").replace("{user_query}", "")
    
    elif agent_type == "skills":
        if any(word in user_query.lower() for word in ['what', 'which', 'should']):
            base_context += """
- Focus on the most important skills to develop
- Provide specific, actionable learning suggestions
- Prioritize based on their current profile and goals
- Align recommendations with their total experience
"""
        else:
            base_context += prompt_template.replace("{profile_data}", "").replace("{user_query}", "")
    
    else:
        base_context += """
- Provide balanced feedback focusing on what they specifically asked about
- Be encouraging while being honest about areas for improvement
- Give practical next steps
"""

    return base_context

def _format_profile_data(profile_data: dict, total_experience_years: float, experience_level: str) -> str:
    """
    Transform raw LinkedIn profile data into a structured, readable format.
    
    Extracts and organizes key profile elements for LLM consumption,
    ensuring consistent formatting regardless of source data structure.
    
    Args:
        profile_data: Raw profile data dictionary from scraper
        total_experience_years: Calculated total experience in years
        experience_level: Determined experience level
        
    Returns:
        Formatted string representation of profile data
    """
    if not profile_data:
        return "Profile data not available"
    
    summary_parts = []
    
    if profile_data.get("name"):
        summary_parts.append(f"Name: {profile_data['name']}")
    
    if profile_data.get("headline"):
        summary_parts.append(f"Headline: {profile_data['headline']}")
    
    if profile_data.get("summary") or profile_data.get("about"):
        about = profile_data.get("about", profile_data.get("summary", ""))
        summary_parts.append(f"About: {about[:500]}..." if len(about) > 500 else f"About: {about}")
    
    if profile_data.get("experience"):
        summary_parts.append("Recent Experience:")
        for exp in profile_data["experience"][:4]:
            if isinstance(exp, dict):
                title = exp.get("title", "")
                company = exp.get("company", "")
                duration = exp.get("duration", f"{exp.get('start_date', '')} - {exp.get('end_date', '')}")
                if title and company:
                    summary_parts.append(f"  - {title} at {company} ({duration})")
    
    if profile_data.get("skills"):
        skills_list = profile_data["skills"]
        if isinstance(skills_list, list) and skills_list:
            skills_names = [str(skill) for skill in skills_list[:15] if skill]
            summary_parts.append(f"Skills: {', '.join(skills_names)}")
    
    if profile_data.get("education"):
        summary_parts.append("Education:")
        for edu in profile_data["education"][:2]:
            if isinstance(edu, dict):
                school = edu.get("title", edu.get("school", ""))
                degree = edu.get("degree", "")
                if school:
                    summary_parts.append(f"  - {degree} from {school}" if degree else f"  - {school}")
    
    if profile_data.get("certifications"):
        certs = [cert.get('title', '') for cert in profile_data["certifications"] if isinstance(cert, dict)][:2]
        if certs:
            summary_parts.append(f"Certifications: {', '.join(certs)}")
    
    summary_parts.append(f"Total Experience: {total_experience_years} years")
    summary_parts.append(f"Experience Level: {experience_level}")
    
    return "\n".join(summary_parts) if summary_parts else "Limited profile information available"

def _validate_response(response: str) -> str:
    """
    Ensure response quality and add specialized formatting for certain response types.
    
    Validates response length, adds match percentage summaries for job fit responses,
    and provides fallback messages for inadequate responses.
    
    Args:
        response: Raw response from the LLM
        
    Returns:
        Validated and potentially enhanced response
    """
    if not response or not isinstance(response, str):
        return "Sorry, I couldn't generate a proper response. Could you try rephrasing your question?"
    
    if len(response.strip()) < 30:
        return "Could you provide more details about what you'd like to know? I'd be happy to give you a more comprehensive answer."
    
    if "match" in response.lower() or "role" in response.lower():
        match_scores = re.findall(r"\b\d{1,3}%\b", response)
        if match_scores:
            response += f"\n\n📊 **Match Summary:** {', '.join(match_scores)}"
    
    return response.strip()

def route_agent(state: AgentState) -> AgentState:
    """
    Intelligent routing system that directs queries to the most appropriate specialized agent.
    
    Analyzes user intent from natural language queries and routes to the best-suited
    agent for handling specific types of career advice requests.
    
    Args:
        state: Current agent state with user query
        
    Returns:
        Updated state with next_node set for workflow routing
    """
    try:
        user_query = state['user_query'].lower()
        
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
