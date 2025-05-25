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
    """
    State object that flows through the agent workflow.
    
    Contains all necessary data for processing LinkedIn profiles and 
    maintaining conversation context across different agent types.
    """
    profile_url: str
    profile_data: Optional[dict]
    user_query: str
    job_role: Optional[str]
    analysis_result: Optional[str]
    session_id: str
    chat_history: List[Dict[str, str]]
    next_node: Optional[str]

def truncate_chat_history(chat_history: List[Dict[str, str]], max_turns=15) -> List[Dict[str, str]]:
    """
    Limit chat history to prevent context window overflow.
    
    Args:
        chat_history: List of conversation messages
        max_turns: Maximum number of conversation turns to keep
        
    Returns:
        Truncated chat history maintaining recent context
    """
    if len(chat_history) <= max_turns * 2:
        return chat_history
    return chat_history[-max_turns * 2:]

def call_llm_api(messages: List[Dict[str, str]]) -> str:
    """
    Interface to Google's Gemini API for generating responses.
    
    Combines message history into a single prompt and calls the Gemini model
    with optimized parameters for conversational responses.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Generated response text from the model
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Combine all messages into a single prompt for better context understanding
        combined_prompt = ""
        for msg in messages:
            role_label = msg['role'].upper()
            combined_prompt += f"{role_label}: {msg['content']}\n\n"
        
        combined_prompt += "ASSISTANT: "
        
        response = model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,  # Balanced for detailed responses
                temperature=0.4,         # Slightly creative but focused
                top_p=0.9,
            )
        )

        if response.text:
            result = response.text.strip()
        else:
            result = "No valid response received."

        return result

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."

def scrape_agent(state: AgentState) -> AgentState:
    """
    Entry point agent that scrapes LinkedIn profile data if not already cached.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with profile data
    """
    if not state.get("profile_data"):
        profile_data = scrape_profile(state["profile_url"])
        state["profile_data"] = profile_data
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
        Updated state with analysis results and chat history
    """
    if not state.get("profile_data"):
        state["analysis_result"] = "Profile data missing. Cannot proceed."
        return state

    try:
        profile_summary = _format_profile_data(state["profile_data"])
        
        # Create contextual prompt based on user query specificity
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

        # Update conversation history for context continuity
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
    Generate context-aware prompts based on user query complexity and agent type.
    
    Adapts the response style from simple Q&A to detailed analysis based on
    the user's question specificity and the agent's specialization.
    
    Args:
        prompt_template: Base template for the agent
        profile_data: Formatted profile information
        user_query: User's specific question
        job_role: Extracted job role if any
        agent_type: Agent specialization identifier
        
    Returns:
        Contextually optimized prompt string
    """
    
    base_context = f"""
You are a helpful LinkedIn career advisor. Answer the user's question naturally and conversationally.

PROFILE DATA:
{profile_data}

IMPORTANT GUIDELINES:
- Answer the user's specific question directly
- Be conversational and natural, not overly structured
- Only provide the level of detail the user is asking for
- If they ask a simple question, give a simple answer
- Use bullet points sparingly and only when truly helpful
- Don't always follow rigid templates - adapt to the conversation
"""

    # Specialized handling based on agent type and query complexity
    if agent_type == "job_fit":
        if any(word in user_query.lower() for word in ['quick', 'what', 'which', 'best']):
            base_context += """
- For job role questions: Focus on 3-5 specific job titles that match well
- ALWAYS include match percentages (e.g., "82% match") for each role
- Include brief reasons why each role fits and justify the percentage
- Only mention gaps/improvements if specifically asked
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
"""
        else:
            base_context += prompt_template.replace("{profile_data}", "").replace("{user_query}", "")
    
    else:  # analysis
        base_context += """
- Provide balanced feedback focusing on what they specifically asked about
- Be encouraging while being honest about areas for improvement
- Give practical next steps
"""

    return base_context

def _format_profile_data(profile_data: dict) -> str:
    """
    Transform raw LinkedIn profile data into a structured, readable format.
    
    Extracts and organizes key profile elements for LLM consumption,
    ensuring consistent formatting regardless of source data structure.
    
    Args:
        profile_data: Raw profile data dictionary from scraper
        
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
    if profile_data.get("summary"):
        summary_parts.append(f"Summary: {profile_data['summary'][:500]}...")  # Prevent prompt overflow
    if profile_data.get("experience"):
        summary_parts.append("Recent Experience:")
        for exp in profile_data["experience"][:4]:  # Focus on recent experience
            if isinstance(exp, dict):
                title = exp.get("title", "")
                company = exp.get("company", "")
                duration = exp.get("duration", "")
                if title and company:
                    summary_parts.append(f"  - {title} at {company} {f'({duration})' if duration else ''}")
    if profile_data.get("skills"):
        skills_list = profile_data["skills"]
        if isinstance(skills_list, list) and skills_list:
            # Handle both dict and string skill formats
            if isinstance(skills_list[0], dict):
                skills_names = [skill.get("name", "") for skill in skills_list[:15]]
            else:
                skills_names = skills_list[:15]
            summary_parts.append(f"Skills: {', '.join(str(s) for s in skills_names if s)}")
    
    if profile_data.get("education"):
        education_list = profile_data["education"]
        if isinstance(education_list, list) and education_list:
            summary_parts.append("Education:")
            for edu in education_list[:2]:  # Most recent education entries
                if isinstance(edu, dict):
                    school = edu.get("school", "")
                    degree = edu.get("degree", "")
                    if school:
                        summary_parts.append(f"  - {degree} from {school}" if degree else f"  - {school}")
    
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
    
    # Enhanced validation for job fit responses to ensure match percentages
    if "match" in response.lower() or "role" in response.lower():
        match_scores = re.findall(r"\b\d{1,3}%\b", response)
        if match_scores:
            # Add a summary of parsed match scores for easy reference
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
        
        # Intent-based routing using keyword analysis
        if any(phrase in user_query for phrase in ['job', 'roles', 'position', 'career', 'suited', 'fit', 'work as', 'good for']):
            state["next_node"] = "job_fit"
        elif any(phrase in user_query for phrase in ['improve', 'enhance', 'better', 'rewrite', 'content', 'headline', 'summary', 'description']):
            state["next_node"] = "content_enhancement"
        elif any(phrase in user_query for phrase in ['skill', 'learn', 'gap', 'missing', 'development', 'course', 'training']):
            state["next_node"] = "skill_gap"
        elif any(phrase in user_query for phrase in ['analyze', 'review', 'feedback', 'thoughts', 'look', 'profile']):
            state["next_node"] = "profile_analysis"
        else:
            # Default fallback for ambiguous queries
            state["next_node"] = "profile_analysis"
            
        return state
    except Exception as e:
        print(f"Routing error: {e}")
        state["next_node"] = "profile_analysis"
        return state
