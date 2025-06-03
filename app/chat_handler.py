from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    AgentState,
    UnifiedContext,
    create_unified_context,
    scrape_agent,
    profile_analysis_agent,
    job_fit_agent,
    content_enhancement_agent,
    skill_gap_agent,
    route_agent,
)


class ChatHandler:
    """
    Main orchestrator for the LinkedIn profile analysis chat system.
    
    Manages the agent workflow graph, session state, and handles
    user interactions with the multi-agent system. Maintains conversation
    context and profile data across multiple queries within a session.
    """
    
    def __init__(self):
        """Initialize the chat handler with workflow and memory management."""
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.unified_contexts = {}  # Session-based unified contexts

    def _build_workflow(self):
        """
        Construct the LangGraph workflow that orchestrates all agents.
        
        Creates a directed graph where queries flow from scraping through
        routing to specialized agents, with conditional edges based on
        user intent classification.
        
        Returns:
            Compiled workflow graph with checkpointing enabled
        """
        workflow = StateGraph(AgentState)

        # Add all agent nodes to the workflow
        workflow.add_node("scrape", scrape_agent)
        workflow.add_node("route", route_agent)
        workflow.add_node("profile_analysis", profile_analysis_agent)
        workflow.add_node("job_fit", job_fit_agent)
        workflow.add_node("content_enhancement", content_enhancement_agent)
        workflow.add_node("skill_gap", skill_gap_agent)

        # Define workflow flow: always start with scraping, then route to appropriate agent
        workflow.set_entry_point("scrape")
        workflow.add_edge("scrape", "route")
        
        # Conditional routing based on query analysis
        workflow.add_conditional_edges(
            "route",
            lambda state: state.get("next_node", "profile_analysis"),
            {
                "profile_analysis": "profile_analysis",
                "job_fit": "job_fit",
                "content_enhancement": "content_enhancement",
                "skill_gap": "skill_gap"
            }
        )

        # All specialized agents end the workflow
        workflow.add_edge("profile_analysis", END)
        workflow.add_edge("job_fit", END)
        workflow.add_edge("content_enhancement", END)
        workflow.add_edge("skill_gap", END)

        return workflow.compile(checkpointer=self.memory)

    def handle_chat(self, profile_url: str, user_query: str, session_id: str):
        """
        Process a user query through the agent workflow system.
        
        Manages session state, invokes the appropriate workflow path,
        and returns a cleaned response to the user. Handles error cases
        gracefully with informative error messages.
        
        Args:
            profile_url: LinkedIn profile URL to analyze
            user_query: User's natural language question
            session_id: Unique session identifier for state management
            
        Returns:
            Processed response string from the appropriate agent
        """
        try:
            # Initialize unified context if new session
            if session_id not in self.unified_contexts:
                self.unified_contexts[session_id] = create_unified_context()

            unified_context = self.unified_contexts[session_id]

            # Prepare state for workflow execution
            state = {
                "profile_url": profile_url,
                "unified_context": unified_context,
                "user_query": user_query,
                "job_role": self._extract_job_role(user_query),  # Extract mentioned job roles
                "analysis_result": None,
                "session_id": session_id,
                "next_node": None
            }

            config = {"configurable": {"thread_id": session_id}}

            print(f"Processing query: {user_query[:50]}...")
            result = self.workflow.invoke(state, config=config)

            # Update unified context with workflow results
            if isinstance(result, dict) and result.get("unified_context"):
                self.unified_contexts[session_id] = result["unified_context"]

                response = result.get("analysis_result", "No response generated")
            else:
                response = "Error: Unexpected response format"

            print(f"Generated response length: {len(str(response))}")
            return self._clean_response(response)

        except Exception as e:
            print(f"Chat error: {e}")
            return (
                "I apologize, but I encountered an error processing your request. "
                f"Please try again or rephrase your question. Error details: {str(e)}"
            )

    def _extract_job_role(self, user_query: str) -> str:
        """
        Extract mentioned job roles from user queries using pattern matching.
        
        Helps contextualize responses by identifying specific roles the user
        is interested in, enabling more targeted career advice.
        
        Args:
            user_query: User's natural language input
            
        Returns:
            Extracted job role title or empty string if none found
        """
        query_lower = user_query.lower()
        job_patterns = [
            "data scientist", "software engineer", "product manager", "marketing manager",
            "business analyst", "project manager", "designer", "developer", "analyst",
            "consultant", "manager", "director", "engineer", "specialist", "coordinator"
        ]
        for pattern in job_patterns:
            if pattern in query_lower:
                return pattern.title()
        return ""
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format agent responses for optimal user experience.
        
        Removes empty lines, ensures minimum response quality, and provides
        fallback messages for inadequate responses. Maintains professional
        formatting while being user-friendly.
        
        Args:
            response: Raw response from the agent workflow
            
        Returns:
            Cleaned, user-ready response string
        """
        if not isinstance(response, str):
            response = str(response)

        # Remove empty lines and clean whitespace
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        cleaned_lines = []
        for line in lines:
            # Only keep substantive lines (filter out formatting artifacts)
            if len(line) > 3 and not line.replace('.', '').replace('*', '').strip() == '':
                cleaned_lines.append(line)

        result = '\n\n'.join(cleaned_lines) if cleaned_lines else response

        # Quality check: ensure response is substantial and helpful
        if len(result.strip()) < 50 or "error" in result.lower():
            return (
                "I couldn't generate a detailed response. Please try rephrasing your question "
                "or ensure your LinkedIn profile has enough information to analyze."
            )

        return result.strip()

    def clear_session(self, session_id: str):
        """
        Clean up session data to prevent memory leaks.
        
        Args:
            session_id: Session identifier to clear
        """
        if session_id in self.unified_contexts:
            del self.unified_contexts[session_id]
