# chat_handler.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    AgentState,
    scrape_agent,
    profile_analysis_agent,
    job_fit_agent,
    content_enhancement_agent,
    skill_gap_agent,
    route_agent,
)


class ChatHandler:
    def __init__(self):
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.session_data = {}  # Store session-specific data

    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("scrape", scrape_agent)
        workflow.add_node("route", route_agent)
        workflow.add_node("profile_analysis", profile_analysis_agent)
        workflow.add_node("job_fit", job_fit_agent)
        workflow.add_node("content_enhancement", content_enhancement_agent)
        workflow.add_node("skill_gap", skill_gap_agent)

        workflow.set_entry_point("scrape")
        workflow.add_edge("scrape", "route")
        
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

        workflow.add_edge("profile_analysis", END)
        workflow.add_edge("job_fit", END)
        workflow.add_edge("content_enhancement", END)
        workflow.add_edge("skill_gap", END)

        return workflow.compile(checkpointer=self.memory)

    def handle_chat(self, profile_url: str, user_query: str, session_id: str):
        try:
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    "profile_data": None,
                    "chat_history": []
                }

            session_info = self.session_data[session_id]

            state = {
                "profile_url": profile_url,
                "profile_data": session_info.get("profile_data"),
                "user_query": user_query,
                "job_role": self._extract_job_role(user_query),
                "analysis_result": None,
                "session_id": session_id,
                "chat_history": session_info.get("chat_history", []),
                "next_node": None
            }

            config = {"configurable": {"thread_id": session_id}}

            print(f"Processing query: {user_query[:50]}...")
            result = self.workflow.invoke(state, config=config)

            if isinstance(result, dict):
                if result.get("profile_data"):
                    self.session_data[session_id]["profile_data"] = result["profile_data"]
                if result.get("chat_history"):
                    self.session_data[session_id]["chat_history"] = result["chat_history"]

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
        Clean and format the agent's response for user output.
        Strips whitespace, removes empty lines, and handles short or error responses.

        Args:
            response (str): The raw response from the agent workflow.

        Returns:
            str: A user-friendly, cleaned response.
        """
        if not isinstance(response, str):
            response = str(response)

        lines = [line.strip() for line in response.split('\n') if line.strip()]

        cleaned_lines = []
        for line in lines:
            # Only keep non-trivial lines
            if len(line) > 3 and not line.replace('.', '').replace('*', '').strip() == '':
                cleaned_lines.append(line)

        result = '\n\n'.join(cleaned_lines) if cleaned_lines else response

        if len(result.strip()) < 50 or "error" in result.lower():
            return (
                "I couldn't generate a detailed response. Please try rephrasing your question "
                "or ensure your LinkedIn profile has enough information to analyze."
            )

        return result.strip()

def clear_session(self, session_id: str):
    if session_id in self.session_data:
        del self.session_data[session_id]
