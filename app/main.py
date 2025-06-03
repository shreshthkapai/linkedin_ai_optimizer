import streamlit as st

# ✅ Set page config FIRST before any other Streamlit command
st.set_page_config(page_title="LearnTube", page_icon="💼", layout="wide")

import uuid
import os
from chat_handler import ChatHandler

# Set environment variables from Streamlit secrets (for cloud) or .env (for local)
try:
    if hasattr(st, 'secrets') and st.secrets:
        os.environ['GEMINI_API_KEYY'] = st.secrets.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
        os.environ['APIFY_API_TOKEN'] = st.secrets.get('APIFY_API_TOKEN', os.getenv('APIFY_API_TOKEN', ''))
        os.environ['LI_AT_COOKIE'] = st.secrets.get('LI_AT_COOKIE', os.getenv('LI_AT_COOKIE', ''))
except Exception:
    # Fallback to environment variables (local development)
    from dotenv import load_dotenv
    load_dotenv()

# Initialize session state variables for persistent UI state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_handler" not in st.session_state:
    st.session_state.chat_handler = ChatHandler()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile_url" not in st.session_state:
    st.session_state.profile_url = ""

def main():
    """
    Main Streamlit application for LinkedIn profile analysis.
    
    Provides a chat interface where users can input their LinkedIn profile
    and ask questions about career optimization, job fit, and profile improvement.
    Uses a multi-agent system to provide specialized advice based on query intent.
    """
    st.title("🚀 LearnTube - LinkedIn Profile Optimizer")
    st.markdown("*by CareerNinja*")
    
    # LinkedIn URL input section with load button
    col1, col2 = st.columns([3, 1])
    with col1:
        profile_url = st.text_input(
            "LinkedIn Profile URL:", 
            value=st.session_state.profile_url,
            placeholder="https://www.linkedin.com/in/your-profile"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Visual alignment
        if st.button("Load Profile", type="primary"):
            if profile_url:
                st.session_state.profile_url = profile_url
                st.session_state.messages = []  # Reset chat history for new profile
                st.success("Profile loaded! Start chatting below.")
                st.rerun()
    
    # Main chat interface - only show if profile is loaded
    if st.session_state.profile_url:
        st.markdown("---")
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input handling
        if prompt := st.chat_input("Ask about your profile, job fit, career guidance..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your profile..."):
                    response = st.session_state.chat_handler.handle_chat(
                        profile_url=st.session_state.profile_url,
                        user_query=prompt,
                        session_id=st.session_state.session_id
                    )
                    
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "⚠️ Unable to process request. Please try again or check your LinkedIn URL."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        # Welcome screen with usage instructions
        st.info("👆 Enter your LinkedIn profile URL above to start chatting with the AI assistant.")
        
        st.markdown("### 💡 Example Questions:")
        st.markdown("""
        - "Analyze my LinkedIn profile and suggest improvements"
        - "How well does my profile match a Software Engineer role?"
        - "Rewrite my About section for better impact"
        - "What skills am I missing for a Data Scientist position?"
        - "Give me career guidance for transitioning to Product Management"
        """)

if __name__ == "__main__":
    main()
