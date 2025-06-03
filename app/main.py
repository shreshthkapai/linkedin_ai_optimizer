import streamlit as st

# ✅ Set page config FIRST before any other Streamlit command
st.set_page_config(page_title="LearnTube", page_icon="💼", layout="wide")

import uuid
import os

# Enhanced environment variable loading for Streamlit Cloud
def load_environment_variables():
    """
    Load environment variables from Streamlit secrets (cloud) or .env file (local).
    Returns True if all required variables are loaded successfully.
    """
    required_vars = ['GEMINI_API_KEY', 'APIFY_API_TOKEN']
    loaded_vars = {}
    
    # Try to load from Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and len(st.secrets) > 0:
            st.write("🔧 Loading from Streamlit secrets...")  # Debug info
            for var in required_vars:
                if var in st.secrets:
                    os.environ[var] = st.secrets[var]
                    loaded_vars[var] = "✅ Loaded from secrets"
                else:
                    loaded_vars[var] = "❌ Missing from secrets"
            
            # Also load optional variables
            optional_vars = ['LI_AT_COOKIE', 'HUGGING_FACE_API_KEY']
            for var in optional_vars:
                if var in st.secrets:
                    os.environ[var] = st.secrets[var]
                    
        else:
            st.write("📁 Secrets not found, trying .env file...")  # Debug info
            # Fallback to .env file for local development
            from dotenv import load_dotenv
            load_dotenv()
            
            for var in required_vars:
                if os.getenv(var):
                    loaded_vars[var] = "✅ Loaded from .env"
                else:
                    loaded_vars[var] = "❌ Missing from .env"
                    
    except Exception as e:
        st.error(f"Error loading environment variables: {str(e)}")
        return False, loaded_vars
    
    # Check if all required variables are set
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    success = len(missing_vars) == 0
    
    return success, loaded_vars

# Load environment variables and check status
env_success, env_status = load_environment_variables()

# Show environment status in sidebar for debugging
with st.sidebar:
    st.subheader("🔧 Environment Status")
    for var, status in env_status.items():
        st.write(f"{var}: {status}")
    
    if not env_success:
        st.error("⚠️ Some required environment variables are missing!")
        st.write("Required variables:")
        st.write("- GEMINI_API_KEY")
        st.write("- APIFY_API_TOKEN")

# Only import chat_handler after environment variables are set
if env_success:
    try:
        from chat_handler import ChatHandler
        chat_handler_available = True
    except ImportError as e:
        st.error(f"Failed to import ChatHandler: {str(e)}")
        chat_handler_available = False
else:
    chat_handler_available = False

# Initialize session state variables for persistent UI state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_handler" not in st.session_state and chat_handler_available:
    try:
        st.session_state.chat_handler = ChatHandler()
    except Exception as e:
        st.error(f"Failed to initialize ChatHandler: {str(e)}")
        chat_handler_available = False

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
    
    # Show warning if environment is not properly configured
    if not env_success or not chat_handler_available:
        st.error("⚠️ Application is not properly configured. Please check the sidebar for details.")
        return
    
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
                    try:
                        # Add debugging information
                        st.write("🔍 Debug Info:")
                        st.write(f"- Profile URL: {st.session_state.profile_url[:50]}...")
                        st.write(f"- Session ID: {st.session_state.session_id}")
                        st.write(f"- API Keys available: {bool(os.getenv('GEMINI_API_KEY')) and bool(os.getenv('APIFY_API_TOKEN'))}")
                        
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
                            
                    except Exception as e:
                        error_msg = f"❌ Error processing request: {str(e)}"
                        st.error(error_msg)
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
