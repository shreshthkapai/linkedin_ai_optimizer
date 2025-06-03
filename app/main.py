import streamlit as st

# ✅ Set page config FIRST before any other Streamlit command
st.set_page_config(page_title="LearnTube", page_icon="💼", layout="wide")

import uuid
import os

# ✅ CRITICAL FIX: Set environment variables BEFORE importing any modules that need them
try:
    # For Streamlit Cloud deployment
    if hasattr(st, 'secrets') and st.secrets:
        for key in ['GEMINI_API_KEY', 'APIFY_API_TOKEN', 'LI_AT_COOKIE', 'HUGGING_FACE_API_KEY']:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
                print(f"✅ Set {key} from Streamlit secrets")
    
    # For local development - only try to load .env if secrets aren't available
    elif not os.getenv('GEMINI_API_KEY'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded environment variables from .env file")
        except ImportError:
            print("⚠️ python-dotenv not available, using system environment variables")
    
    # Validate critical environment variables
    required_vars = ['GEMINI_API_KEY', 'APIFY_API_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please configure these in your Streamlit secrets or .env file")
        st.stop()
    else:
        print("✅ All required environment variables are set")

except Exception as e:
    st.error(f"❌ Error setting up environment: {str(e)}")
    st.stop()

# ✅ Now import modules that depend on environment variables
try:
    from chat_handler import ChatHandler
    print("✅ Successfully imported ChatHandler")
except Exception as e:
    st.error(f"❌ Error importing ChatHandler: {str(e)}")
    st.info("This might be due to missing dependencies or environment variables")
    st.stop()

# Initialize session state variables for persistent UI state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_handler" not in st.session_state:
    try:
        st.session_state.chat_handler = ChatHandler()
        print("✅ ChatHandler initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing ChatHandler: {str(e)}")
        st.stop()
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
    
    # Debug section (only visible in development)
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("**Environment Status:**")
        st.sidebar.write(f"GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
        st.sidebar.write(f"APIFY_API_TOKEN: {'✅ Set' if os.getenv('APIFY_API_TOKEN') else '❌ Missing'}")
        st.sidebar.write(f"LI_AT_COOKIE: {'✅ Set' if os.getenv('LI_AT_COOKIE') else '❌ Missing'}")
        st.sidebar.write(f"Session ID: {st.session_state.session_id[:8]}...")
    
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
                # Validate URL format
                if not profile_url.startswith("https://www.linkedin.com/in/"):
                    st.error("❌ Please enter a valid LinkedIn profile URL")
                    return
                
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
                        print(f"Chat error: {e}")  # Log for debugging
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
