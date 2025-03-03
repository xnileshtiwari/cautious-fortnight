import streamlit as st
import time
import random
from main import generate

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Hide Streamlit default elements
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    .viewerBadge_link__1S137 {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    .stAttribution {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Social Media Analytics Assistant📈")


# Custom styling
st.markdown("""
<style>
    .user-avatar {
        background-color: #4b8bf4;
        color: white;
        padding: 0.5rem;
        border-radius: 50%;
        margin-right: 0.5rem;
        font-weight: bold;
    }
    .assistant-avatar {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 50%;
        margin-right: 0.5rem;
        font-weight: bold;
    }
    .chat-message {
        margin-bottom: 1rem;
    }
    /* Custom styling for sample prompt buttons */
    div[data-testid="stButton"] > button {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        color: #495057;
        font-size: 0.9rem;
        padding: 0.5rem;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #e9ecef;
        border-color: #dee2e6;
    }
    .sample-prompts-container {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)






# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Predefined prompts
predefined_prompts = [
    "🔍 Which is the most liked post?",
    "💡 Which post has the most comments?",
    "🔥 Which post has the most shares?",
    "📜 Where the post 7176541902119280640 is uploaded?"
]

# Create a container for sample prompts with consistent styling
st.markdown('<div class="sample-prompts-container"></div>', unsafe_allow_html=True)




st.subheader("Sample Prompts", divider="gray")

# Create a 2x2 grid for the sample prompts
col1, col2,  = st.columns(2)
with col1:
    if st.button(predefined_prompts[0]):
        st.session_state.temp_input = predefined_prompts[0]
    
    if st.button(predefined_prompts[2]):
        st.session_state.temp_input = predefined_prompts[2]

with col2:
    if st.button(predefined_prompts[1]):
        st.session_state.temp_input = predefined_prompts[1]
    
    if st.button(predefined_prompts[3]):
        st.session_state.temp_input = predefined_prompts[3]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("✨ Ask me anything...")

# Use predefined prompt if selected, otherwise use user input
if 'temp_input' in st.session_state and st.session_state.temp_input:
    user_input = st.session_state.temp_input
    st.session_state.temp_input = None  # Clear the temporary input

# main.py

try:
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        assistant_message_placeholder = st.empty()  # Use st.empty() as a placeholder
        full_response = ""

            # Add status element to display verbose output
        with st.status("Processing with LangChain...", expanded=True) as status:
            status.update(label="Running LangChain pipeline with real-time logs", state="running")
            
            # Get response from main.py
            response_data = generate(user_input)
            
            # Extract the result and logs
            result = response_data["result"]
            verbose_logs = response_data["logs"]
            
            # Display the verbose logs in the status component
            for log in verbose_logs:
                log_type = log["type"]
                message = log["message"]
                
                # Format the log message based on its type for better readability
                if log_type == "llm_start":
                    status.write(f"🧠 **LLM Started**: {message}")
                elif log_type == "llm_end":
                    status.write(f"🧠 **LLM Completed**: {message}")
                elif log_type == "llm_token":
                    # Don't display individual tokens to avoid clutter
                    pass
                elif log_type == "chain_start":
                    status.write(f"⛓️ **Chain Started**: {message}")
                    # Add a divider for better visual separation
                    status.write("---")
                elif log_type == "chain_end":
                    # Add a divider for better visual separation
                    status.write("---")
                    status.write(f"⛓️ **Chain Completed**: {message}")
                elif log_type == "tool_start":
                    status.write(f"🔧 **Database Query**: {message}")
                elif log_type == "tool_end":
                    status.write(f"🔧 **Query Results**: {message}")
                    # Add a divider after database operations
                    status.write("---")
                else:
                    status.write(f"ℹ️ {message}")
            
            # Set the full response
            full_response = result
            
            # Update the assistant message placeholder
            assistant_message_placeholder.markdown(full_response)
            
            # Complete the status
            status.update(label="✅ Processing complete! See response below", state="complete")

        assistant_message_placeholder.empty() # Clear the placeholder
        with st.chat_message("assistant"):  # Now create the final chat message
            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        st.rerun()  # Refresh UI after response

except Exception as e:
    st.error(f"Response was blocked due to safety concerns. Please try different question.")

# Rerun the app to update the chat immediately
if user_input:
    st.rerun()
