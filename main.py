from langchain.graphs import Neo4jGraph
import dotenv
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
import json
import streamlit as st

graph = Neo4jGraph(
    max_connection_pool_size=50,
    connection_acquisition_timeout=30  # seconds
)

graph = Neo4jGraph(
    encrypted=True,
    trust="TRUST_ALL_CERTIFICATES"  # For cloud deployments
)

# Load environment variables - try both methods to support local and cloud deployment
dotenv.load_dotenv()

# Try to get credentials from Streamlit Secrets (for cloud deployment)
# Fall back to environment variables (for local development)
try:
    # Try to access Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] 
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"] 
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_DATABASE = st.secrets["NEO4J_DATABASE"]
    print("Using Streamlit secrets for configuration")
except (KeyError, AttributeError):
    # Fall back to environment variables
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4")
    print("Using environment variables for configuration")

# Print connection info for debugging (remove in production)
print(f"Connecting to database: {NEO4J_URI} with database: {NEO4J_DATABASE}")

class VerboseHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("name", "LLM")
        prompt_summary = prompts[0][:100] + "..." if prompts and len(prompts[0]) > 100 else prompts
        self.logs.append({
            "type": "llm_start", 
            "message": f"Starting {model_name} with prompt: {prompt_summary}"
        })
    
    def on_llm_end(self, response, **kwargs):
        self.logs.append({
            "type": "llm_end", 
            "message": "Model generation completed successfully"
        })
    
    def on_llm_new_token(self, token, **kwargs):
        # We don't log every token to avoid cluttering the output
        pass
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        chain_type = serialized.get("name", "Chain")
        # Clean inputs for display - only keep the question
        clean_inputs = {}
        if "query" in inputs:
            clean_inputs["query"] = inputs["query"]
        elif "question" in inputs:
            clean_inputs["question"] = inputs["question"]
        
        self.logs.append({
            "type": "chain_start", 
            "message": f"Starting {chain_type} processing with query: {json.dumps(clean_inputs, default=str)}"
        })
    
    def on_chain_end(self, outputs, **kwargs):
        # Clean outputs for display
        clean_outputs = {}
        if "result" in outputs:
            result_preview = str(outputs["result"])[:100] + "..." if len(str(outputs["result"])) > 100 else outputs["result"]
            clean_outputs["result"] = result_preview
        
        self.logs.append({
            "type": "chain_end", 
            "message": f"Chain processing completed with result preview: {json.dumps(clean_outputs, default=str)}"
        })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "database_tool")
        input_preview = input_str[:100] + "..." if len(input_str) > 100 else input_str
        self.logs.append({
            "type": "tool_start", 
            "message": f"Executing Neo4j database query: {input_preview}"
        })
    
    def on_tool_end(self, output, **kwargs):
        output_preview = str(output)[:100] + "..." if len(str(output)) > 100 else output
        self.logs.append({
            "type": "tool_end", 
            "message": f"Database query completed with results: {output_preview}"
        })


def generate(user_input):
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

        # Create our callback handler
        handler = VerboseHandler()
        callback_manager = CallbackManager([handler])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            api_key=GOOGLE_API_KEY,
            callbacks=[handler]
        )
        
        chain = GraphCypherQAChain.from_llm(
            graph=graph, 
            llm=llm, 
            allow_dangerous_requests=True, 
            verbose=True,
            callbacks=[handler]
        )

        
        response = chain(user_input)
        result = response.get('result') or response.get('answer') or response

    finally:
        graph.driver.close()  # Critical cleanup

        # Return both the result and the logs
        return {
            "result": result,
            "logs": handler.logs
        }

# Helper function to get the verbose logs for st.status
def get_verbose_logs():
    handler = VerboseHandler()
    return handler


