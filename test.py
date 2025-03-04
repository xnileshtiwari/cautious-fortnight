import streamlit as st
import dotenv
import os
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph

# Load environment variables from a .env file
dotenv.load_dotenv()

# Retrieve configuration variables from the environment
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"

def generate(user_input):
    # Create the Neo4jGraph instance without passing the unsupported parameters.
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD, 
        database=NEO4J_DATABASE,
        refresh_schema=True  # Uses the default behavior; you can also pass a schema if available.
    )
    
    # Initialize the Google Generative AI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        api_key=GOOGLE_API_KEY,
    )
    
    # Create the chain to convert natural-language questions into Cypher queries
    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        allow_dangerous_requests=True,
        verbose=True
    )
    
    response = chain(user_input)
    return response.get("result") or response.get("answer") or response

def main():
    st.title("Neo4j Graph Query with Google Generative AI")
    st.write("Enter your query to interact with your Neo4j graph database.")

    user_input = st.text_area("Enter your query:")

    if st.button("Submit") and user_input:
        st.write("Processing your query...")
        try:
            result = generate(user_input)
            st.markdown("### Query Result:")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
