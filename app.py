import streamlit as st
import weaviate
import weaviate.exceptions
import ollama
import pypdf  # Using pypdf instead of deprecated PyPDF2
import os
import tempfile
import json
import re
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="HR Policy Assistant", layout="wide", page_icon="👔")

# Initialize session state for chat history and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversational memories with a window of 5 exchanges for each agent
# Using the new API to avoid deprecation warnings
if "general_memory" not in st.session_state:
    st.session_state.general_memory = ConversationBufferWindowMemory(return_messages=True)
    # Set window size to 5
    st.session_state.general_memory.k = 5
    
if "hr_memory" not in st.session_state:
    st.session_state.hr_memory = ConversationBufferWindowMemory(return_messages=True)
    # Set window size to 5
    st.session_state.hr_memory.k = 5
    
# Track which agent handled each message
if "agent_mapping" not in st.session_state:
    st.session_state.agent_mapping = {}

# Connect to Weaviate instance
@st.cache_resource
def get_weaviate_client():
    weaviate_host = os.environ.get("WEAVIATE_HOST", "localhost")
    weaviate_grpc_host = os.environ.get("WEAVIATE_GRPC_HOST", weaviate_host)
    return weaviate.connect_to_custom(
        http_host=weaviate_host,
        http_port=8080,
        http_secure=False,  # Set to True if using HTTPS
        grpc_host=weaviate_grpc_host,
        grpc_port=50051,
        grpc_secure=False,  # Set to True if using secure gRPC
        additional_config=weaviate.AdditionalConfig(
            trust_env=True  # Required for custom SSL certificates
        )
    )
# Create a data collection if it doesn't exist
def initialize_collection(client):
    collection_name = "hr_policies"
    try:
        # Try to get the collection first
        collection = client.collections.get(collection_name)
    except weaviate.exceptions.WeaviateCollectionDoesNotExistException:
        # If collection doesn't exist, create it
        collection = client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="policy_category", data_type=DataType.TEXT),
                Property(name="last_updated", data_type=DataType.DATE),
            ],
        )
    return collection

# Extract text from PDF
def extract_text_from_pdf(pdf_file, file_name, policy_category, last_updated):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text_chunks = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            text_chunks.append({
                "text": text,
                "source": file_name,
                "page": page_num + 1,
                "policy_category": policy_category,
                "last_updated": last_updated
            })
    
    return text_chunks

# Function to embed text and store in Weaviate
def embed_and_store(collection, text_chunks):
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk in text_chunks:
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk["text"])
            batch.add_object(
                properties={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "policy_category": chunk.get("policy_category", "General"),
                    "last_updated": chunk.get("last_updated", "")
                },
                vector=response["embedding"],
            )
    
    return len(text_chunks)

# Function to perform RAG query
def query_documents(collection, query, category=None, limit=5):
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)
    
    filters = None
    if category and category != "All Categories":
        filters = wvc.query.Filter.by_property("policy_category").equal(category)
    
    results = collection.query.near_vector(
        near_vector=query_embedding["embedding"], 
        filters=filters,
        limit=limit,
        return_properties=["text", "source", "page", "policy_category", "last_updated"]
    )
    
    contexts = []
    for obj in results.objects:
        contexts.append({
            "text": obj.properties["text"],
            "source": obj.properties["source"],
            "page": obj.properties["page"],
            "policy_category": obj.properties.get("policy_category", "General"),
            "last_updated": obj.properties.get("last_updated", "")
        })
    
    return contexts

# Function to generate response with RAG and memory
def generate_rag_response(query, contexts):
    # Get chat history from HR memory
    memory_variables = st.session_state.hr_memory.load_memory_variables({})
    chat_history = memory_variables.get("history", [])
    
    # Format chat history for context
    chat_history_text = ""
    if chat_history:
        chat_history_text = "\n".join([
            f"Human: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}"
            for message in chat_history
        ])
        chat_history_text = f"\nChat History:\n{chat_history_text}\n"
    
    # Format retrieved documents for context
    context_text = "\n\n".join([
        f"Source: {ctx['source']}, Category: {ctx['policy_category']}, Page: {ctx['page']}, Last Updated: {ctx['last_updated']}\n{ctx['text']}" 
        for ctx in contexts
    ])
    
    # Create augmented prompt with both document context and chat history
    augmented_prompt = f"""You are an HR Policy Assistant. Using the following company policy documents and chat history, answer the HR professional's question about company policies. 
Be concise, accurate, and helpful. If you don't know the answer based on the provided context, say you don't have enough information.

Context from HR policy documents:
{context_text}
{chat_history_text}
HR professional's question: {query}"""

    # Configure Ollama endpoint
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    os.environ["OLLAMA_HOST"] = f"http://{ollama_host}:11434"
    
    # Generate response
    response = ollama.generate(
        model="llama3",
        prompt=augmented_prompt,
        stream=False,
    )
    
    # Update HR memory with the new exchange
    st.session_state.hr_memory.save_context(
        {"input": query},
        {"output": response["response"]}
    )
    
    return response["response"], contexts

# HR Policy Tool for the RAG Agent
@tool
def query_hr_policies(query: str) -> str:
    """
    Use this tool to search for information in HR policy documents.
    This tool should be used for questions about company policies, 
    procedures, benefits, or other HR-related information.
    """
    client = get_weaviate_client()
    collection = initialize_collection(client)
    
    # Get selected category from sidebar if available
    category = st.session_state.get("selected_category", "All Categories")
    
    # Search for relevant policy documents
    contexts = query_documents(collection, query, category=category)
    
    if not contexts:
        return "I couldn't find any relevant policy documents. Please upload HR policy documents first or try a different query."
    
    # Generate response using RAG
    response, sources = generate_rag_response(query, contexts)
    
    # Store sources in session state for display later
    st.session_state.last_sources = sources
    
    return response

# General Conversation Tool for basic chats
@tool
def general_conversation(query: str) -> str:
    """
    Use this tool for general conversation, greetings, 
    small talk, or non-HR policy questions.
    """
    # Get chat history from general memory
    memory_variables = st.session_state.general_memory.load_memory_variables({})
    chat_history = memory_variables.get("history", [])
    
    # Format chat history
    chat_history_text = ""
    if chat_history:
        chat_history_text = "\n".join([
            f"Human: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}"
            for message in chat_history
        ])
        chat_history_text = f"\nChat History:\n{chat_history_text}\n"
    
    # Create prompt for general conversation
    prompt = f"""You are a helpful and friendly AI assistant for an HR department. 
Respond to the user's message in a professional but conversational tone.
This is for general conversation only, not for HR policy questions.

{chat_history_text}
User's message: {query}"""

    # Configure Ollama endpoint
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    os.environ["OLLAMA_HOST"] = f"http://{ollama_host}:11434"
    
    # Generate response
    response = ollama.generate(
        model="llama3",
        prompt=prompt,
        stream=False,
    )
    
    # Update general memory
    st.session_state.general_memory.save_context(
        {"input": query},
        {"output": response["response"]}
    )
    
    # Mark that no sources are available for this response
    st.session_state.last_sources = None
    
    return response["response"]

# Create tools list for the agent
tools = [
    query_hr_policies,
    general_conversation
]

# Using a simpler approach for our agent
# We don't need the complex React agent setup, so we're removing it

# Define tool descriptions for the prompt
tool_descriptions = """
- query_hr_policies: Use this tool to search for information in HR policy documents. This tool should be used for questions about company policies, procedures, benefits, or other HR-related information.
- general_conversation: Use this tool for general conversation, greetings, small talk, or non-HR policy questions.
"""

# Create a simple template for the agent
template = """You are an intelligent HR assistant who can:
1. Answer questions about company policies using the query_hr_policies tool
2. Engage in general conversation using the general_conversation tool

{tool_descriptions}

Based on the user's query, decide which tool to use:
1. If the question is about HR policies, procedures, benefits, etc., use query_hr_policies
2. For greetings, general questions, or small talk, use general_conversation

User query: {query}

Think about which tool is most appropriate:"""

# Create a simple decision function instead of using ReAct
def determine_tool(query):
    """Determine which tool to use based on the query"""
    prompt = template.format(
        tool_descriptions=tool_descriptions,
        query=query
    )
    
    # Ask the LLM to decide which tool to use
    response = ollama.generate(
        model="llama3",
        prompt=prompt,
        stream=False
    )
    print(response["response"])
    # Parse the response to determine which tool to use
    tool_choice = response["response"].lower()
    
    if "query_hr_policies" in tool_choice or "policies" in tool_choice or "hr" in tool_choice:
        return query_hr_policies
    else:
        return general_conversation

# Define policy categories
POLICY_CATEGORIES = [
    "All Categories",
    "Recruitment",
    "Onboarding",
    "Compensation & Benefits",
    "Performance Management",
    "Learning & Development",
    "Employee Relations",
    "Health & Safety",
    "Termination",
    "Code of Conduct",
    "Diversity & Inclusion",
    "Remote Work",
    "Leave Policies",
    "Other"
]

# Main application
def main():
    client = get_weaviate_client()
    collection = initialize_collection(client)
    
    # Set default category to "All Categories"
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All Categories"
      # Main Policy Assistant chat interface with improved styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>HR Policy Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; margin-bottom: 20px;'>An AI tool to help HR professionals navigate company policies</p>", unsafe_allow_html=True)
    
    # Add horizontal line for visual separation
    st.markdown("<hr style='margin: 10px 0px 20px 0px;'>", unsafe_allow_html=True)
      # Document manager link with better styling
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'><a href='http://localhost:8502/' target='_blank' style='text-decoration: none; color: #4169E1;'>📁 Go to Document Manager - Upload and manage policy documents</a></div>", unsafe_allow_html=True)
    
    # Action buttons in a cleaner layout
    st.markdown("<p style='font-weight: bold; margin-bottom: 10px;'>Actions:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🧹 Clear Conversation Memory", use_container_width=True):
            st.session_state.hr_memory.clear()
            st.session_state.general_memory.clear()
            st.info("Conversation memory cleared. The assistant will no longer remember previous interactions.", icon="✅")
      # Quick access to common HR questions
    with st.expander("Common HR Policy Questions", expanded=True):
        common_questions = [
            "What is our parental leave policy?",
            "How is performance evaluation conducted?",
            "What are our remote work guidelines?",
            "What is the procedure for handling employee grievances?",
            "What are our diversity and inclusion initiatives?"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(common_questions):
            with cols[i % 3]:
                if st.button(question, key=f"q_{i}", use_container_width=True):
                    st.session_state.current_question = question
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If there are sources, display them
            if "sources" in message:
                with st.expander("View policy sources"):
                    for source in message["sources"]:
                        st.write(f"**Policy Document:** {source['source']}")
                        st.write(f"**Category:** {source.get('policy_category', 'General')}")
                        st.write(f"**Page:** {source['page']}")
                        st.write(f"**Last Updated:** {source.get('last_updated', '')}")
                        st.markdown("---")
                        st.text(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
      # Input for new query with enhanced styling
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    if "current_question" in st.session_state:
        prompt = st.session_state.current_question
        del st.session_state.current_question
    else:
        prompt = st.chat_input("Ask any question about company policies or general inquiries...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                # Configure Ollama endpoint
                ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
                os.environ["OLLAMA_HOST"] = f"http://{ollama_host}:11434"
                
                # Use our decision function to choose the appropriate tool
                try:
                    # Determine which tool to use
                    selected_tool = determine_tool(prompt)
                    # Execute the selected tool
                    response = selected_tool(prompt)
                    
                    # Display the response
                    st.markdown(response)
                    
                    # If the HR policies tool was used and sources are available, show them
                    if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
                        with st.expander("View policy sources"):
                            for source in st.session_state.last_sources:
                                st.write(f"**Policy Document:** {source['source']}")
                                st.write(f"**Category:** {source.get('policy_category', 'General')}")
                                st.write(f"**Page:** {source['page']}")
                                st.write(f"**Last Updated:** {source.get('last_updated', '')}")
                                st.markdown("---")
                                st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                        
                        # Add response to chat history with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": st.session_state.last_sources
                        })
                    else:
                        # Add response to chat history without sources (general conversation)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.markdown("I encountered an error while processing your request. Please try again.")

if __name__ == "__main__":
    main()
