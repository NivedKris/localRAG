import streamlit as st
import weaviate
import weaviate.exceptions
import ollama
import PyPDF2
import os
import tempfile
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

# Set page configuration
st.set_page_config(page_title="Local RAG System", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Connect to local Weaviate instance
@st.cache_resource
def get_weaviate_client():
    return weaviate.connect_to_local()

# Create a data collection if it doesn't exist
def initialize_collection(client):
    collection_name = "documents"
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
            ],
        )
    return collection

# Extract text from PDF
def extract_text_from_pdf(pdf_file, file_name):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text_chunks = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            text_chunks.append({
                "text": text,
                "source": file_name,
                "page": page_num + 1
            })
    
    return text_chunks

# Function to embed text and store in Weaviate
def embed_and_store(collection, text_chunks):
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk in text_chunks:
            response = ollama.embeddings(model="all-minilm", prompt=chunk["text"])
            batch.add_object(
                properties={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"]
                },
                vector=response["embedding"],
            )
    
    return len(text_chunks)

# Function to perform RAG query
def query_documents(collection, query, limit=3):
    query_embedding = ollama.embeddings(model="all-minilm", prompt=query)
    results = collection.query.near_vector(
        near_vector=query_embedding["embedding"], 
        limit=limit,
        return_properties=["text", "source", "page"]
    )
    
    contexts = []
    for obj in results.objects:
        contexts.append({
            "text": obj.properties["text"],
            "source": obj.properties["source"],
            "page": obj.properties["page"]
        })
    
    return contexts

# Function to generate response with RAG
def generate_rag_response(query, contexts):
    context_text = "\n\n".join([f"Source: {ctx['source']}, Page: {ctx['page']}\n{ctx['text']}" for ctx in contexts])
    augmented_prompt = f"""Using only the following context, answer the question. If you don't know the answer based on the context, say you don't have enough information.

Context:
{context_text}

Question: {query}"""

    response = ollama.generate(
        model="tinyllama",
        prompt=augmented_prompt,
        stream=False,
    )
    
    return response["response"], contexts

# Main application
def main():
    client = get_weaviate_client()
    collection = initialize_collection(client)
    
    # Sidebar for stats
    with st.sidebar:
        st.title("Local RAG System")
        st.write("A fully local RAG system using Weaviate, Ollama, and Streamlit")
        
        # Count documents in the collection
        try:
            doc_count = collection.query.fetch_objects().total_count
            st.info(f"Documents in database: {doc_count}")
        except:
            st.info(f"Documents in database: 0")
        
        st.write("---")
        st.write("Models used:")
        st.write("- Embeddings: all-minilm")
        st.write("- LLM: tinyllama")
    
    # Create tabs for Upload and Chat
    tab1, tab2 = st.tabs(["📁 Upload Documents", "💬 Chat"])
    
    # Upload tab
    with tab1:
        st.header("Upload PDF Documents")
        
        uploaded_files = st.file_uploader("Choose PDF files", 
                                          type="pdf", 
                                          accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = 0
                for i, pdf_file in enumerate(uploaded_files):
                    file_name = pdf_file.name  # Store the name separately
                    status_text.write(f"Processing: {file_name}")
                    # Save the uploaded file temporarily to process with PyPDF2
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    with open(tmp_path, 'rb') as f:
                        text_chunks = extract_text_from_pdf(f, file_name)
                        chunks_count = embed_and_store(collection, text_chunks)
                        total_chunks += chunks_count
                    
                    # Remove the temporary file
                    os.unlink(tmp_path)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                progress_bar.progress(1.0)
                st.success(f"Successfully processed {len(uploaded_files)} files with {total_chunks} text chunks.")
    
    # Chat tab
    with tab2:
        st.header("Chat with your Documents")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If there are sources, display them
                if "sources" in message:
                    with st.expander("View sources"):
                        for source in message["sources"]:
                            st.write(f"**Source:** {source['source']}, **Page:** {source['page']}")
        
        # Input for new query
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response in chat
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    contexts = query_documents(collection, prompt)
                    if contexts:
                        response, sources = generate_rag_response(prompt, contexts)
                        st.markdown(response)
                        
                        # Add sources expander
                        with st.expander("View sources"):
                            for source in sources:
                                st.write(f"**Source:** {source['source']}, **Page:** {source['page']}")
                                st.text(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                        
                        # Add response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                    else:
                        st.markdown("No documents found in the database. Please upload documents first.")

if __name__ == "__main__":
    main()
