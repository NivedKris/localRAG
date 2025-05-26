import streamlit as st
import weaviate
import weaviate.exceptions
import ollama
import pypdf
import os
import tempfile
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

# Set page configuration
st.set_page_config(page_title="HR Policy Document Manager", layout="wide", page_icon="📁")

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
    # Store UUIDs of inserted objects mapped to their source document
    inserted_uuids = []
    
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk in text_chunks:
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk["text"])
            uuid = batch.add_object(
                properties={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "policy_category": chunk.get("policy_category", "General"),
                    "last_updated": chunk.get("last_updated", "")
                },
                vector=response["embedding"],
            )
            inserted_uuids.append(uuid)
    
    # Store the mapping in session state for use in removal
    if "document_uuid_map" not in st.session_state:
        st.session_state.document_uuid_map = {}
    
    # Map the document name to its UUIDs
    source_name = text_chunks[0]["source"] if text_chunks else None
    if source_name:
        st.session_state.document_uuid_map[source_name] = inserted_uuids
    
    return len(text_chunks)

# Function to remove documents from database by source name
def remove_document(collection, document_name):
    """Remove all chunks related to a specific document from the database
    
    Args:
        collection: Weaviate collection
        document_name: Name of the document to remove
        
    Returns:
        int: Number of objects deleted
    """
    # Check if we have stored UUIDs for this document
    if "document_uuid_map" in st.session_state and document_name in st.session_state.document_uuid_map:
        uuids = st.session_state.document_uuid_map[document_name]
        before_count = len(uuids)
        
        try:
            # Delete objects one by one using their UUIDs
            for uuid in uuids:
                try:
                    collection.data.delete_by_id(uuid)
                except Exception as e:
                    print(f"Warning: Failed to delete UUID {uuid}: {e}")
            
            # Remove the entry from our mapping
            del st.session_state.document_uuid_map[document_name]
            return before_count
            
        except Exception as e:
            raise Exception(f"Failed to delete objects using stored UUIDs: {e}")
    
    # Fallback to filtering if we don't have stored UUIDs
    document_filter = wvc.query.Filter.by_property("source").equal(document_name)
    
    # Count matching documents before deletion
    try:
        # Try using the newer API format
        query_result = collection.query.fetch_objects(
            filters=document_filter,
        )
        if hasattr(query_result, 'total_count'):
            before_count = query_result.total_count
        else:
            # Alternative approach if total_count is not available
            before_count = len(query_result.objects)
    except Exception as e:
        # Fallback approach if counting fails
        print(f"Warning: Error counting objects before deletion: {e}")
        before_count = 0
    
    # Try multiple deletion approaches
    deleted = False
    error_messages = []
    
    # Approach 1: Try to fetch objects and delete by ID
    if not deleted:
        try:
            results = collection.query.fetch_objects(
                filters=document_filter,
                return_properties=["source"],
                include_vector=False
            )
            
            if hasattr(results, 'objects') and results.objects:
                uuids = [obj.uuid for obj in results.objects]
                
                # Delete objects one by one
                for uuid in uuids:
                    try:
                        collection.data.delete_by_id(uuid)
                    except Exception as e:
                        error_messages.append(f"Failed to delete UUID {uuid}: {e}")
                
                deleted = True
        except Exception as e:
            error_messages.append(f"Approach 1 failed: {e}")
    
    # Approach 2: Try with older API format
    if not deleted:
        try:
            # Use the older API format
            results = collection.query.get(
                ["source"]
            ).with_where({
                "path": ["source"],
                "operator": "Equal",
                "valueString": document_name
            }).do()
            
            if "data" in results and "Get" in results["data"] and collection.name in results["data"]["Get"]:
                objects = results["data"]["Get"][collection.name]
                uuids = []
                
                for obj in objects:
                    if "_additional" in obj and "id" in obj["_additional"]:
                        uuids.append(obj["_additional"]["id"])
                
                # Delete objects one by one
                for uuid in uuids:
                    try:
                        collection.data.delete_by_id(uuid)
                    except Exception as e:
                        error_messages.append(f"Failed to delete UUID {uuid}: {e}")
                
                deleted = True
        except Exception as e:
            error_messages.append(f"Approach 2 failed: {e}")
    
    if not deleted and error_messages:
        raise Exception(f"Failed to delete objects: {'; '.join(error_messages)}")
    
    return before_count

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

def main():
    client = get_weaviate_client()
    collection = initialize_collection(client)
    
    # Styled header
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>HR Policy Document Manager</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; margin-bottom: 20px;'>Upload and manage policy documents for the HR Policy Assistant</p>", unsafe_allow_html=True)
    
    # Add horizontal line for visual separation
    st.markdown("<hr style='margin: 10px 0px 20px 0px;'>", unsafe_allow_html=True)
    
    # Create tabs for Upload and Dashboard
    tab1, tab2 = st.tabs(["📁 Upload Policies", "📊 Policy Dashboard"])
    
    # Upload tab
    with tab1:
        st.header("Upload HR Policy Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_files = st.file_uploader("Choose PDF policy documents", 
                                            type="pdf", 
                                            accept_multiple_files=True)
        
        with col2:
            if uploaded_files:
                st.info(f"{len(uploaded_files)} files selected")
                
                # Additional metadata for policy documents
                policy_category = st.selectbox(
                    "Select Policy Category",
                    options=POLICY_CATEGORIES[1:],  # Exclude "All Categories"
                    index=0
                )
                
                last_updated = st.date_input("Policy Last Updated Date")
                
                if st.button("Process Policy Documents"):
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
                            text_chunks = extract_text_from_pdf(f, file_name, policy_category, last_updated.strftime("%Y-%m-%d"))
                            chunks_count = embed_and_store(collection, text_chunks)
                            total_chunks += chunks_count
                        
                        # Remove the temporary file
                        os.unlink(tmp_path)
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    progress_bar.progress(1.0)
                    st.success(f"Successfully processed {len(uploaded_files)} policy documents with {total_chunks} text chunks.")
    
    # Dashboard tab - Policy Insights
    with tab2:
        st.header("HR Policy Dashboard")
        
        try:            # Get policy statistics
            try:
                # Retrieve all policy documents with error handling
                try:
                    results = collection.query.fetch_objects(
                        limit=1000, 
                        return_properties=["source", "policy_category", "last_updated"]
                    )
                    documents = results.objects
                except AttributeError:
                    # Fall back to a different query approach if the API has changed
                    st.warning("Using alternative query method due to API differences")
                    results = collection.query.get(
                        ["source", "policy_category", "last_updated"]
                    ).with_limit(1000).do()
                    
                    # Extract objects from different response format
                    if "data" in results and "Get" in results["data"] and collection.name in results["data"]["Get"]:
                        documents = results["data"]["Get"][collection.name]
                    else:
                        documents = []
                        st.warning("Could not extract documents from response")
                
                if documents and len(documents) > 0:
                    # Count by category
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Policies by Category")
                        
                        # Count documents by category
                        category_counts = {}
                        for doc in documents:
                            category = doc.properties.get("policy_category", "General")
                            if category in category_counts:
                                category_counts[category] += 1
                            else:
                                category_counts[category] = 1
                                
                        # Display as a table
                        st.dataframe(
                            data={"Category": category_counts.keys(), "Count": category_counts.values()},
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("Recently Updated Policies")
                        
                        # Get unique documents by source and their last updated date
                        docs_by_source = {}
                        for doc in documents:
                            source = doc.properties.get("source")
                            last_updated = doc.properties.get("last_updated", "")
                            category = doc.properties.get("policy_category", "General")
                            
                            if source and source not in docs_by_source:
                                docs_by_source[source] = {
                                    "last_updated": last_updated,
                                    "category": category
                                }
                        
                        # Convert to a list of dictionaries for the dataframe
                        recent_docs = [
                            {"Policy": source, "Category": data["category"], "Last Updated": data["last_updated"]}
                            for source, data in docs_by_source.items()
                        ]
                        
                        # Sort by last updated date (descending)
                        recent_docs.sort(key=lambda x: x["Last Updated"], reverse=True)
                          # Display as a table
                        st.dataframe(recent_docs, use_container_width=True)
                        
                        # Show stored document UUIDs if available (for debugging purposes)
                        if "document_uuid_map" in st.session_state and st.session_state.document_uuid_map:
                            with st.expander("📝 Document UUID Mapping"):
                                st.info(f"UUID mapping stored for {len(st.session_state.document_uuid_map)} documents")
                                for doc_name, uuids in st.session_state.document_uuid_map.items():
                                    st.write(f"**{doc_name}**: {len(uuids)} chunks")
                      # Search functionality for policies
                    st.subheader("Search Policies")
                    search_col1, search_col2 = st.columns([3, 1])
                    
                    with search_col1:
                        search_term = st.text_input("Search for policy documents by name", placeholder="Enter keywords...")
                    
                    with search_col2:
                        search_button = st.button("Search", key="search_button")
                    
                    if search_term and search_button:
                        st.subheader(f"Search Results: {search_term}")
                        
                        # Simple search based on source name containing the search term
                        results = [
                            {"Policy": source, "Category": data["category"], "Last Updated": data["last_updated"]}
                            for source, data in docs_by_source.items()
                            if search_term.lower() in source.lower()
                        ]
                        
                        if results:
                            st.dataframe(results, use_container_width=True)
                        else:
                            st.info("No matching policies found.")
                          # Document removal functionality
                    st.subheader("Remove Policy Documents")
                    st.warning("⚠️ Warning: This action cannot be undone. The document will be completely removed from the database.")
                    
                    # Get all available document names
                    if docs_by_source:
                        document_names = list(docs_by_source.keys())
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            document_to_remove = st.selectbox(
                                "Select document to remove",
                                options=document_names,
                                key="document_to_remove"
                            )
                        
                        with col2:
                            remove_button = st.button("🗑️ Remove Document", key="remove_button", type="primary", use_container_width=True)
                        
                        if remove_button and document_to_remove:
                            try:
                                with st.spinner(f"Removing document: {document_to_remove}"):
                                    deleted_count = remove_document(collection, document_to_remove)
                                    st.success(f"Successfully removed document '{document_to_remove}' ({deleted_count} chunks deleted)")
                                    st.info("Refresh the page to update the document lists.")
                                    
                                    # Add a refresh button for convenience
                                    if st.button("🔄 Refresh Dashboard"):
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error removing document: {str(e)}")
                                st.info("Please try again or check the logs for more information.")
                    else:
                        st.info("No documents available to remove.")
                else:
                    st.info("No policy documents available in the database.")
            except Exception as e:
                st.error(f"Error retrieving policy statistics: {e}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please upload policy documents to see statistics.")    # Add link to main app
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("[Go to HR Policy Assistant](http://localhost:8501/)")
    st.sidebar.info("This is the document management interface. Use the HR Policy Assistant for Q&A.")

if __name__ == "__main__":
    main()
