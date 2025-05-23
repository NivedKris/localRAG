# HR Policy Assistant

An intelligent AI-powered HR Policy Assistant that helps HR professionals quickly access and understand company policies. Built with a fully local RAG (Retrieval-Augmented Generation) system and an intelligent agent architecture that determines whether to handle general HR inquiries or search specific policy documents based on the user's question.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Uploading & Managing HR Policies](#1-uploading--managing-hr-policies)
  - [Interacting with the HR Policy Assistant](#2-interacting-with-the-hr-policy-assistant)
  - [Agent Capabilities](#3-agent-capabilities)
  - [Exploring Policy Insights](#4-exploring-policy-insights)
- [How it works](#how-it-works)
  - [Document Processing & Storage](#document-processing--storage)
  - [Intelligent Agent Architecture](#intelligent-agent-architecture)
  - [Advanced Memory Management](#advanced-memory-management)
  - [User Interface & Visualization](#user-interface--visualization)
- [Deployment](#deployment)
  - [Local Deployment](#local-deployment)
  - [Server Deployment](#server-deployment)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Document Management**:
  - Upload HR policy PDF documents with categories and metadata
  - Automatic text extraction, embedding, and indexing using pypdf and Weaviate
  - Add policy categories and last updated dates for better organization

- **Intelligent Agent Architecture**:
  - Smart routing between policy-specific questions and general inquiries
  - Policy questions use RAG with vector search for accurate information retrieval
  - General inquiries use conversational AI for natural interaction
  - Automatic tool selection based on query content analysis

- **Enhanced User Experience**:
  - Filter queries by policy categories for more targeted results
  - Interactive dashboard with policy statistics and insights
  - View detailed policy sources and references for all responses
  - Common HR policy questions for quick access
  - Three-tab interface (Upload, Chat, Dashboard) for easy navigation

- **Advanced Memory Management**:
  - Separate conversation memories for policy discussions and general chat
  - Window-based memory system to maintain context while preventing overflow
  - Consistent conversation flow across different query types
  
- **Privacy & Performance**:
  - Fully local operation - no data leaves your machine
  - Optimized for performance with batch processing of documents
  - Support for large policy document collections

## Requirements

- Python 3.8+
- Weaviate running locally (vector database)
- Ollama running locally with the following models:
  - all-minilm (for embeddings)
  - tinyllama (for generating responses)
- LangChain for conversation memory and tool orchestration

## Setup

1. Make sure you have Weaviate running locally:
   ```
   docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.24.8
   ```

2. Make sure you have Ollama running locally and the required models installed:
   ```
   ollama pull all-minilm
   ollama pull tinyllama
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

### 1. Uploading & Managing HR Policies

- Navigate to the "Upload Policies" tab
- Upload one or more HR policy PDF files
- Select the appropriate policy category (Recruitment, Benefits, etc.)
- Set the "last updated" date for the policy documents
- Click "Process Policy Documents" to extract text and store embeddings
- The progress bar will show processing status and confirm successful upload

### 2. Interacting with the HR Policy Assistant

- Switch to the "Policy Assistant" tab
- Choose how you want to interact:
  - Ask custom questions about company policies
  - Make general inquiries about HR topics
  - Use one of the common HR policy questions buttons for quick access
  - Engage in follow-up discussions about previous topics
- Filter results by selecting a policy category from the sidebar
- The system automatically determines whether to:
  - Search policy documents using RAG (for policy questions)
  - Respond conversationally (for general inquiries, greetings, etc.)
- For policy-related responses:
  - Expand the "View policy sources" section to see the source documents
  - Check policy categories, page numbers, and last updated dates
  - View the relevant text excerpts that informed the response

### 3. Agent Capabilities

- **Policy-Specific Questions**:
  - "What is our maternity leave policy?"
  - "How many vacation days do employees receive?"
  - "What is the process for reporting harassment?"
  - "Explain our remote work policy."
  - "When was the health insurance policy last updated?"

- **Follow-up Questions**:
  - "Who should I contact about this?"
  - "When was this policy implemented?"
  - "Are there any exceptions to this rule?"
  - "Can you provide more details about that?"

- **General HR Inquiries**:
  - "How should I prepare for an employee performance review?"
  - "What are best practices for onboarding new employees?"
  - "How can I improve employee engagement?"
  - "What should I include in a job description?"

- **Basic Conversation**:
  - "Hello, how can you help me today?"
  - "Thank you for the information."
  - "Tell me about your capabilities."

### 4. Exploring Policy Insights

- Check the "Policy Dashboard" tab for statistics and insights
- View policy distribution by category with the category breakdown table
- See recently updated policies with their categories and dates
- Search for specific policy documents by name using the search function
- Use these insights to identify gaps in policy coverage or outdated policies

## How it works

### Document Processing & Storage

1. **Document Ingestion**:
   - HR policy PDF documents are uploaded with metadata (category, last updated date)
   - Documents are processed using the pypdf library and split into chunks by page
   - Each chunk is embedded using the all-minilm model via Ollama
   - Embeddings and metadata are stored in the Weaviate vector database with proper indexing

2. **Metadata Management**:
   - Each document chunk is stored with comprehensive metadata:
     - Source document name
     - Page number
     - Policy category (Recruitment, Benefits, etc.)
     - Last updated date
   - This enables powerful filtering and source tracking

### Intelligent Agent Architecture

3. **Query Analysis & Tool Selection**:
   - When a user asks a question, the system uses an LLM-based decision function
   - The query is analyzed to determine if it's policy-related or general conversation
   - A prompt template guides the model to choose between two specialized tools:
     ```python
     def determine_tool(query):
         # Create a decision prompt
         prompt = template.format(tool_descriptions=tool_descriptions, query=query)
         # Get LLM decision
         response = ollama.generate(model="tinyllama", prompt=prompt, stream=False)
         # Parse response to select appropriate tool
         tool_choice = response["response"].lower()
         # Return the selected tool function
         if "query_hr_policies" in tool_choice or "policies" in tool_choice:
             return query_hr_policies
         else:
             return general_conversation
     ```

4. **Policy Question Handling (RAG Pipeline)**:
   - Implemented as a LangChain tool with `@tool` decorator
   - The query is embedded using all-minilm and similar policy chunks are retrieved from Weaviate
   - Results are filtered by selected policy category if specified
   - Policy-specific conversation history is added from HR memory
   - The retrieved context and conversation history are sent to tinyllama
   - Sources are tracked and displayed alongside the response

5. **General Conversation Handling**:
   - Also implemented as a LangChain tool with `@tool` decorator
   - A separate conversation memory maintains the chat context
   - The LLM responds conversationally without searching policy documents
   - Maintains a natural, helpful tone for non-policy questions

### Advanced Memory Management

6. **Dual Memory System**:
   - Separate memory systems for policy discussions and general chat:
     ```python
     # Initialize conversational memories with a window of 5 exchanges
     st.session_state.general_memory = ConversationBufferWindowMemory(return_messages=True)
     st.session_state.general_memory.k = 5
     
     st.session_state.hr_memory = ConversationBufferWindowMemory(return_messages=True)
     st.session_state.hr_memory.k = 5
     ```
   - Each memory maintains a window of 5 exchanges to preserve context
   - Prevents context overflow while maintaining relevant conversation history
   - Enables seamless follow-up questions within each domain

### User Interface & Visualization

7. **Interactive Multi-Tab Interface**:
   - Upload tab: Document management with metadata assignment
   - Chat tab: Intelligent assistant with source references
   - Dashboard tab: Policy statistics and insights

8. **Policy Analytics**:
   - Document distribution by category
   - Recently updated policies tracking
   - Search functionality for policy discovery
   - Source attribution for all policy responses

## Deployment

### Local Deployment

For personal or small team use, the local setup described in the Setup section is sufficient. This ensures maximum privacy as all data and processing remain on your local machine.

### Server Deployment

For team-wide access, you can deploy the HR Policy Assistant on an internal server:

1. **Set up a dedicated server** with the following requirements:
   - Python 3.8+ environment
   - Docker for Weaviate
   - Ollama for local LLM inference

2. **Clone the repository** to your server:
   ```bash
   git clone https://github.com/yourusername/hr-policy-assistant.git
   cd hr-policy-assistant
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Weaviate** as a background service:
   ```bash
   docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.24.8
   ```

5. **Start Ollama** and download required models:
   ```bash
   # Start Ollama service
   ollama serve &
   
   # Download models
   ollama pull all-minilm
   ollama pull tinyllama
   ```

6. **Run the application** with Streamlit's server options:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

7. **Set up a reverse proxy** (optional) with Nginx or Apache to add SSL and authentication.

## Future Improvements

The HR Policy Assistant can be enhanced with several planned improvements:

1. **Advanced Document Processing**:
   - Support for more document formats (DOCX, HTML, Markdown)
   - Improved chunking strategies (semantic chunking instead of page-based)
   - Automatic category detection for policy documents

2. **Enhanced Agent Capabilities**:
   - Multi-tool agent with specialized tools for different HR functions
   - Document comparison capabilities to identify policy changes
   - Policy summarization and simplification features

3. **User Experience Enhancements**:
   - User authentication and role-based access
   - Customizable UI themes and layouts
   - Mobile-responsive design for on-the-go access

4. **Performance Optimizations**:
   - Caching frequently asked questions
   - Background indexing for large document sets
   - Parallel processing for batch document uploads

5. **Integration Capabilities**:
   - Connect with HR management systems
   - Calendar integration for policy review reminders
   - Export and sharing functionality for policy insights

## Conclusion

The HR Policy Assistant demonstrates how local RAG systems with intelligent agent architecture can transform the way HR professionals interact with company policies. By combining document retrieval, conversational AI, and structured policy management, the system provides a comprehensive solution for policy access and understanding.

The dual-memory architecture ensures that conversations maintain context appropriately while limiting memory usage to prevent overflow. This approach creates a seamless experience where users can freely switch between policy questions and general inquiries without losing conversation flow.

Whether you're looking up specific policy details, getting general HR advice, or analyzing your policy database, the HR Policy Assistant provides a powerful, privacy-focused solution that runs entirely on your local machine.

## Contributing

Contributions to the HR Policy Assistant are welcome! Here's how you can contribute:

1. **Fork the repository** on GitHub
2. **Create a new branch** for your feature or bugfix
3. **Make your changes** and commit them with descriptive messages
4. **Push your branch** to your forked repository
5. **Submit a pull request** to the main repository

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments for complex code sections
- Update documentation when changing functionality
- Add tests for new features when possible

### Reporting Issues

If you encounter any bugs or have feature requests, please:

1. Check if the issue already exists in the GitHub issues
2. Create a new issue with a clear description and steps to reproduce

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: May 23, 2025*
