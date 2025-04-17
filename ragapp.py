import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot

@st.cache_resource
def initialize_chatbot(file_path):
    try:
        processor = DocumentProcessor(file_path)
        texts = processor.load_and_split()
        
        if not texts:
            raise ValueError("No text extracted from the document")
        
        indexer = EmbeddingIndexer()
        vectorstore = indexer.create_vectorstore(texts)
        rag_chain = RAGChain(vectorstore)
        return Chatbot(rag_chain.create_chain())
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chat Assistant Demo")


# File uploader with PDF support
uploaded_file = st.file_uploader(
    "Upload a document for the knowledge base", 
    type=["pdf", "txt"],
    help="Upload a PDF or TXT file to create your knowledge base"
)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Initialize chatbot with loading spinner
    with st.spinner("Processing document and creating knowledge base..."):
        chatbot = initialize_chatbot(tmp_file_path)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    if chatbot:
        st.success(f"✅ Successfully loaded: {uploaded_file.name}")
        
        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the document"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.get_response(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            chatbot.clear_history()
            st.rerun()
    else:
        st.error("Failed to initialize chatbot. Please check your document and try again.")
else:
    st.info("👆 Please upload a PDF or TXT file to start chatting!")