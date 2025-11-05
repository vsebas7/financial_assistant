import streamlit as st
import tempfile
import pandas as pd
from tools import get_finance_agent
from datetime import datetime 
from streamlit_pdf_viewer import pdf_viewer
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def home():
    st.set_page_config(
        page_title="Vincent Sebastian Capstone Project",
        page_icon="👨‍💻"
    )
    
    # Header Section
    st.title("Financial Assistant")
    st.markdown("---")
    
    # Introduction Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Welcome to Your Financial Assistant
        
        Welcome! This specialized financial AI assistant is designed specifically for 
        **Indonesian capital market (IDX) analysis** and intelligent document processing.
        """)
        
    with col2:
        st.info("""
        **👨‍💻 Developer**  
        Vincent Sebastian
        
        **📅 Created**  
        November 2025
        """)
    
    st.markdown("---")
    
    # Key Features Section
    st.markdown("## 🚀 Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        ### 💬 Indonesian Market Chatbot
        """)
        
    with feature_col2:
        st.markdown("""
        ### 📄 Document Assistant
        """)
    
    st.markdown("---")
    
    # How to Use Section
    st.markdown("## 📖 How to Use")
    
    tab1, tab2 = st.tabs(["💬 Financial Chatbot", "📄 Document Assistant"])
    
    with tab1:
        st.markdown("""
        ### Getting Started with Indonesian Market Analysis
        
        **Step-by-Step Guide:**
        
        1. **📍 Navigate to Financial Chatbot**
           - Click on "💬 Financial Chatbot" in the sidebar
           
        2. **💭 Start Your Analysis**
           - Ask questions about IDX-listed companies
        """)
    
    with tab2:
        st.markdown("""
        ### Document Analysis Workflow
        
        **Step-by-Step Guide:**
        
        1. **📍 Navigate to Document Assistant**
           - Click on "📄 Document Assistant" in the sidebar
           
        2. **📁 Upload Your Document**
           - Use the file uploader to select a PDF and less than 200mb
           - Documents can be financial reports, research papers, etc. 
           
        3. **🔍 Choose Analysis Mode:**
           - **📝 Summarization**: Get quick document overview
           - **💬 Q&A Mode**: Ask specific questions about content
        """)
    
    st.markdown("---")
    
def pageFinancialChatbot():
    st.set_page_config(page_title="Financial Assistant Chatbot",page_icon="💬")
    st.subheader("Financial Assistant Chatbot with LLM Agents and RAG", divider=True)

    if "selectbox_selection" not in st.session_state:
        st.session_state['selectbox_selection'] = ["Default Chat"]

    selectbox_selection = st.session_state['selectbox_selection']

    if st.sidebar.button("✍️ Create New Chat", use_container_width=True):
        selectbox_selection.append(f"New Chat - {datetime.now().strftime('%H:%M:%S')}")

    session_id = st.sidebar.selectbox("Chats", options=selectbox_selection, index=len(selectbox_selection)-1)

    chat_history = StreamlitChatMessageHistory(key=session_id)

    for message in chat_history.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    prompt = st.chat_input("Ask your question here!")
    agent = get_finance_agent()
    if prompt:
        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message("ai"):
            response = agent.invoke({"input": prompt},
                                    config={"configurable": {"session_id": session_id}})

            st.markdown(response['output'])


def pageDocumentQna():
    MODEL = "llama-3.3-70b-versatile"
    llm = ChatGroq(model=MODEL, temperature=0.0)

    # --- App Configuration ---
    st.set_page_config(page_title="AI Document Assistant", page_icon="📄")

    # --- AI Logic ---
    @st.cache_resource()
    def create_pdf_agent(file) :

        # Simpan PDF ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file :
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200   
        )

        all_splits = text_splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vector_store_pdf = InMemoryVectorStore(embedding)
        _ = vector_store_pdf.add_documents(documents=all_splits)

        tools = create_retriever_tool(vector_store_pdf.as_retriever(search_kwargs={'k': 5}),
                                        name = "pdf_document_retriever",
                                        description= "Retrieve PDF as context to accurately and concisely answer the user's question")

        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        '''
                        You are a helpful and detail-oriented assistant. 
                        You are provided with a tool to retrieve a PDF document from a vector store. 
                        Use the context to accurately and concisely answer the user's question. 

                        You need to follow these rules:
                        - Only use data from tools provided. Never guess or use outside data.  
                        - If data is not available, say so clearly and do not make it up. Suggest alternative sources if possible.  
                        - Add follow-up questions to help users dive deeper  
                        '''
                    ),
                    (
                        "human", "{input}"
                    ),
                    MessagesPlaceholder("agent_scratchpad")
                ]
            )

        # Create the Agent and AgentExecutor
        agent = create_tool_calling_agent(llm, [tools], prompt)
        
        agent_executor_pdf = AgentExecutor(agent=agent, tools=[tools], verbose=True)

        return agent_executor_pdf

    # --- Sidebar Upload ---
    st.sidebar.header("📂 Upload Your Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload the document you want to summarize or query."
    )

    # --- Page Navigation ---
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Want to:", ["📝 Summarize", "💬 QnA"])

    # --- Main Title ---
    st.title("Document Assistant")

    # --- Summarization ---
    if page == "📝 Summarize":
        st.subheader("📄 Document Summarization")

        if uploaded_file is not None :
            with st.expander("📘 Uploaded Document", expanded=False):
                if uploaded_file is not None:
                    # Read the uploaded file as binary data
                    binary_data = uploaded_file.getvalue()

                    # Display the PDF using pdf_viewer
                    pdf_viewer(input=binary_data, width=700)
                # st.pdf(uploaded_file, height=700)
            
            st.markdown("### Generated Summary")
            summary_placeholder = st.empty()
            
            if st.button("Generate AI Summary") :
                with st.spinner("Generating summary...") :
                    agent_executor_pdf = create_pdf_agent(uploaded_file)
                    prompt_input = "Please summarize the document into 1 sentence"
                    response = agent_executor_pdf.invoke({"input":prompt_input})
                    summary_placeholder.success(response['output'])
        else:
            st.warning("Please upload a document first to generate a summary.")

    # --- QnA ---
    elif page == "💬 QnA":
        st.subheader("💬 Chat with Your Document")

        # AI Logic
        if uploaded_file is not None :
            agent_executor_pdf = create_pdf_agent(uploaded_file)
            st.markdown("Ask any question related to your document below 👇")
            user_input = st.chat_input("Type your question here...")

            if user_input:
                with st.chat_message("human"):
                    st.write(user_input)

                with st.chat_message("ai"):
                    response = agent_executor_pdf.invoke({"input" : user_input})
                    st.write(response['output'])
        else:
            st.warning("Please upload a document to start chatting with it.")

pages = [
    st.Page(home, title="Overview"),
    st.Page(pageFinancialChatbot, title="💬 Financial Chatbot", url_path="/Chatbot"),
    st.Page(pageDocumentQna, title="📄 Document Assistant", url_path="/Docbot"),
]

pg = st.navigation(pages)
pg.run()