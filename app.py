import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure API keys are loaded
if not groq_api_key:
    st.error("GROQ_API_KEY is missing! Please check your environment variables.")
if not google_api_key:
    st.error("GOOGLE_API_KEY is missing! Please check your environment variables.")

genai.configure(api_key=google_api_key)

# Initialize AI Model
model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Chat Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
This is your introduction - Your name is "JustiFI : AI That Simplifies Justice👩🏻‍⚖️📚𓍝" and you are developed by " R LAKSHMI SURENDRA".
You're a go-to platform for all legal queries. You are embedded with the entire data of three newly enacted criminal laws:
- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagrik Suraksha Sanhita (BNSS)
- Bharatiya Sakshya Adhiniyam (BSA)

Greet users with '🙏'.
Ensure to provide suitable, accurate, and concise answers.
If the answer is not in the provided context, reply with: "Answer is not available in the context".

Generally, user starts with a greeting first. So, greet them accordingly, and ask them for their queries.

You'll never use any arabic words in your conversation.

If user asks anything about yourself, then answer them with polite words. don't give very straight forward one liner answers.

Ensure to provide suitable answers - if the answer demands more detail, provide it, but don't give lengthy answers unnecessarily.
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer

Context:
{context}

Question:
{input}

Answer:
""")

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None


# Function to process documents
def get_vector_store():
    if st.session_state.vectors is None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./bns")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunking
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings)  # Vector Store


def main():
    st.set_page_config(page_title='JustiFi', layout='wide', page_icon="⚖️")
    st.sidebar.title("JustiFI : AI That Simplifies Justice 𓍝")
    st.sidebar.image("logo/side bar logo.png", width=150)

    with st.sidebar.container():
        st.image('logo/logo.png', use_container_width=True, caption='JustiFI : AI That Simplifies Justice👩🏻‍⚖️📚𓍝')
        with st.expander("About Us", icon=":material/info:"):
            st.success(
                "Hii, I am your go-to platform for all legal queries. We provide accurate information on Indian laws.")
        st.sidebar.markdown("---")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Embedding button
    with st.sidebar:
        st.title("Start the App by Clicking Here ✅")
        doc = st.button("Start Documents Embedding")

        if doc:
            with st.spinner("Processing..."):
                get_vector_store()
                st.info("VectorDB Store is Ready")
                st.success("You're good to go! Ask Questions now...")

    # Chat Input (Disabled if embeddings are not ready)
    user_question = st.chat_input("Ask me a legal question...") if st.session_state.vectors else st.chat_input(
        "Embedding in progress... Please wait!", disabled=True)

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.vectors:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    document_chain = create_stuff_documents_chain(model, prompt_template)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    start_time = time.process_time()
                    response = retrieval_chain.invoke({'input': user_question})
                    print("Response time:", time.process_time() - start_time)

                    st.write(response['answer'])

                    with st.expander("Document Similarity Search"):
                        for i, doc in enumerate(response["context"]):
                            st.write(doc.page_content)
                            st.write("--------------------------------")

            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    st.sidebar.title("Looking to Restart your Conversation 🔄")
    st.sidebar.button('Start a New Chat', on_click=clear_chat_history)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<h3 style='text-align: center;'>Developed with ❤️ for GenAI by <a style='text-decoration: none' href='https://www.linkedin.com/in/rlaskshmisurendra'>  R.LAKSHMI SURENDRA</a></h3>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
