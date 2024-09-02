import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from htmlTemplates import css, bot_template, user_template
import numpy as np
# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Function to create a FAISS vector store
def get_vectorstore(text_chunks):
    try:
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the text chunks to get their embeddings
        embeddings = model.encode(text_chunks, show_progress_bar=True)

        # Convert embeddings to FAISS-compatible format
        embeddings = np.array(embeddings)

        # Create the FAISS vector store with embeddings and corresponding text chunks
        vectorstore = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)

        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Custom conversation chain using local GPT-2 model
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=lambda input_text: generate_response(input_text, tokenizer, model),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to generate a response using GPT-2
def generate_response(input_text, tokenizer, model):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.error("Please process the PDFs before asking questions.")
        
    st.image("docs/PDF-LangChain.jpg", use_column_width=True) 
    
    st.image("docs/open_ai_api_flow.jpeg", use_column_width=True)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Ensure the vectorstore is created before proceeding
                    if vectorstore:
                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    else:
                        st.error("Failed to create vector store.")
            else:
                st.error("Please upload at least one PDF.")
        
        # Add LinkedIn hyperlink
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '[Connect with me on LinkedIn](https://linkedin.com/in/anupvjpawar)')

if __name__ == '__main__':
    main()
