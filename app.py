import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css

# Returns extracted texts from the uploaded pfd files
def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, str):  # If it's a file path
        try:
            pdf_reader = PdfReader(pdf_docs)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
    else:  # If it's a list of uploaded files
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")
    return text


# Returns a list of text chunks from the extracted raw text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=[" ", ",", "/n"],
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks


# Returns storage of embeddings i.e. numerical representation of your chunks
def get_vectorstore(chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore



# Return conversation
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4o')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        max_tokens_limit=4000
    )
    return conversation_chain

# Returns Chat
def handle_user_question(prompt):
    response = st.session_state.conversation({'question':prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            # st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            with st.chat_message('assistant'):
                st.markdown(message.content)           

# Main function that runs everything 
def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title='LuluChat', page_icon=':cyclone:')
    
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # user_question = st.text_input('')
    prompt = st.chat_input('Ask a question...')

    if prompt:
        if st.session_state.chat_history:
            handle_user_question(prompt=prompt)
        else:
            with st.chat_message('assistant'):
                st.markdown("I'm a blank slate for now—ready to soak up some knowledge! Feed me a document and let's get this learning party started! 🎉📚")

    with st.sidebar:
        st.subheader("TRAIN WITH BILLY'S CV")

        cv_path = 'billy_koech_cv.pdf'
        if st.button("Train on Billy's CV"):
            with st.spinner('Processing...'):
                cv_raw_text = get_pdf_text(cv_path)
                cv_text_chunk = get_text_chunks(cv_raw_text)
                cv_vectorstore = get_vectorstore(cv_text_chunk, openai_api_key)
                st.session_state.conversation = get_conversation_chain(cv_vectorstore)
                st.success("Model trained on Billy's CV")
        
        st.subheader('TRAIN WITH PDFs')
        pdf_docs = st.file_uploader('add PDFs here', accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner('Processing...'):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Break into chunks
                text_chunks = get_text_chunks(raw_text)

                # Vector Store
                vectorstore = get_vectorstore(text_chunks, openai_api_key)

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")
    
    

                           
if __name__ == '__main__':
    main()