# PDFChat - Intelligent Document Conversation Assistant

PDFChat is an advanced document interaction tool that transforms the way users engage with PDF documents. By leveraging the power of LLMs and vector databases, it enables natural conversation-style interactions with document content, making information retrieval more intuitive and efficient.

## 🌟 Features

- **Interactive Chat Interface**: Natural conversation-style interaction with your documents
- **Multi-Document Support**: Process and analyze multiple PDF files simultaneously
- **Intelligent Context Retention**: Maintains conversation history for more coherent interactions
- **Real-time Processing**: Dynamic document processing with progress indicators
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI
- **Vector-Based Search**: Efficient document querying using FAISS vector storage

## 🔧 Technical Architecture

The application is built with several key components:

1. **Document Processing Pipeline**:

   - PDF text extraction using PyPDF2
   - Text chunking with RecursiveCharacterTextSplitter
   - Vector embedding generation using OpenAI Embeddings
   - Vector storage using FAISS

2. **Conversation Chain**:

   - GPT-4 language model integration
   - Conversation buffer memory for context retention
   - ConversationalRetrievalChain for managing Q&A flow

3. **User Interface**:
   - Streamlit-based frontend
   - File upload functionality
   - Real-time chat interface
   - Processing status indicators

Add the following to your `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎮 Usage

1. You can use the application in two ways:

   - Upload custom PDF documents using the sidebar
   - Use the quick access feature to analyze the pre-configured CV

2. After processing the documents, interact with the content through the chat interface

## 💡 Code Structure and Examples

### PDF Text Extraction

Extracts text from PDF files

```python
def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, str):  # For file paths
        try:
            pdf_reader = PdfReader(pdf_docs)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
    else:  # For uploaded files
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")
    return text
```

### Text Chunking

Splits text into manageable chunks with overlap

```python
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ",", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
```

### Vector Store Creation

Creates and manages vector embeddings

```python
def get_vectorstore(chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore
```

### Conversation Chain Setup

Sets up the conversation pipeline

```python
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4')
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        max_tokens_limit=4000
    )
    return conversation_chain
```

### Chat Handler

Processes user inputs and manages chat history

```python
def handle_user_question(prompt):
    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            with st.chat_message('assistant'):
                st.markdown(message.content)
```

### Complete Application Flow Example

```python

# Usage in Streamlit app
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
```

## ⚠️ Important Notes

1. The application requires a valid OpenAI API key
2. Large PDF files may take longer to process
3. The chat history is maintained only for the current session
4. The model uses GPT-4 for optimal performance

## 🔒 Security Considerations

- API keys are managed through environment variables
- Document processing is done locally
- No permanent storage of uploaded documents
- Session-based conversation history
