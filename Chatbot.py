import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-proj-1peGtq93J3rQ67ibUlCvT3BlbkFJI7rnxkY8Whhlw4BzOikv" #openAI key

st.header("Document Analyzer")

#upload pdf files
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file) #reading a file and sent to pdf_reader
    text = ""  #inorder to store entire text
    for page in pdf_reader.pages:
        text += page.extract_text()
       # st.write(text) # for testing purpose

#Break it into chunks
    text_splitter =  RecursiveCharacterTextSplitter(
        separators="\n", #saying to break it on a new line
        chunk_size=1000, #every chunk contains 1000 characters
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        # output results
        # chain -> take the question, get relevant document, pass it to the LLM,
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)



