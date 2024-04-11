import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import time
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_conversation_chain(user_question):
    response = st.session_state.conversation({'question': user_question})
    ans = response["answer"]
    st.write(ans)
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config("Chat with mulitple PDFs",page_icon=":books")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your document:")
    
    if user_question:
        handle_conversation_chain(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                start_time = time.time()
                raw_text = get_pdf_text(pdf_docs)
                end_time = time.time()
                print(f"the time for extracting data to a string is {end_time - start_time}")
                # get text chuncks
                start_time = time.time()
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                end_time=time.time()
                print(f"The time it took to make it chunks is {end_time-start_time}")
                
                
                # create vector store
                start_time=time.time()
                vectorstore = get_vectorstore(text_chunks)
                end_time=time.time()
                print("processing is done !!!!!!!!")
                print(f"the time it took for embedding is {end_time-start_time}")
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                

    
if __name__ == '__main__':
    main()