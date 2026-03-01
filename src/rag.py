import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "data/chroma_db"
DOCS_PATH = "data/destinations"

def load_and_index_documents():
    if os.path.exists(CHROMA_PATH):
        return
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k":3})

def retrieve_context(query:str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)