from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings  
import os

# Set your OpenRouter key
os.environ["OPENAI_API_KEY"] = "put_your_openrouter_api_key" 

# LLM setup (OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    model="google/gemma-3n-e2b-it:free",
    temperature=0.5,
)

# Use local embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# YouTube to VectorStore
def get_doc_from_youtube(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(split_docs, embedding)
    return vectorstore

# LLM + prompt pipe
def get_response(question, context):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\nQuestion: {question}\nAnswer:",
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
