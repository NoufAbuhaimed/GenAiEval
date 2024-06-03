from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv, find_dotenv
import openai
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate


load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

ollama = Ollama(base_url="http://localhost:11434", model="llama3:8b")

loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_hawaii_wildfires")
data = loader.load()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.split_documents(data)


db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

new_db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = new_db.as_retriever()


template = """You are a helpful AI bot that assits users with {context} and answers about the user's queries.

Please provide the most suitable response for the user's question.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
llm = Ollama(model="llama3:8b")
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke("What is the population of Hawaii in 2023?"))
print(chain.invoke("Tell me about hawai"))
