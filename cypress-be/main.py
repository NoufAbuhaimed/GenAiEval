from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
import os
import openai
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
import logging

global chat_history
chat_history = []

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



loader = TextLoader("./scraped_data.txt")
pages = loader.load_and_split()


# loader = PyPDFLoader("./database.pdf")
# pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(pages)
print(f"{len(pages)} vs {len(documents)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector = Chroma.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#future work integrate the other llms following and the evaluation metric below
"""
it too long to generate a res
    highest_score = 0
    best_response = None
    metric = FaithfulnessMetric(threshold=0.5)
    
    for model_name, output in responses.items():
        test_case = LLMTestCase(
            input=combined_prompt,
            actual_output=output,
            retrieval_context=search_results
        )
        metric.measure(test_case)
        if metric.score > highest_score:
            highest_score = metric.score
            best_response = (model_name, output)
            
    return best_response

lama model
llama_input = {
        "top_p": 1,
        "prompt": prompt,
        "temperature": 0.5,
        "system_prompt": (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
            "while being safe. you must answer the questions based upon the the context or information provided in the prompt "
            "of answering something not correct. If you don't know the answer to a question, please don't share false information."
        ),
        "max_new_tokens": 500
    }

    llama_output = []
    for event in replicate.stream("meta/llama-2-70b-chat", input=llama_input):
        llama_output.append(event)
    return "".join(llama_output)

falcom model 
falcon_input = {
        "prompt": prompt,
        "temperature": 1
    }

    falcon_output = []
    for event in replicate.stream(
        "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
        input=falcon_input
    ):
        falcon_output.append(event)
    return "".join(falcon_output)


"""
"""import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Any
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import weaviate
import re
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from uuid import uuid4
from langchain_core.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = "sk-proj-P1fr2jSQpx8JfDkbI03PT3BlbkFJiA1L0UrDQUsLCR04In82"


prompt = "You are a helpful assistant in question answering. Your job is to answer user query based on a given context. If the answer is NOT in the context say the question is not related to the information in the wbesite

Context: {context}

User query: {input}
" should be three" but since its a comment will be edited later on


question = "Where is riyadh"
context = "Riyadh is the centre of saudi arabia"
Rag_prompt = PromptTemplate(
template=prompt, input_variables=["input", "context", "question_language", "few_shots_used"] )

# gpt4_llm = ChatOpenAI(streaming=True,model=, temperature=0.0001, callbacks=CallbackManager([StreamingStdOutCallbackHandler()]) )

llm_rag = ChatOpenAI(streaming=True, model="gpt-4", temperature=0.000001, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))  

chain = LLMChain(llm=llm_rag,prompt=Rag_prompt, verbose= False)

if _name=="main_":
    for chunk in chain.stream({
    "input": question,
    "context": context
    }):
        print("\n")
        print(chunk, flush=True)"""


output_parser = StrOutputParser()
retriever = vector.as_retriever()




instruction_to_system = """
Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

question_chain = question_maker_prompt | llm | StrOutputParser()

# Use three sentences maximum and keep the answer concise.\
qa_system_prompt = """You are an AI assistant for question-answering tasks. \
You are designed by Pwc to answer based on the context provided do not answer with outside information. \
Always Generate your asnwer in longer context. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, provide a summary of the context. Do not generate your answer.\


{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]




retriever_chain = RunnablePassthrough.assign(
    context=contextualized_question | retriever  # | format_docs
)

rag_chain = (
        retriever_chain
        | qa_prompt
        | llm
)


def format_chunk(chunk: str) -> str:
    """
    Formats a chunk of text for streaming.
    :param chunk: The text chunk to be formatted.
    :return: Formatted text chunk.
    """
    return (
        chunk.replace("\n", "<new-line>").replace("\t", "<tab>").replace(" ", " ")
    )


@app.get("/chat")
async def AI_search(input: str):
    print(input)
    try:

        async def stream():
            global chat_history
            memory2 = ""
            async for s in rag_chain.astream({"question": input, "chat_history": chat_history}):
                formatted_chunk = format_chunk(str(s.content))
                memory2 += format_chunk(str(s.content))
                print(formatted_chunk)
                yield f"data: {formatted_chunk}\n\n"
                if len(chat_history) >= 4:  # This limits chat history to 2 messages
                    chat_history.pop(0)
                    chat_history.pop(0)
                chat_history.extend([HumanMessage(content=input), AIMessageChunk(content=memory2)])

        headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        return StreamingResponse(stream(), headers=headers)

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the message."
        )
