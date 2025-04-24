from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

ollama_llm = ChatOllama(
    model="gemma:2b",
    temperature=0
)


def get_answer(question):
    answer = llm.invoke(question)
    return answer.content


def get_answer_ollama(question):
    answer = ollama_llm.invoke(question)
    return answer.content
