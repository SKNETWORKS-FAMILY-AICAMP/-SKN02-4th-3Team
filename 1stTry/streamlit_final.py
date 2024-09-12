import streamlit as st
from rag_openai import main
import os


host = st.secrets["host"]
port = st.secrets["port"]
username = st.secrets["username"]
password = st.secrets["password"]
database_schema = st.secrets["database_schema"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

host1 = os.environ["HOST"] = host
port1 = os.environ["PORT"] = port
username1 = os.environ["USERNAME"] = username
password1 = os.environ["PASSWORD"] = password
database_schema1 = os.environ["DATABASE_SCHEMA"] = database_schema
openai_api_key1 = os.environ["OPENAI_API_KEY"] = openai_api_key


# 1.py의 main 함수를 사용하여 retriever와 llm 가져오기
retriever, rag_chain = main(username1, password1, host1, port1, database_schema1, openai_api_key1)

# RetrievalQA 생성
from langchain.chains import RetrievalQA

# Streamlit 애플리케이션
st.title("Chroma DB 기반 질문 응답 시스템")
question = st.text_input("질문을 입력하세요:")

def get_answer_from_rag_chain(question):
    # RAG 체인에 질문을 전달하여 답변을 생성
    response = rag_chain.invoke(question)
    return response.content if response else "답변을 생성할 수 없습니다."

# 질문 제출 버튼
if st.button("질문 제출"):
    if question:
        # 질문을 RAG 체인에 전달하여 답변 얻기
        answer = get_answer_from_rag_chain(question)
        st.write("답변:", answer)
    else:
        st.write("질문을 입력해 주세요.")
