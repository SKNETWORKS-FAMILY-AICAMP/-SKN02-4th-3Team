from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import pymysql

import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('DB_HOST')
port = os.getenv('port')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database_schema = os.getenv('DB_NAME')
openai_api_key = os.getenv('OPEN_API_KEY')

def db_connect():
    conn = pymysql.connect(
        host=host,
        user=username,
        password=password,
        database=database_schema,
        charset="utf8"
    )
    return conn

def fetch_long_text():
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute('SELECT txt FROM naver_finance ORDER BY time DESC LIMIT 1')
        result = cursor.fetchone()
        long_text = result[0] if result else None
        cursor.close()
        conn.close()
        return long_text
    except pymysql.Error as e:
        print(f"Error: {e}")
        return None
def url_to_db(username,password,host,port,database_schema):
    mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}?charset=utf8"
    db = SQLDatabase.from_uri(mysql_uri)
    db1=db.run("SELECT * FROM news WHERE DATE(time) = CURDATE();")
    db2=fetch_long_text()
    db3=db2+db1
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(db2)
    return db3

def split_text(db, chunk_size=500, chunk_overlap=0):
    """문서 내용을 작은 chunk 단위로 나누는 함수."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_text(db)
    return splits


def create_vector_store(splits, OPENAI_API_KEY):
    """문서를 벡터로 변환하여 Vector Store를 생성하고 저장하는 함수."""
    # Document 객체 리스트 생성
    documents = [Document(page_content=split, metadata={}) for split in splits]
    ids = [f"doc{i+1}" for i in range(len(documents))]

    # OpenAI Embeddings 초기화
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Chroma 벡터 스토어 생성 및 저장 설정
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        persist_directory="./chroma_db"  # 저장될 디렉토리 지정
    )
    
    vectorstore.persist()

    # 벡터 스토어를 검색 가능한 리트리버로 설정
    retriever = vectorstore.as_retriever()
    return retriever

def create_prompt_template():
    """곤란한 질문에도 적절한 답변을 제공하는 커스텀 프롬프트 템플릿을 생성하는 함수."""
    template = """
    기본적으로 네이버 파이낸스에 있는 정보를 토대로 대답하십시오.
    뉴스에 대한 질문에는 뉴스 제목, 요약, 링크를 첨부해서 설명해 주십시오.
    언제나 '모든 투자에 대한 책임은 본인에게 있습니다.'이라고 답변 끝에 적어주세요.
    대답할 때 상대방을 존중하고 예의를 갖추어 답변해 주세요.
    {context}
    질문: {question}
    최적의 답변:
    """
    return PromptTemplate.from_template(template)

def initialize_llm(openai_api_key):
    """ChatOpenAI 모델을 초기화하는 함수."""
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# Example usage
def main(username,password,host,port,database_schema,openai_api_key):
    db = url_to_db(username,password,host,port,database_schema)
    splits = split_text(db)
    retriever = create_vector_store(splits, openai_api_key)
    rag_prompt_custom = create_prompt_template()
    llm = initialize_llm(openai_api_key)
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
    return retriever, rag_chain

retriever, rag_chain = main(username, password, host, port, database_schema, openai_api_key)
def get_answer_from_rag_chain(question):
    response = rag_chain.invoke(question)
    return response.content if response else "답변을 생성할 수 없습니다."