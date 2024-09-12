from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('host')
port = os.getenv('port')
username = os.getenv('username')
password = os.getenv('password')
database_schema = os.getenv('database_schema')
openai_api_key = os.getenv('OPENAI_API_KEY')

def url_to_db(username,password,host,port,database_schema):
    mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}?charset=utf8"
    db = SQLDatabase.from_uri(mysql_uri)
    db1=db.run("SELECT * FROM news WHERE DATE(time) = CURDATE();;")
    db2=db.run("SELECT * FROM naver_finance WHERE DATE(time) = CURDATE();")
    db3=db1+db2
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

    # OpenAI Embeddings 초기화
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Chroma 벡터 스토어 생성 및 저장 설정
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="./chroma_db"  # 저장될 디렉토리 지정
    )
    
    # 벡터 스토어를 검색 가능한 리트리버로 설정
    retriever = vectorstore.as_retriever()
    return retriever

def create_prompt_template():
    """곤란한 질문에도 적절한 답변을 제공하는 커스텀 프롬프트 템플릿을 생성하는 함수."""
    template = """
    주식이나 뉴스, 증시에 관한 내용이 아니면 잘 알 수 없는 내용이니 대답할 수 없다고 말하십시오.
    질문에 대해 가능한 한 최선의 답변을 제공하되, 정확하지 않거나 모호한 부분은 솔직히 인정하십시오.
    주식의 경우 관련 도메인이나 테마를 언급해줘.
    정답을 확신할 수 없는 경우 '이 부분은 정확히 알 수 없습니다.'라고 명확히 말하십시오.
    답변을 단계적으로 생각하고 필요한 순서를 정한 후, 그 순서에 따라 자세히 링크가 있으면 각각 다른 링크를 첨부해서 설명해 주십시오.
    언제나 '모든 투자에 대한 책임은 본인에게 있습니다.'이라고 답변 끝에 적어주세요.
    계산 문제는 반드시 단계별로 과정을 보여주십시오.
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