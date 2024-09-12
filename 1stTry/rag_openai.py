from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from sqlalchemy import create_engine


def url_to_db(username,password,host,port,database_schema):
    mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"
    db = SQLDatabase.from_uri(mysql_uri)
    engine = create_engine(mysql_uri)
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
    
    # 저장 명령 실행
    vectorstore.persist()  # 벡터 스토어 상태를 지정된 디렉토리에 저장
    
    # 벡터 스토어를 검색 가능한 리트리버로 설정
    retriever = vectorstore.as_retriever()
    return retriever

def create_prompt_template():
    """커스텀 프롬프트 템플릿을 생성하는 함수."""
    template = """
    다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
    만약 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마십시오.
    답변은 step by step으로 생각하고 질문 대답의 필요한 순서를 정하고 순서대로 말해주십시오.
    항상 '추가적인 질문 기다리는중...?'라고 답변 끝에 말해주시오.
    영화 대사를 이용해서 이해가 쉽게 답변해 주시오.
    계산 문제일 경우 그 과정을 step by step으로 알려주시오.
    대답을 할 때는 상대방을 높여주시오.{context}
    질문: {question}
    도움이 되는 답변:
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
