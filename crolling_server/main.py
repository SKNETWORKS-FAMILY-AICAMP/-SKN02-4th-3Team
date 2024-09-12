from fastapi import FastAPI
import time
import requests
from bs4 import BeautifulSoup as bs
import MySQLdb
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import urljoin
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

# .env 파일에서 환경 변수 로드
load_dotenv()


# 환경 변수 가져오기
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
DB_CHARSET = os.getenv('DB_CHARSET')

OPENAI_API_KEY = os.getenv('OPEN_API_KEY')

# 데이터베이스 연결


def db_connect():
    conn = MySQLdb.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset=DB_CHARSET
    )
    return conn


def init_db():
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS naver_finance (
                seq INT AUTO_INCREMENT PRIMARY KEY,
                time DATETIME NOT NULL,
                txt TEXT NOT NULL
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except MySQLdb.Error as e:
        print(f"Error: {e}")


# 크롤링 함수
def finance_crawl():
    # 웹 페이지 URL
    url = 'https://finance.naver.com/'

    # 웹 페이지 요청
    response = requests.get(url)
    response.encoding = 'euc-kr'  # 네이버 금융 페이지는 euc-kr 인코딩을 사용합니다.

    # HTML 소스 파싱
    soup = bs(response.text, 'html.parser')

    # 모든 텍스트 추출
    text = soup.get_text()

    # 불필요한 공백 제거
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)

    # 데이터베이스에 저장
    try:
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO naver_finance (time, txt)
            VALUES (NOW(), %s)
        ''', (cleaned_text,))
        conn.commit()
        cursor.close()
        conn.close()
        print("INFO: finance DB 저장 완료")
    except MySQLdb.Error as e:
        print(f"Error: {e}")
    return cleaned_text


def main_url(url):
    main_address = []
    today = datetime.now().strftime('%Y%m%d')
    response = requests.get(url)
    html = response.content
    soup = bs(html, 'html.parser')
    Page = soup.select_one('td.pgRR > a')
    end_page = Page.attrs['href'].split("=")[-1]
    for i in range(int(end_page)+1, 0, -1):
        main_link = url+f'date={today}'+'&page='+str(i)
        main_address.append(main_link)
    return main_address


def news_crawl():
    url = 'https://finance.naver.com/news/mainnews.naver?'
    main_address = main_url(url)
    news = []
    for address in main_address:
        response = requests.get(address)
        file = response.content
        soup1 = bs(file, 'html.parser')
        Info1 = soup1.select('dd.articleSubject')
        Info2 = soup1.select('dd.articleSummary')
        for idx, subject in enumerate(Info1):
            a_tag = subject.select_one('a')
            if a_tag is not None:
                T_url = urljoin(address, a_tag["href"])  # 상대 경로를 절대 경로로 변환
                title = a_tag.text.strip()
                if idx < len(Info2):
                    summary_dd = Info2[idx]
                    summary_text = summary_dd.get_text(strip=True)
                    # if '#.' in summary_text else 'No text available'
                    summary = summary_text.split('#.')[0].strip()
                else:
                    summary = 'No summary available'

                news.append(f"('{title}', '{summary}', '{T_url}')")
                # 데이터베이스에 저장
                try:
                    conn = db_connect()
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT IGNORE INTO news (time, subject, content, link)
                        VALUES (NOW(), %s, %s, %s)
                    ''', (title, summary, T_url))
                    conn.commit()
                    cursor.close()
                    conn.close()
                except MySQLdb.Error as e:
                    print(f"Error: {e}")
    print("INFO: news DB 저장 완료")
    return ", ".join(news)


def split_text(db, chunk_size=500, chunk_overlap=0):
    """문서 내용을 작은 chunk 단위로 나누는 함수."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        persist_directory="../chormadb/chroma_db"  # 저장될 디렉토리 지정
    )

    # 저장 명령 실행
    vectorstore.persist()  # 벡터 스토어 상태를 지정된 디렉토리에 저장


def start_crawling():
    while True:
        finance = finance_crawl()
        news = news_crawl()
        splits = split_text(finance+news)
        create_vector_store(splits, OPENAI_API_KEY)

        time.sleep(600)  # 10분 (600초) 대기


@asynccontextmanager
async def lifespan(app: FastAPI):
    # When service starts.
    start_crawling()

    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI web crawler"}
