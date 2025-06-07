import os
import psycopg  # เวอร์ชันใหม่ต้องการ import นี้เพื่อเช็ค driver
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_postgres.vectorstores import PGVector

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# สร้าง Connection String (ถูกต้องสำหรับเวอร์ชันใหม่)
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg",
    host=os.environ.get("POSTGRES_HOST"),
    port=int(os.environ.get("POSTGRES_PORT")),
    database=os.environ.get("POSTGRES_DB"),
    user=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
)

COLLECTION_NAME = "machine_repair_docs"


def main():
    # ... ส่วนโหลดและแบ่งเอกสารเหมือนเดิม ...
    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("กำลังสร้าง Vector Store และจัดเก็บข้อมูล...")

    # เวอร์ชันใหม่ใช้ `connection` และส่ง `Connection String` เข้าไปได้เลย
    # มันจะจัดการสร้าง Connection Object เองเบื้องหลัง
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,  # <--- **ใช้ 'connection'**
        pre_delete_collection=True,
    )

    print("จัดเก็บข้อมูลลงใน PostgreSQL (pgvector) สำเร็จ!")


if __name__ == "__main__":
    main()