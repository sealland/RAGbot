# src/ingest.py (Final Correct Version)

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <--- แก้ไข 1
from langchain_postgres.vectorstores import PGVector

# --- 1. ตั้งค่า Path และโหลด Environment Variables ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- 2. สร้าง Connection String ---
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
    print("เริ่มต้นกระบวนการ Ingest ข้อมูล...")
    print(f"กำลังค้นหาไฟล์ PDF ใน: {DATA_PATH}")
    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    documents = loader.load()

    if not documents:
        print("ไม่พบไฟล์ PDF ในโฟลเดอร์ data/ กรุณาตรวจสอบว่ามีไฟล์อยู่จริง")
        return

    print(f"โหลดเอกสาร PDF สำเร็จ จำนวน {len(documents)} ไฟล์")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"แบ่งเอกสารออกเป็น {len(docs)} ส่วน (chunks)")

    print("กำลังโหลด Embedding model...")
    # --- แก้ไข 2 ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("โหลด Embedding model สำเร็จ")

    print("กำลังสร้าง Vector Store และจัดเก็บข้อมูล...")

    # --- แก้ไข 3 (ใช้ 'embedding' และ 'connection') ---
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    print("จัดเก็บข้อมูลลงใน PostgreSQL (pgvector) สำเร็จ!")

if __name__ == "__main__":
    main()