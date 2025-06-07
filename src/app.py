# src/app.py (Final Correct Version)

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # <--- แก้ไข 1
from langchain_postgres.vectorstores import PGVector

# --- 1. โหลด Environment Variables และตั้งค่าพื้นฐาน ---
load_dotenv()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg",
    host=os.environ.get("POSTGRES_HOST"),
    port=int(os.environ.get("POSTGRES_PORT")),
    database=os.environ.get("POSTGRES_DB"),
    user=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
)
COLLECTION_NAME = "machine_repair_docs"

# --- 2. สร้าง Prompt Template ---
template = """
คุณคือผู้ช่วย AI เชี่ยวชาญด้านการซ่อมบำรุงเครื่องจักรของบริษัท
จงตอบคำถามโดยใช้บริบท (Context) ที่ให้มาเท่านั้น ห้ามเสริมข้อมูลหรือตอบจากความรู้อื่น
หากข้อมูลในบริบทไม่เพียงพอที่จะตอบคำถาม ให้ตอบว่า "ฉันไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในคู่มือครับ/ค่ะ"
จงตอบให้กระชับและเป็นขั้นตอนที่เข้าใจง่าย

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 3. ฟังก์ชันสำหรับโหลดโมเดลและ Retriever (ใช้ Cache เพื่อประสิทธิภาพ) ---
@st.cache_resource
def get_llm():
    """โหลดโมเดลภาษาจาก Google Gemini"""
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1 ,max_output_tokens=8192 )

@st.cache_resource
def get_retriever():
    """สร้าง Retriever เพื่อเชื่อมต่อและค้นหาข้อมูลจาก PGVector"""
    # --- แก้ไข 2 ---
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # สร้างการเชื่อมต่อไปยัง Vector Store ที่มีข้อมูลอยู่แล้ว
    # --- แก้ไข 3 (ใช้ 'embedding' และ 'connection') ---
    db = PGVector(
        embeddings=embeddings,  # <--- **แก้ไขเป็น `embeddings` (มี s)**
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )
    # สร้าง Retriever พื้นฐานก่อน
    base_retriever = db.as_retriever(search_kwargs={'k': 3})  # k ที่นี่อาจไม่ต้องเยอะมาก

    # ห่อด้วย MultiQueryRetriever
    # มันจะใช้ llm ที่เรามี (Gemini) เพื่อช่วยแตกคำถาม
    llm = get_llm()  # ดึง llm มาใช้
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    return multi_query_retriever

def format_docs(docs):
    """จัดรูปแบบเอกสารที่ค้นเจอให้อยู่ในรูปแบบสตริงเดียว"""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. สร้างหน้าเว็บด้วย Streamlit ---
st.set_page_config(page_title="🤖 ผู้ช่วยช่างซ่อม", layout="wide", initial_sidebar_state="collapsed")
st.title("🤖 ผู้ช่วยช่างซ่อมอัจฉริยะ")
st.markdown("พิมพ์คำถามเกี่ยวกับปัญหาเครื่องจักร แล้วผมจะค้นหาคำตอบจากคู่มือซ่อมบำรุงให้ครับ")

try:
    llm = get_llm()
    retriever = get_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("เครื่องจักร XYZ มีปัญหาอะไร?"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลในคู่มือและเรียบเรียงคำตอบ..."):
                response = rag_chain.invoke(user_question)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเริ่มต้นระบบ: {e}")
    st.error("กรุณาตรวจสอบการตั้งค่าในไฟล์ .env และตรวจสอบว่าได้รัน ingest.py สำเร็จแล้ว")