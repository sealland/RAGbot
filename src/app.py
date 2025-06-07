# src/app.py (Final Correct Version with Source Citation)

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel  # ### <<< แก้ไขตรงนี้ 1: Import เพิ่ม
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
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
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1, max_output_tokens=8192)


@st.cache_resource
def get_retriever():
    """สร้าง Retriever เพื่อเชื่อมต่อและค้นหาข้อมูลจาก PGVector"""
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    db = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )
    base_retriever = db.as_retriever(search_kwargs={'k': 3})
    llm = get_llm()
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    return multi_query_retriever


def format_docs(docs):
    """จัดรูปแบบเอกสารที่ค้นเจอให้อยู่ในรูปแบบสตริงเดียว"""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 4. สร้างหน้าเว็บด้วย Streamlit ---
st.set_page_config(page_title="🤖 AI Chat", layout="wide", initial_sidebar_state="collapsed")
st.title("🤖 ผู้ช่วยอัจฉริยะ")
st.markdown("พิมพ์คำถามแล้วผมจะค้นหาคำตอบจากคู่มือให้ครับ")

try:
    llm = get_llm()
    retriever = get_retriever()

    # ### <<< แก้ไขตรงนี้ 2: ปรับปรุง RAG Chain ทั้งหมด ---
    # Chain เดิม:
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # Chain ใหม่ที่ส่งคืน Sources:
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    rag_chain = (
            setup_and_retrieval
            | {
                "answer": (
                        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
                        | prompt
                        | llm
                        | StrOutputParser()
                ),
                "sources": lambda x: x["context"]
            }
    )
    # ---------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # แสดง sources ถ้ามี
            if "sources" in message and message["sources"]:
                with st.expander("แหล่งข้อมูลอ้างอิง"):
                    for source in message["sources"]:
                        # ตรวจสอบว่า metadata และ source key มีอยู่จริง
                        if hasattr(source, 'metadata') and 'source' in source.metadata:
                            st.info(f"**จากไฟล์:** {os.path.basename(source.metadata['source'])}")
                            # ตรวจสอบว่ามี page number หรือไม่
                            if 'page' in source.metadata:
                                st.text(f"หน้า: {source.metadata['page'] + 1}")  # Page number มักจะเริ่มที่ 0
                            st.caption(source.page_content)

    if user_question := st.chat_input("เครื่องจักร XYZ มีปัญหาอะไร?"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลในคู่มือและเรียบเรียงคำตอบ..."):
                # ### <<< แก้ไขตรงนี้ 3: วิธีเรียกใช้และจัดการผลลัพธ์
                response_data = rag_chain.invoke(user_question)

                response_answer = response_data["answer"]
                response_sources = response_data["sources"]

                st.markdown(response_answer)

                # แสดงแหล่งข้อมูลอ้างอิงใต้คำตอบ
                if response_sources:
                    with st.expander("แหล่งข้อมูลอ้างอิง"):
                        for source in response_sources:
                            # ตรวจสอบว่า metadata และ source key มีอยู่จริง
                            if hasattr(source, 'metadata') and 'source' in source.metadata:
                                st.info(f"**จากไฟล์:** {os.path.basename(source.metadata['source'])}")
                                # ตรวจสอบว่ามี page number หรือไม่
                                if 'page' in source.metadata:
                                    st.text(f"หน้า: {source.metadata['page'] + 1}")  # Page number มักจะเริ่มที่ 0
                                st.caption(source.page_content)

        # ### <<< แก้ไขตรงนี้ 4: บันทึกข้อมูลทั้งหมดลง session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_answer,
            "sources": response_sources  # เก็บ sources ไว้ด้วย
        })


except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเริ่มต้นระบบ: {e}")
    st.error("กรุณาตรวจสอบการตั้งค่าในไฟล์ .env และตรวจสอบว่าได้รัน ingest.py สำเร็จแล้ว")