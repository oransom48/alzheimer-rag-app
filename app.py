import os
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# โหลด API Key (จาก .env ในเครื่อง หรือ Secrets บน Cloud)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. ตั้งค่า Model และ Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

# 2. เชื่อมต่อกับฐานข้อมูล ChromaDB (ระบุ Path ที่คุณเก็บโฟลเดอร์ไว้)
# หากรันบน Hugging Face อย่าลืมอัปโหลดโฟลเดอร์ฐานข้อมูลไปด้วยนะครับ
DB_PATH = "ChromaDB_BGE_M3"

# เตรียมส่วนประกอบสำหรับโหมด RAG
if os.path.exists(DB_PATH):
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
else:
    print("⚠️ Warning: ChromaDB folder not found!")

# 3. กำหนด Chat Profiles ให้ User เลือกก่อนเริ่มแชท
@cl.set_chat_profiles
async def set_chat_profile():
    return [
        cl.ChatProfile(
            name="RAG System",
            markdown_description="ถามตอบข้อมูลโดยใช้ฐานข้อมูลคู่มือผู้ป่วยอัลไซเมอร์ (แม่นยำสูง)",
            icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        ),
        cl.ChatProfile(
            name="Vanilla LLM",
            markdown_description="ถามตอบโดยใช้ความรู้ทั่วไปของ AI (อาจมีความผิดพลาดเชิงเทคนิค)",
            icon="https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
        ),
    ]

@cl.on_chat_start
async def start():
    # เก็บข้อมูลโหมดที่เลือกไว้ใน Session
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(content=f"สวัสดีค่ะ! คุณกำลังคุยกับระบบในโหมด **{chat_profile}** มีอะไรให้ช่วยไหมคะ?").send()

@cl.on_message
async def main(message: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    
    if chat_profile == "RAG System":
        # --- โหมด RAG: ค้นหาข้อมูลก่อนตอบ ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        res = await qa_chain.acall(message.content)
        answer = res["result"]
    else:
        # --- โหมด Vanilla: ตอบตรงๆ จาก AI ---
        res = await llm.ainvoke(message.content)
        answer = res.content

    await cl.Message(content=answer).send()