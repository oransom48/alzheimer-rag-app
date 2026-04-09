import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Alzheimer's Care AI", page_icon="🧠", layout="centered")

# โหลด API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. ฟังก์ชันโหลดโมเดลและฐานข้อมูล (ใช้ @st.cache_resource เพื่อไม่ให้มันโหลดซ้ำทุกครั้งที่พิมพ์แชท)
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
    
    DB_PATH = "ChromaDB_BGE_M3"
    retriever = None
    if os.path.exists(DB_PATH):
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    return llm, retriever

# เรียกใช้ฟังก์ชันโหลดระบบ
llm, retriever = load_system()

# 3. สร้างเมนู Sidebar สำหรับสลับโหมด
st.sidebar.title("⚙️ ตั้งค่าระบบ")
mode = st.sidebar.radio(
    "เลือกโหมดการทำงานของ AI:",
    ["🧠 RAG System (แม่นยำสูง)", "💬 Vanilla LLM (ความรู้ทั่วไป)"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** ลองถามคำถามเดียวกันสลับโหมดดู เพื่อเปรียบเทียบความถูกต้องของคำตอบครับ")

# 4. ส่วนหัวของแชท
st.title("🧠 ผู้ช่วยดูแลผู้ป่วยโรคอัลไซเมอร์")
st.caption(f"ปัจจุบันใช้งานโหมด: **{mode}**")

# 5. ระบบจำประวัติการแชท (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงข้อความแชทเก่าๆ บนหน้าจอ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. รับข้อความจาก User และประมวลผล
if prompt := st.chat_input("เช่น คุณแม่ลืมทานยาควรทำอย่างไรดีคะ?"):
    
    # แสดงข้อความของ User
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ส่วนของ AI ประมวลผลและตอบกลับ
    with st.chat_message("assistant"):
        # แสดงลูกข่างหมุนๆ ตอนกำลังคิด
        with st.spinner("AI กำลังคิด..."):
            
            if "RAG System" in mode:
                # --- โหมด RAG ---
                if retriever:
                    docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    rag_prompt = f"""You are a specialized assistant for Alzheimer's caregivers in Thailand.
                    Use the following pieces of context to answer the question at the end.

                    RULES:
                    1. Answer ONLY based on the context provided.
                    2. If the answer is not in the context, you MUST say exactly:
                    "ฉันไม่ทราบข้อมูลนี้ค่ะเนื่องจากไม่มีในเอกสารอ้างอิง" (I don't know because it's not in the reference).
                    3. Do not make up answers.
                    4. Answer in helpful, polite Thai language.

                    Context:
                    {context_text}

                    Question: {prompt}
                    """
                    
                    response = llm.invoke(rag_prompt)
                    answer = response.content

                    reference_text = "\n\n---\n**📚 แหล่งข้อมูลอ้างอิง:**\n"
                    
                    # สร้าง set ว่างๆ ขึ้นมาเพื่อเก็บชื่อเอกสารแบบไม่ซ้ำ
                    unique_sources = set()
                    
                    # วนลูปเก็บชื่อไฟล์เข้า set (ถ้าชื่อซ้ำ set จะตัดทิ้งให้เอง)
                    for doc in docs:
                        source = doc.metadata.get("source", "ไม่ทราบแหล่งที่มา")
                        unique_sources.add(source)
                    
                    # นำชื่อเอกสารที่กรองแล้วมาจัดรูปแบบแสดงผล
                    for source in unique_sources:
                        # เช็กว่าข้อมูลใน source เป็น URL ของเว็บไซต์หรือไม่
                        if source.startswith("http://") or source.startswith("https://"):
                            # ถ้าเป็นลิงก์เว็บ ให้ทำเป็น Hyperlink คลิกได้
                            reference_text += f"- 🔗 [{source}]({source})\n"
                        else:
                            # ถ้าเป็นชื่อไฟล์ธรรมดา (ไม่ได้มาจากเว็บ) ให้แสดงเป็นข้อความปกติ
                            reference_text += f"- 📄 `{source}`\n"
                    
                    # 6. เอาคำตอบของ AI มาเย็บติดกับแหล่งอ้างอิง
                    answer = answer + reference_text

                else:
                    answer = "ขออภัยค่ะ ระบบฐานข้อมูล RAG ขัดข้อง (หาโฟลเดอร์ ChromaDB ไม่เจอ)"
            else:
                # --- โหมด Vanilla ---
                vanilla_prompt = f"""คุณคือผู้เชี่ยวชาญด้านการดูแลผู้ป่วยอัลไซเมอร์ในประเทศไทย
                กรุณาตอบคำถามต่อไปนี้ด้วยความรู้พื้นฐานที่คุณมีให้ดีที่สุด โดยตอบให้กระชับและตรงประเด็น

                Question: {prompt}
                """
                response = llm.invoke(vanilla_prompt)
                answer = response.content

        # แสดงคำตอบและเซฟลงประวัติ
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})