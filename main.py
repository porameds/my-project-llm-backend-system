import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# SQL Agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# LLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
#  1. ตั้งค่า Database และ LLM
# ==========================================
# สมมติว่าฐานข้อมูลเอกสาร และฐานข้อมูลตารางข้อมูล (SQL) อยู่ในก้อนเดียวกัน
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "all_company_docs"

LLM_MODEL_NAME = "qwen-3" 
LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg" 
LLM_BASE_URL = "http://localhost:4000/v1"

app = FastAPI(title="Company Super Agent API")

#  1.1 โหลด Embedding และ Vector Store รอไว้ (สำหรับเอกสาร)
print(" กำลังโหลด Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

#  1.2 โหลด LLM รอไว้
print(" กำลังเชื่อมต่อ LLM...")
llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0)

#  1.3 ตั้งค่า SQL Agent รอไว้ (สำหรับดึงข้อมูลตัวเลข/ตาราง)
print(" กำลังเชื่อมต่อ SQL Database และสร้าง SQL Agent...")
db = SQLDatabase.from_uri(CONNECTION_STRING)
sql_agent = create_sql_agent(
    llm=llm, 
    db=db, 
    agent_type="zero-shot-react-description", # ใช้ type นี้จะเข้ากับโมเดล Open-source ได้ดี
    verbose=True # เปิด True ไว้ดูวิธีคิดของ AI ใน Terminal
)

# ==========================================
#  2. โครงสร้างข้อมูล Pydantic
# ==========================================
class StructuredChatResponse(BaseModel):
    answer: str = Field(description="คำตอบที่ละเอียด ครอบคลุม รักษารูปแบบหัวข้อและ List (ตอบภาษาไทย)")
    sentiment: str = Field(description="อารมณ์ของประโยคคำถาม (Positive, Negative, Neutral)")
    confidence_score: float = Field(description="ความมั่นใจในคำตอบ 0.0 - 1.0")

structured_llm = llm.with_structured_output(StructuredChatResponse)

class ChatRequest(BaseModel):
    query: str
    department: Optional[str] = None # ถ้าส่งมาเป็น "dataInsights" จะไปเข้า SQL
    
# ==========================================
#  3. Endpoint หลัก (Super Agent)
# ==========================================
@app.post("/api/chat")
async def chat_with_company_bot(request: ChatRequest):
    try:
        print(f"\n [เริ่มทำงาน] คำถาม: '{request.query}' | แผนก: '{request.department}'")

        # ---  STEP 1: Routing สับรางจากข้อมูลที่ Frontend ส่งมา ---
        if request.department == "dataInsights":
            target_db = "SQL_DB"
        else:
            target_db = "VECTOR_DB"
            
        print(f"🚦 ระบบสับรางไปยัง: {target_db}")

        # ==========================================
        #  เส้นทางที่ 1: Data Insights -> ให้ SQL Agent ลุย!
        # ==========================================
        if target_db == "SQL_DB":
            print(" กำลังประมวลผลผ่าน SQL Agent...")
            
            # สั่งให้ SQL Agent ทำงาน (แปลงคำถามเป็น SQL -> ดึงข้อมูล -> สรุปเป็นคำตอบ)
            response = sql_agent.invoke({"input": request.query})
            
            # ดึงคำตอบสุดท้ายออกมา
            sql_answer = response.get("output", "ไม่สามารถดึงข้อมูลจากระบบฐานข้อมูลได้")
            
            return {
                "answer": sql_answer,
                "sentiment": "Neutral",
                "confidence_score": 0.95, # SQL มักจะแม่นยำสูง
                "sources": ["SQL Database (Data Insights)"]
            }

        # ==========================================
        #  เส้นทางที่ 2: แผนกอื่นๆ -> ค้นหาใน Vector DB
        # ==========================================
        else:
            print(" กำลังค้นหาในเอกสาร Vector DB...")
            
            search_kwargs = {"k": 5}
            # กรองข้อมูลตามแผนก (ถ้ามีการส่งชื่อแผนกมา และไม่ใช่ค่าว่าง)
            if request.department:
                search_kwargs["filter"] = {"department": request.department}
                
            results = vector_store.similarity_search_with_score(request.query, **search_kwargs)
            
            if not results:
                return {
                    "answer": f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามในเอกสารของแผนก {request.department or 'ทั้งหมด'}",
                    "sentiment": "Neutral",
                    "confidence_score": 0.0,
                    "sources": []
                }

            # รวบรวมข้อมูลอ้างอิง
            raw_context = ""
            source_files = []
            for doc, score in results:
                file_name = doc.metadata.get("source_file", "ไม่ระบุไฟล์")
                raw_context += f"[{file_name}] {doc.page_content}\n"
                if file_name not in source_files:
                    source_files.append(file_name)

            print(" กำลังส่งให้ LLM เรียบเรียงข้อมูลจากเอกสาร...")
            system_instruction = f"""คุณคือผู้ช่วยอัจฉริยะองค์กร จงตอบคำถามโดยอ้างอิงจากข้อมูลอ้างอิงต่อไปนี้เท่านั้น 
ข้อมูลอ้างอิง:
{raw_context}"""

            messages = [SystemMessage(content=system_instruction), HumanMessage(content=request.query)]
            response_object = structured_llm.invoke(messages)
            
            # แยกข้อมูลออกจาก Pydantic Object
            if hasattr(response_object, 'parsed') and response_object.parsed is not None:
                ans = response_object.parsed.answer
                sent = response_object.parsed.sentiment
                conf = response_object.parsed.confidence_score
            else:
                ans = getattr(response_object, 'answer', 'ไม่สามารถเรียบเรียงได้')
                sent = getattr(response_object, 'sentiment', 'Neutral')
                conf = getattr(response_object, 'confidence_score', 0.0)
            
            return {
                "answer": str(ans),
                "sentiment": str(sent),
                "confidence_score": float(conf),
                "sources": source_files
            }

    except Exception as e:
        print(f" Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)