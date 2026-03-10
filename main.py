from venv import logger

import uvicorn
import uuid
import json
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ==========================================
#  1. Imports ทั้งหมดที่จำเป็น
# ==========================================
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
#  2. ตั้งค่าการเชื่อมต่อ (Configurations)
# ==========================================
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@host.docker.internal/llm_db"
COLLECTION_NAME = "all_company_docs"

LLM_MODEL_NAME = "llama-3"
LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg" 
LLM_BASE_URL = "http://host.docker.internal:4000/v1"

app = FastAPI(title="Company Super Agent API")

#  Mount Static Files (จุดที่ใช้แสดงรูป)
app.mount("/static", StaticFiles(directory="/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env"), name="static")

# Endpoint สำหรับตรวจสอบไฟล์แบบเจาะลึก (เป็นเหมือน File Explorer)
@app.get("/check-static")
@app.get("/check-static/{subpath:path}")
async def check_static_files(subpath: str = ""):
    base_dir = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env"
    
    # รวม Path ที่ต้องการดู
    target_path = os.path.join(base_dir, subpath) if subpath else base_dir
    
    if not os.path.exists(target_path):
        return {"status": "error", "message": f"ไม่พบ Path นี้ในมุมมองของ Docker: {target_path}"}
        
    if os.path.isdir(target_path):
        items = os.listdir(target_path)
        return {
            "status": "success", 
            "type": "directory",
            "current_path": target_path,
            "total_items": len(items),
            "items_found": items # โชว์ทั้งหมดเลย
        }
    else:
        return {
            "status": "success",
            "type": "file",
            "file_size_bytes": os.path.getsize(target_path)
        }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้าง Engine สำหรับต่อ Database ทั่วไป
engine = create_engine(CONNECTION_STRING)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
#  3. โครงสร้างตาราง Cache ใน PostgreSQL
# ==========================================
class LlmPromptCache(Base):
    __tablename__ = "llm_prompt_cache"
    __table_args__ = {"schema": "public"} 
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model = Column(String, index=True)           
    input_model = Column(Text, index=True)       
    output_model = Column(Text)                  
    condition = Column(Text, nullable=True)      
    meta_data = Column("meta", JSONB, nullable=True) 
    expires_date = Column(DateTime)              
    created_at = Column(DateTime, default=datetime.utcnow) 

print(" กำลังตรวจสอบและสร้างตาราง Cache...")
Base.metadata.create_all(bind=engine)

# ==========================================
#  4. โหลด AI และ Tools รอไว้
# ==========================================
print(" กำลังโหลด Embedding และ Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store = PGVector(
    embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True
)

print(" กำลังเชื่อมต่อ LLM...")
llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0,max_tokens=8000 )

print(" กำลังเชื่อมต่อ SQL Database และสร้าง SQL Agent...")
db = SQLDatabase.from_uri(CONNECTION_STRING, include_tables=["machine_logs"])
sql_agent = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", verbose=True)

# ==========================================
#  5. โครงสร้างข้อมูล Pydantic
# ==========================================
# ==========================================
#  5. โครงสร้างข้อมูล Pydantic
# ==========================================
class StructuredChatResponse(BaseModel):
    answer: str = Field(description="คำตอบที่ละเอียดและครอบคลุม (ตอบภาษาไทยเท่านั้น ห้ามตอบภาษาจีน)")
    image_links: List[str] = Field(description="รายการ URL ของรูปภาพทั้งหมดที่พบในข้อมูลอ้างอิง ให้ดึงมาเฉพาะลิงก์ HTTP ข้างในวงเล็บ () เท่านั้น หากไม่มีให้ปล่อยว่าง", default=[])
    sentiment: str = Field(description="อารมณ์ของประโยคคำถาม (Positive, Negative, Neutral)")
    confidence_score: float = Field(description="ความมั่นใจในคำตอบ 0.0 - 1.0")

structured_llm = llm.with_structured_output(StructuredChatResponse)

structured_llm = llm.with_structured_output(StructuredChatResponse)

class ChatRequest(BaseModel):
    query: str
    department: Optional[str] = None

# ==========================================
#  6. Endpoint หลัก (Super Agent API)
# ==========================================
@app.post("/api/chat")
async def chat_with_company_bot(request: ChatRequest):
    db_session = SessionLocal()
    try:
        now = datetime.utcnow()
        cache_condition = request.department if request.department else "ALL_DEPARTMENTS"

        cached_record = db_session.query(LlmPromptCache).filter(
            LlmPromptCache.model == LLM_MODEL_NAME,
            LlmPromptCache.input_model == request.query,
            LlmPromptCache.condition == cache_condition,
            LlmPromptCache.expires_date > now  
        ).first()

        if cached_record:
            # logger.info(f"\n [CACHE HIT] โหลดจาก Cache ข้ามการค้นหาใหม่ -> คำถาม: '{request.query}'", flush=True)
            return json.loads(cached_record.output_model)

        target_db = "SQL_DB" if request.department == "dataInsights" else "VECTOR_DB"
        final_response_dict = {}

        if target_db == "SQL_DB":
            # logger.info(f" [ROUTE] วิ่งเข้า SQL Database (Data Insights)", flush=True)
            smart_prompt = f"คำถามจากผู้ใช้: {request.query}"
            response = sql_agent.invoke({"input": smart_prompt})
            sql_answer = response.get("output", "ไม่สามารถดึงข้อมูลจากระบบฐานข้อมูลได้")
            
            final_response_dict = {
                "answer": sql_answer,
                "sentiment": "Neutral",
                "confidence_score": 0.95,
                "sources": ["SQL Database (Data Insights)"]
            }

        else:
            # logger.info(f"[ROUTE] วิ่งเข้า Vector Database (เอกสารความรู้)", flush=True)
            search_kwargs = {"k": 5}
            if request.department and request.department != "dataInsights":
                search_kwargs["filter"] = {"department": request.department}
                
            results = vector_store.similarity_search_with_score(request.query, **search_kwargs)
            
            if not results:
                # logger.info(f" [VECTOR DB] หาไม่เจอ! Vector Store ไม่พบเนื้อหาที่ตรงกับคำถาม", flush=True)
                final_response_dict = {
                    "answer": f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามในเอกสาร",
                    "sentiment": "Neutral",
                    "confidence_score": 0.0,
                    "sources": []
                }
            else:
                # logger.info(f" [VECTOR DB] ค้นพบ {len(results)} รายการที่เกี่ยวข้อง", flush=True)
                raw_context = ""
                source_files = []
                for doc, score in results:
                    file_name = doc.metadata.get("source_file", "ไม่ระบุไฟล์")
                    raw_context += f"[{file_name}] {doc.page_content}\n"
                    if file_name not in source_files:
                        source_files.append(file_name)
                print("\n" + "="*50)
                print("="*50, flush=True)
                print(raw_context, flush=True)
                print("="*50 + "\n", flush=True)
                system_instruction = f"""คุณคือผู้ช่วยอัจฉริยะโรงงาน จงตอบคำถามโดยอ้างอิงจากข้อมูลต่อไปนี้เท่านั้น
กฎกติกาขั้นเด็ดขาด (CRITICAL RULES):
1. ไม่ว่าผู้ใช้จะพิมพ์มาแค่คีย์เวิร์ดสั้นๆ คุณต้อง "อธิบายรายละเอียดทั้งหมด" ที่เกี่ยวข้องให้ครบถ้วน ห้ามตอบกลับแค่ชื่อหัวข้อเด็ดขาด
2. สำคัญมาก: หากพบแท็กรูปภาพ เช่น ![](http://...) ในข้อมูลอ้างอิง ให้สกัดเฉพาะ URL (http://...) ออกมาใส่ในช่อง image_links แยกไว้ต่างหาก ห้ามเอาไปปนในข้อความ
3. บังคับสูงสุด: ต้องตอบกลับเป็น "ภาษาไทย" เท่านั้น! ห้ามแปลหรือตอบเป็นภาษาจีน
4. ดึงข้อมูลมาตอบให้ครบถ้วน 100% โดยเฉพาะส่วนที่เป็นรายการข้อๆ (Bullet points) หรือมีเครื่องหมายขีด (-) 
5. ห้ามสรุปรวบรัด ห้ามเขียนรวบเป็นย่อหน้าเดียว และห้ามตัดเนื้อหาทิ้งเด็ดขาด 
6. ต้องตอบกลับโดยรักษารูปแบบข้อๆ ไว้ตามต้นฉบับ
ข้อมูลอ้างอิง:
{raw_context}"""
                
                messages = [SystemMessage(content=system_instruction), HumanMessage(content=request.query)]
                response_object = structured_llm.invoke(messages)
                
                if hasattr(response_object, 'parsed') and response_object.parsed is not None:
                    ans = response_object.parsed.answer
                    img_list = response_object.parsed.image_links # ดึง list รูปภาพ
                    sent = response_object.parsed.sentiment
                    conf = response_object.parsed.confidence_score
                else:
                    ans = getattr(response_object, 'answer', 'ไม่สามารถเรียบเรียงได้')
                    img_list = getattr(response_object, 'image_links', [])
                    sent = getattr(response_object, 'sentiment', 'Neutral')
                    conf = getattr(response_object, 'confidence_score', 0.0)
                
                final_response_dict = {
                    "answer": str(ans),
                    "images": img_list,
                    "sentiment": str(sent),
                    "confidence_score": float(conf),
                    "sources": source_files
                }

        new_cache = LlmPromptCache(
            model=LLM_MODEL_NAME,
            input_model=request.query,
            output_model=json.dumps(final_response_dict, ensure_ascii=False), 
            condition=cache_condition,
            meta_data={"department_requested": request.department, "routed_to": target_db},
            expires_date=now + timedelta(days=7) 
        )
        db_session.add(new_cache)
        db_session.commit() 

        return final_response_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# import uvicorn
# import uuid
# import json
# from datetime import datetime, timedelta
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import os
# # ==========================================
# #  1. Imports ทั้งหมดที่จำเป็น
# # ==========================================
# # Database & Cache (SQLAlchemy)
# from sqlalchemy import create_engine, Column, String, Text, DateTime
# from sqlalchemy.dialects.postgresql import UUID, JSONB
# from sqlalchemy.orm import declarative_base, sessionmaker

# # Vector DB
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector

# # SQL Agent
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import create_sql_agent

# # LLM
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage

# # ==========================================
# #  2. ตั้งค่าการเชื่อมต่อ (Configurations)
# # ==========================================
# # CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
# CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@host.docker.internal/llm_db"
# COLLECTION_NAME = "all_company_docs"

# LLM_MODEL_NAME = "qwen-3" 
# LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg" 
# # LLM_BASE_URL = "http://localhost:4000/v1"
# LLM_BASE_URL = "http://host.docker.internal:4000/v1"



# app = FastAPI(title="Company Super Agent API")

# app.mount("/static", StaticFiles(directory="/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env"), name="static")


# # 2. เพิ่ม Endpoint สำหรับตรวจสอบไฟล์
# @app.get("/check-static")
# async def check_static_files():
#     # ระบุ Path เดียวกับที่ใส่ใน app.mount ด้านบน
#     base_dir = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env"
    
#     if not os.path.exists(base_dir):
#         return {"status": "error", "message": f"ไม่พบโฟลเดอร์: {base_dir}"}
        
#     # ลองลิสต์รายชื่อโฟลเดอร์ย่อย/ไฟล์ ที่อยู่ข้างในมาดู
#     items = os.listdir(base_dir)
#     return {
#         "status": "success", 
#         "message": "มองเห็นโฟลเดอร์ Static แล้ว!",
#         "items_found": items[:10]  # โชว์แค่ 10 อันดับแรกกันรก
#     }

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # อนุญาตทุกโดเมน (ถ้าขึ้นระบบจริงค่อยเปลี่ยนเป็น URL ของเว็บ Frontend)
#     allow_credentials=True,
#     allow_methods=["*"], # อนุญาตทุก Method (GET, POST, OPTIONS, ฯลฯ)
#     allow_headers=["*"], # อนุญาตทุก Header
# )



# # สร้าง Engine สำหรับต่อ Database ทั่วไป
# engine = create_engine(CONNECTION_STRING)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # ==========================================
# #  3. โครงสร้างตาราง Cache ใน PostgreSQL
# # ==========================================
# class LlmPromptCache(Base):
#     __tablename__ = "llm_prompt_cache"
#     __table_args__ = {"schema": "public"} 
#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     model = Column(String, index=True)           
#     input_model = Column(Text, index=True)       
#     output_model = Column(Text)                  
#     condition = Column(Text, nullable=True)      
#     meta_data = Column("meta", JSONB, nullable=True) 
#     expires_date = Column(DateTime)              
#     created_at = Column(DateTime, default=datetime.utcnow) 

# # สร้างตาราง Cache ถ้ายังไม่มี
# print(" กำลังตรวจสอบและสร้างตาราง Cache...")
# Base.metadata.create_all(bind=engine)

# # ==========================================
# #  4. โหลด AI และ Tools รอไว้
# # ==========================================
# print(" กำลังโหลด Embedding และ Vector Store...")
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# vector_store = PGVector(
#     embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True
# )

# print(" กำลังเชื่อมต่อ LLM...")
# llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0)

# print(" กำลังเชื่อมต่อ SQL Database และสร้าง SQL Agent...")
# db = SQLDatabase.from_uri(CONNECTION_STRING, include_tables=["machine_logs"])
# sql_agent = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", verbose=True)

# # ==========================================
# #  5. โครงสร้างข้อมูล Pydantic
# # ==========================================
# class StructuredChatResponse(BaseModel):
#     answer: str = Field(description="คำตอบที่ละเอียด ครอบคลุม รักษารูปแบบหัวข้อและ List (ตอบภาษาไทย)")
#     sentiment: str = Field(description="อารมณ์ของประโยคคำถาม (Positive, Negative, Neutral)")
#     confidence_score: float = Field(description="ความมั่นใจในคำตอบ 0.0 - 1.0")

# structured_llm = llm.with_structured_output(StructuredChatResponse)

# class ChatRequest(BaseModel):
#     query: str
#     department: Optional[str] = None # ถ้าส่งมาเป็น "dataInsights" จะไปเข้า SQL

# # ==========================================
# #  6. Endpoint หลัก (Super Agent API)
# # ==========================================
# @app.post("/api/chat")
# async def chat_with_company_bot(request: ChatRequest):
#     db_session = SessionLocal()
#     try:
#         now = datetime.utcnow()
#         print(f"\n [เริ่มทำงาน] คำถาม: '{request.query}' | แผนก/โหมด: '{request.department}'")

#         # กำหนด Condition ของ Cache (เพื่อไม่ให้คำถามเดียวกันแต่คนละแผนกมาปนกัน)
#         cache_condition = request.department if request.department else "ALL_DEPARTMENTS"

#         # ---  STEP A: ตรวจสอบ Cache ---
#         cached_record = db_session.query(LlmPromptCache).filter(
#             LlmPromptCache.input_model == request.query,
#             LlmPromptCache.condition == cache_condition,
#             LlmPromptCache.expires_date > now  
#         ).first()

#         if cached_record:
#             print(" [Cache Hit] ดึงคำตอบจากความจำเดิม ")
#             return json.loads(cached_record.output_model)

#         print(" [Cache Miss] ไม่พบในความจำ เริ่มกระบวนการค้นหาและคิดใหม่...")

#         # ---  STEP B: Routing สับราง ---
#         target_db = "SQL_DB" if request.department == "dataInsights" else "VECTOR_DB"
#         print(f" ระบบสับรางไปยัง: {target_db}")

#         final_response_dict = {}

#         # ==========================================
#         #  เส้นทางที่ 1: SQL Agent (Data Insights)
#         # ==========================================
#         if target_db == "SQL_DB":
#             print(" กำลังประมวลผลผ่าน SQL Agent...")
#             smart_prompt = f"""
# คำถามจากผู้ใช้: {request.query}
# """
#             response = sql_agent.invoke({"input": smart_prompt})
#             sql_answer = response.get("output", "ไม่สามารถดึงข้อมูลจากระบบฐานข้อมูลได้")
            
#             final_response_dict = {
#                 "answer": sql_answer,
#                 "sentiment": "Neutral",
#                 "confidence_score": 0.95,
#                 "sources": ["SQL Database (Data Insights)"]
#             }

#         # ==========================================
#         #  เส้นทางที่ 2: Vector DB (เอกสารทั่วไป)
#         # ==========================================
#         else:
#             print(" กำลังค้นหาในเอกสาร Vector DB...")
#             search_kwargs = {"k": 10}
#             if request.department and request.department != "dataInsights":
#                 search_kwargs["filter"] = {"department": request.department}
                
#             results = vector_store.similarity_search_with_score(request.query, **search_kwargs)
            
#             if not results:
#                 final_response_dict = {
#                     "answer": f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามในเอกสาร",
#                     "sentiment": "Neutral",
#                     "confidence_score": 0.0,
#                     "sources": []
#                 }
#             else:
#                 raw_context = ""
#                 print(f"\n--- [DEBUG SCORE สำหรับคำถาม: {request.query}] ---")
#                 source_files = []
#                 for doc, score in results:
#                     file_name = doc.metadata.get("source_file", "ไม่ระบุไฟล์")
#                     raw_context += f"[{file_name}] {doc.page_content}\n"
#                     if file_name not in source_files:
#                         source_files.append(file_name)
#                 print(f"\n🔍 [DEBUG] ข้อมูลเบื้องหลังที่ AI ได้รับ:\n{raw_context}\n{'-'*50}")
#                 system_instruction = f"""คุณคือผู้ช่วยอัจฉริยะโรงงาน จงตอบคำถามโดยอ้างอิงจากข้อมูลต่อไปนี้เท่านั้น
# กฎกติกาขั้นเด็ดขาด:
# 1. ดึงข้อมูลมาตอบให้ครบถ้วน 100% โดยเฉพาะส่วนที่เป็นรายการข้อๆ (Bullet points) หรือมีเครื่องหมายขีด (-) 
# 2. ห้ามสรุปรวบรัด ห้ามเขียนรวบเป็นย่อหน้าเดียว และห้ามตัดเนื้อหาทิ้งเด็ดขาด 
# 3. ต้องตอบกลับโดยรักษารูปแบบข้อๆ ไว้ตามต้นฉบับ

# ข้อมูลอ้างอิง:
# {raw_context}"""
                
#                 messages = [SystemMessage(content=system_instruction), HumanMessage(content=request.query)]
#                 response_object = structured_llm.invoke(messages)
                
#                 if hasattr(response_object, 'parsed') and response_object.parsed is not None:
#                     ans = response_object.parsed.answer
#                     sent = response_object.parsed.sentiment
#                     conf = response_object.parsed.confidence_score
#                 else:
#                     ans = getattr(response_object, 'answer', 'ไม่สามารถเรียบเรียงได้')
#                     sent = getattr(response_object, 'sentiment', 'Neutral')
#                     conf = getattr(response_object, 'confidence_score', 0.0)
                
#                 final_response_dict = {
#                     "answer": str(ans),
#                     "sentiment": str(sent),
#                     "confidence_score": float(conf),
#                     "sources": source_files
#                 }

#         # ---  STEP C: บันทึกความจำลง Cache ก่อนตอบกลับ ---
#         print(" กำลังบันทึกคำตอบลง Cache...")
#         new_cache = LlmPromptCache(
#             model=LLM_MODEL_NAME,
#             input_model=request.query,
#             output_model=json.dumps(final_response_dict, ensure_ascii=False), 
#             condition=cache_condition,
#             meta_data={"department_requested": request.department, "routed_to": target_db},
#             expires_date=now + timedelta(days=7) 
#         )
#         db_session.add(new_cache)
#         db_session.commit() 

#         return final_response_dict

#     except Exception as e:
#         print(f" Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db_session.close() # ปิด Session ทุกครั้งเพื่อคืน Connection ให้ Database

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)