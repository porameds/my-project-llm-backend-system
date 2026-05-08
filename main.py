from venv import logger

import uvicorn
import pandas as pd
import uuid
import json
import os
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from api_img_list_for_user_delete import router as image_router
from api_img_list_for_user_delete import router as pdfs_router
from QA_api_upload import router as qa_faca
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
from langchain_core.prompts import PromptTemplate

# ==========================================
#  2. ตั้งค่าการเชื่อมต่อ (Configurations)
# ==========================================
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@host.docker.internal/llm_db"
COLLECTION_NAME = "all_company_docs"

LLM_MODEL_NAME = "qwen-3"
LLM_API_KEY = "sk-laW9Aq8Xv3MH85OSRZHhMQ"
# LLM_MODEL_NAME = "llama-3"
# LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg" 
LLM_BASE_URL = "http://host.docker.internal:4000/v1"

app = FastAPI(
    title="Company Super Agent API",
    docs_url="/docs",        
    redoc_url="/redoc",     
    openapi_url="/openapi.json"  # บังคับสร้างไฟล์ Schema
)

#  Mount Static Files 

# 1. ให้เช็ค Path ที่เจาะจงก่อน (/static/pdfs) และชี้ไปที่โฟลเดอร์ QA_FACA_pdf
app.mount(
    "/static/pdfs", 
    StaticFiles(directory="/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_pdf"), 
    name="pdfs"
)

# 2. สำหรับไฟล์อื่นๆ
app.mount(
    "/static", 
    StaticFiles(directory="/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env"), 
    name="static"
)

# Endpoint สำหรับตรวจสอบไฟล์

@app.get("/check-static")
@app.get("/check-static/{subpath:path}")
async def check_static_files(subpath: str = ""):
    # base_dir = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/folder_for_img_path_url"
    base_pdf_dir = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_pdf"
    # รวม Path ที่ต้องการดู
    target_path = os.path.join(base_pdf_dir, subpath) if subpath else base_pdf_dir
    
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
app.include_router(image_router)
app.include_router(pdfs_router)
app.include_router(qa_faca)
# สร้าง Engine สำหรับต่อ Database ทั่วไป
engine = create_engine(CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300)
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
# เพิ่มตารางสำหรับเก็บ Suggested Prompts
# ==========================================
class SuggestedPrompt(Base):
    __tablename__ = "suggested_prompts"
    __table_args__ = {"schema": "public"}
    id = Column(uuid.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    department = Column(String, index=True)  # แผนกที่ปุ่มนี้จะไปปรากฎ เช่น QA_FACA
    prompt_text = Column(Text, nullable=False) # ข้อความบนปุ่ม
    is_active = Column(DateTime, default=datetime.utcnow) # เอาไว้เปิด/ปิดการใช้งาน (ในที่นี้ใช้ timestamp เช็ค)
    priority = Column(Integer, default=0) # ลำดับการเรียง (เลขน้อยขึ้นก่อน)

# ==========================================
#  4. โหลด AI และ Tools รอไว้
# ==========================================
print(" กำลังโหลด Embedding และ Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store = PGVector(
    embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True
)

print(" กำลังเชื่อมต่อ LLM...")
llm = ChatOpenAI(
    model=LLM_MODEL_NAME, 
    api_key=LLM_API_KEY, 
    base_url=LLM_BASE_URL, 
    temperature=0.1, 
    max_tokens=8000, 
    model_kwargs={
        "stop": ["Observation:", "\nObservation:", "Observation"],
        # "presence_penalty": 1.0,   # ปรับเพิ่มเป็น 1.0 เพื่อให้ AI พยายามพูดเรื่องใหม่ๆ
        # "frequency_penalty": 1.0   # ใช้ตัวนี้แทน repetition_penalty เพื่อป้องกันการพิมพ์คำเดิมซ้ำๆ
    }
)

print(" กำลังเชื่อมต่อ SQL Database และสร้าง SQL Agent...")
db = SQLDatabase.from_uri(CONNECTION_STRING, include_tables=["machine_logs"])

PREFIX = """คุณคือ AI ผู้เชี่ยวชาญด้านฐานข้อมูลโรงงาน จงทำงานตามรูปแบบด้านล่างนี้อย่างเคร่งครัด
ห้ามกล่าวทักทาย ห้ามอธิบายตัวเอง ห้ามขอโทษ และ ห้ามแต่งข้อมูลขึ้นมาเองเด็ดขาด

รูปแบบการทำงานที่บังคับใช้:
Thought: (คิดว่าต้องทำอะไร)
Action: (เลือกเครื่องมือ: sql_db_list_tables, sql_db_schema, หรือ sql_db_query)
Action Input: (คำสั่ง SQL หรือข้อมูลที่ต้องใส่ พิมพ์ในบรรทัดเดียวกัน ห้ามขึ้นบรรทัดใหม่)

--- ตัวอย่างการทำงานที่ถูกต้อง ---
Thought: ฉันต้องคิวรี่ข้อมูลเครื่อง X-64-110 ของวันที่ 7
Action: sql_db_query
Action Input: SELECT * FROM machine_logs WHERE mc_code = 'X-64-110' AND date_time LIKE '%-07 %'
-------------------------

ข้อตกลงสำคัญ (CRITICAL RULES):
1.  ทันทีที่คุณพิมพ์ Action Input เสร็จ คุณต้อง "หยุดพิมพ์ทันที" ห้ามพิมพ์คำว่า Observation หรือ Thought ต่อท้ายขึ้นมาเองเด็ดขาด!
2.  ห้ามคัดลอกวันที่จากข้อมูลตัวอย่าง (Sample rows) มาเขียนคำสั่ง SQL ให้ใช้วันที่จาก "คำถามของผู้ใช้" เท่านั้น
"""

sql_agent = create_sql_agent(
    llm=llm, 
    db=db, 
    agent_type="zero-shot-react-description", 
    verbose=True,
    handle_parsing_errors=True, # ช่วยจัดการกรณีที่ Agent คายรูปแบบคำตอบผิด
    agent_executor_kwargs={"handle_parsing_errors": True},
    prefix=PREFIX,
    max_iterations=5,
    top_k=10
    )


# # ==========================================
# #  5. โครงสร้างข้อมูล Pydantic
# # ==========================================
# class StructuredChatResponse(BaseModel):
#     answer: str = Field(description="คำตอบที่ละเอียดและครอบคลุม (ตอบภาษาไทยเท่านั้น ห้ามตอบภาษาจีน)")
#     image_links: List[str] = Field(description="รายการ URL ของรูปภาพทั้งหมดที่พบในข้อมูลอ้างอิง ให้ดึงมาเฉพาะลิงก์ HTTP ข้างในวงเล็บ () เท่านั้น หากไม่มีให้ปล่อยว่าง", default=[])
#     sentiment: str = Field(description="อารมณ์ของประโยคคำถาม (Positive, Negative, Neutral)")
#     confidence_score: float = Field(description="ความมั่นใจในคำตอบ 0.0 - 1.0")

# structured_llm = llm.with_structured_output(StructuredChatResponse)

class ChatRequest(BaseModel):
    query: str
    department: Optional[str] = None

class QueryFilterSchema(BaseModel):
    product: Optional[str] = Field(
        description="ชื่อรุ่นสินค้าภาษาอังกฤษ ตัวเลข เครื่องหมาย - และ _ (เช่น RGPZ-542ML-1B_test) ห้ามดึงคำศัพท์ภาษาไทย หากไม่มีให้คืนค่า null",
        default=None
        )

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
            logger.info(f"\n [CACHE HIT] โหลดจาก Cache ข้ามการค้นหาใหม่ -> คำถาม: '{request.query}'", flush=True)
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

# ---------------------------------------------------------
            # ขั้นตอนที่ 1: Self-Querying (สกัด Metadata จากคำถาม)
            # ---------------------------------------------------------
            search_kwargs = {"k": 10, "filter": {}}

            if request.department and request.department != "dataInsights":
                search_kwargs["filter"] = {"department": request.department}
            
            extractor_llm = llm.with_structured_output(QueryFilterSchema)
            extracted_product = None # 1. สร้างตัวแปรพักค่า
            
            try:
                # ใช้ Prompt แบบยกตัวอย่าง (AI จะเข้าใจและสกัดคำได้แม่นยำกว่า)
                strict_prompt = f"""
                Extract the Product Name or APN from the user's query.
                
                Examples:
                - Query: "ขอ detail RGPZ-542ML-1B หน่อย" => Product: "RGPZ-542ML-1B"
                - Query: "สรุปปัญหา 821-05561-04" => Product: "821-05561-04"
                - Query: "ขอ detail RGPZ-542ML-1B_test" => Product: "RGPZ-542ML-1B_test"
                - Query: "ขอรายละเอียดหน่อย" => Product: null
                - Query: "มีข้อมูลอะไรบ้าง" => Product: null
                
                Current Query: "{request.query}"
                """
                extracted_meta = extractor_llm.invoke(strict_prompt)
                
                # 2. นำค่าที่สกัดได้ มาอัปเดตใส่ตัวแปร extracted_product
                if extracted_meta and extracted_meta.product:
                    if "ข้อมูล" not in extracted_meta.product:
                        extracted_product = extracted_meta.product
                        logger.info(f"[SELF-QUERY] สกัดชื่อ Product ได้สำเร็จ: {extracted_product}", flush=True)
                    else:
                        logger.info(f"[SELF-QUERY] สกัดได้คำผิด ({extracted_meta.product}) ถือว่าหาไม่เจอ", flush=True)

            except Exception as e:
                logger.warning(f"[SELF-QUERY] สกัดข้อมูลล้มเหลว: {e}")

            # 4. ถ้าหา Product เจอ ให้ใส่ไปใน Filter แล้วค้นหาต่อ
            if extracted_product is not None:
                search_kwargs["filter"]["product"] = extracted_product
            
            # เคลียร์ filter ที่ว่างเปล่าออก เพื่อไม่ให้ VectorDB error
            if not search_kwargs["filter"]:
                del search_kwargs["filter"]
            

            # =======================================
            # เพิ่ม DEBUG PRINT 
            print("\n" + "="*50)
            print(f" [DEBUG] คำถามที่ใช้ค้นหา: '{request.query}'")
            print(f" [DEBUG] Filter ที่ส่งให้ VectorDB: {search_kwargs}")
            print("="*50 + "\n", flush=True)
            # =======================================

            # ---------------------------------------------------------
            # ขั้นตอนที่ 2: ค้นหาข้อมูล (พร้อม Filter อัตโนมัติ)
            # ---------------------------------------------------------
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
                # clean tag url รูปภาพออกจาก raw_context
                image_urls = re.findall(r'!\[.*?\]\((http[s]?://[^\)]+)\)', raw_context) 
                unique_image_urls = list(set(image_urls)) # ลบ URL รูปที่ซ้ำกันออก
                clean_text_context = re.sub(r'!\[.*?\]\((http[s]?://[^\)]+)\)', '', raw_context)


                print("\n" + "="*50)
                print("="*50, flush=True)
                print(raw_context, flush=True)
                print("="*50 + "\n", flush=True)
                
# ใช้คำสั่งเป็นภาษาอังกฤษทั้งหมด เพื่อป้องกันการคิดในหัวเป็นภาษาจีน
#                 system_instruction = f"""You are a professional English-to-Thai translator for factory operations. 
# Your ONLY task is to extract relevant information from the context based on the user's query and translate it into Thai.

# CRITICAL RULES:
# 1. OUTPUT LANGUAGE: Strictly output in Thai language only. ABSOLUTELY NO CHINESE CHARACTERS (禁止输出中文).
# 2. DIRECT OUTPUT: Provide ONLY the translated text. Do not add any conversational text, greetings, explanations, or meta-commentary.
# 3. NO THINKING PROCESS: Do not output your thinking process or apologize. Just give the final translated text.
# 4. FORMATTING: You MUST preserve the exact original formatting, including line breaks, bullet points (1., 2., -), and hierarchy. Each bullet point or sub-item MUST start on a new line. Do not merge them into a single paragraph.
# 5. IMAGE URLS: If you see an image tag like ![](http...), ignore it in the text.
# 6. TOPIC LISTING ONLY (CRITICAL FORMAT): If the user asks a general question like "มีข้อมูลอะไรบ้าง" or asks for available topics, DO NOT translate the detailed content. You MUST ONLY extract the main headings/topics found in the context. 
# You MUST format the output EXACTLY as a vertical bulleted list, putting each topic on a new line.
# Example format:
# - Information
# - Occurrence
# - Containment Action
# - Root cause analysis
# - Corrective action
# - Preventive action

# Context for translation:
# {clean_text_context}"""

                system_instruction = f"""You are a professional English-to-Thai translator for factory operations. 
Your ONLY task is to extract relevant information from the context based on the user's query and translate it into Thai language.

CRITICAL RULES:
1. OUTPUT LANGUAGE: Strictly output in Thai language only (ภาษาไทยเท่านั้น). ABSOLUTELY NO CHINESE CHARACTERS AND NO ENGLISH SENTENCES.
2. DIRECT OUTPUT: Provide ONLY the translated text. Do not add any conversational text or thinking process.
3. FORMATTING: Translate the text, but KEEP the original Markdown elements (like ## for headings, and 1. 2. or - for lists). Each bullet point MUST start on a new line.
4. IMAGE URLS: If you see an image tag like ![](http...), strictly ignore it.
5. TOPIC TO PRODUCT SUMMARY (CRITICAL): If the user does NOT specify a product (e.g., "ขอข้อมูล Bad mark on SUS plate"), you MUST format the output EXACTLY like this:

พบปัญหา **'[Topic Name]'** ในรุ่นสินค้าต่อไปนี้:

- **Product:** [Product Name] (APN: [APN ID])

💡 คุณต้องการทราบรายละเอียดหัวข้อใดเพิ่มเติม เช่น **Information, Occurrence,** หรือ **Root cause analysis** สามารถพิมพ์ถามได้เลยครับ
6. TOPIC LISTING: If the user asks a broad question (e.g., "มีอะไรบ้าง"), ONLY list the translated main topics as bullet points.
7. NO SELF-CORRECTION OR META-COMMENTARY: You are a machine. DO NOT evaluate your own work. DO NOT apologize. Just output the final Thai translation directly.
8. ABSOLUTE CHINESE BAN: Under NO circumstances should any Chinese characters (Hanzi) appear in your output.

Context for translation:
{clean_text_context}"""

                if extracted_product is None:
                    # กรณีไม่ระบุรุ่นสินค้า สั่งให้ตอบแค่รายชื่อรุ่น (สรุป)
                    task_instruction = "Task: The user did NOT specify a product. ONLY list the products associated with this topic as a summary. DO NOT translate the detailed content."
                else:
                    # ถ้าระบุรุ่นสินค้ามาแล้ว ค่อยสั่งให้แปลเนื้อหาละเอียด
                    task_instruction = "Task: Extract the specific details about the requested product from the context and translate into Thai. Keep the Markdown headings/bullets."

                messages = [
                    SystemMessage(content=system_instruction), 
                    HumanMessage(content=f"Query: {request.query}\n{task_instruction}")
                ]
                response_object = llm.invoke(messages)
                answer_text = response_object.content
                
                final_response_dict = {
                    "answer": answer_text,
                    "images": unique_image_urls,   
                    "sentiment": "Neutral",        
                    "confidence_score": 0.95,      
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

