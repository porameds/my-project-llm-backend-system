import uuid
import json
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# ==========================================
# 1. ตั้งค่า Database (PostgreSQL)
# ==========================================
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

#  URL เชื่อมต่อ DB (รหัสผ่าน User@FujikuraN1)
DATABASE_URL = "postgresql://postgres:User%40FujikuraN1@localhost/llm_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class LlmPromptCache(Base):
    __tablename__ = "llm_prompt_cache"
    __table_args__ = {"schema": "llm"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model = Column(String, index=True)           
    input_model = Column(Text, index=True)       
    output_model = Column(Text)                  
    condition = Column(Text, nullable=True)      
    meta_data = Column("meta", JSONB, nullable=True) 
    expires_date = Column(DateTime)              
    created_at = Column(DateTime, default=datetime.utcnow) 

# Base.metadata.create_all(bind=engine) # เปิดคอมเมนต์ถ้าต้องการสร้างตารางใหม่

# ==========================================
# 2. สร้างโครงสร้างคำตอบด้วย Pydantic (Structured Output)
# ==========================================
class StructuredChatResponse(BaseModel):
    answer: str = Field(description="คำตอบที่เป็นประโยชน์และตรงคำถามสำหรับ User (ตอบเป็นภาษาไทย)")
    sentiment: str = Field(description="อารมณ์ของประโยคคำถามจาก User (ให้ตอบแค่ Positive, Negative หรือ Neutral)")
    confidence_score: float = Field(description="คะแนนความมั่นใจในคำตอบของ AI มีค่าตั้งแต่ 0.0 ถึง 1.0")

# ==========================================
# 3. ฟังก์ชันจัดรูปแบบ Prompt 
# ==========================================
def format_messages(model_name: str, question: str, context: str = ""):
    model_name_lower = model_name.lower()
    rag_context = context if context else "ข้อมูลพื้นฐาน: AI สามารถช่วยให้ระบบทำงานได้เร็วขึ้น"
    
    # เนื่องจากเราบังคับโครงสร้าง (Structured Output) ในโค้ดหลักแล้ว 
    # ตรงนี้เราจัดรูปแบบข้อความปกติให้ AI เข้าใจบริบทก็พอครับ
    return [
        SystemMessage(content=f"คุณคือ AI Assistant ผู้ช่วยอัจฉริยะที่เชี่ยวชาญด้านการวิเคราะห์ข้อมูล\nContext:\n{rag_context}"),
        HumanMessage(content=question)
    ]

# ==========================================
# 4. ฟังก์ชันหลักสำหรับรัน Workflow (Pydantic + PG Cache)
# ==========================================
def run_llm_workflow(query: str, model_name: str, latency_mode: str = "standard", data_type: str = "vector"):
    print(f"\n ได้รับ Request: {query} | Model: {model_name}")
    
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        
        # ---------------------------------------------------------
        # Step A: ตรวจสอบ Cache ในตาราง PostgreSQL
        # ---------------------------------------------------------
        cached_record = db.query(LlmPromptCache).filter(
            LlmPromptCache.model == model_name,
            LlmPromptCache.input_model == query,
            LlmPromptCache.expires_date > now  
        ).first()

        if cached_record:
            print(f" [PG Cache Hit] ดึงข้อมูลจากฐานข้อมูล สำหรับ {model_name}!")
            # แปลง String กลับเป็น Dictionary (เนื่องจากตอนเซฟเราเซฟเป็น JSON String)
            cached_output = json.loads(cached_record.output_model)
            return {
                "source": "pg_cache", 
                "data": cached_output,
                "meta": cached_record.meta_data
            }

        print(f" [Cache Miss] ไม่พบข้อมูลใน DB -> เริ่มเรียกโมเดลจริง {model_name}")

        # ---------------------------------------------------------
        # Step B: เรียกใช้งาน LLM แบบ Structured Output ผ่าน LiteLLM
        # ---------------------------------------------------------
        messages = format_messages(model_name, query)
        llm = ChatOpenAI(
            base_url="http://localhost:4000/v1", 
            api_key="sk-hXu_Q9kM5BWMeMVbrpYsdg", 
            model=model_name,
            temperature=0.1 if latency_mode == "low" else 0.7
        )
        
        # บังคับโครงสร้างคำตอบด้วย Pydantic
        structured_llm = llm.with_structured_output(StructuredChatResponse)
        
        print(f" กำลังให้ {model_name} คิดคำตอบและจัดฟอร์แมต...")
        
        # invoke() จะคืนค่ามาเป็น Pydantic Object
        response_object = structured_llm.invoke(messages)
        
        # แปลง Object เป็น Dictionary ทันที
        final_answer_dict = response_object.dict()
        
        # ---------------------------------------------------------
        # Step C: บันทึกคำตอบ (ที่แปลงเป็น String) ลง PostgreSQL
        # ---------------------------------------------------------
        meta_info = {
            "latency_mode": latency_mode,
            "data_type": data_type,
            "recorded_at": now.isoformat()
        }
        
        new_cache = LlmPromptCache(
            model=model_name,
            input_model=query,
            # แปลง Dictionary เป็น JSON String เพื่อเก็บลงคอลัมน์ Text
            output_model=json.dumps(final_answer_dict, ensure_ascii=False), 
            condition="structured_output",
            meta_data=meta_info,
            expires_date=now + timedelta(days=1) 
        )
        db.add(new_cache)
        db.commit() 

        return {
            "source": "llm_model", 
            "data": final_answer_dict
        }
        
    except Exception as e:
        print(f" เกิดข้อผิดพลาด: {str(e)}")
        return {"source": "error", "error_msg": str(e)}
    finally:
        db.close()