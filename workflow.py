import os
import uuid
import json
from datetime import datetime, timedelta

# ==========================================
#  1. Imports ทั้งหมดที่จำเป็น
# ==========================================
# Database & Cache
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

# LangChain & LLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# SQL Agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
#  2. ตั้งค่าการเชื่อมต่อพื้นฐาน (Configurations)
# ==========================================
DB_URI = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents_md_6" # ชื่อ Collection 

# ตั้งค่า LLM หลัก (ชี้ไปที่ LiteLLM/Ollama)
LLM_MODEL_NAME = "qwen-3"
LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg" 
LLM_BASE_URL = "http://localhost:4000/v1"

# สร้าง Engine สำหรับต่อ Database
engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
#  3. โครงสร้างตาราง Cache ใน PostgreSQL
# ==========================================
class LlmPromptCache(Base):
    __tablename__ = "llm_prompt_cache"
    __table_args__ = {"schema": "public"} # ใช้ schema public ปกติ หรือแก้เป็น llm ถ้าคุณสร้างไว้แล้ว
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model = Column(String, index=True)           
    input_model = Column(Text, index=True)       
    output_model = Column(Text)                  
    condition = Column(Text, nullable=True)      
    meta_data = Column("meta", JSONB, nullable=True) 
    expires_date = Column(DateTime)              
    created_at = Column(DateTime, default=datetime.utcnow) 

# สร้างตาราง Cache ถ้ายังไม่มี
Base.metadata.create_all(bind=engine)

# ==========================================
#  4. โครงสร้างคำตอบ (Structured Output Pydantic)
# ==========================================
class StructuredChatResponse(BaseModel):
    answer: str = Field(description="คำตอบที่ละเอียดและครอบคลุมเนื้อหาจากข้อมูลอ้างอิงทั้งหมด กรุณารักษารูปแบบหัวข้อและ List (Bullet points) ไว้ให้ครบถ้วน ห้ามตัดทอนข้อมูลสำคัญ (ตอบเป็นภาษาไทย)")
    sentiment: str = Field(description="อารมณ์ของประโยคคำถามจาก User (ให้ตอบแค่ Positive, Negative หรือ Neutral)")
    confidence_score: float = Field(description="คะแนนความมั่นใจในคำตอบของ AI มีค่าตั้งแต่ 0.0 ถึง 1.0 (ถ้าเจอข้อมูลในบริบทให้ตอบ >0.8)")

# ==========================================
#  5. ฟังก์ชันดึงข้อมูล (Vector & SQL)
# ==========================================
def get_vector_context(query: str) -> str:
    """ ดึงข้อมูลจากเอกสารคู่มือ (Markdown) """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = PGVector(
        embeddings=embeddings, collection_name=COLLECTION_NAME, connection=engine, use_jsonb=True
    )
    results = vector_store.similarity_search_with_score(query, k=5)
    
    context = ""
    for doc, score in results:
        header = doc.metadata.get('Header 3', 'ข้อมูลทั่วไป')
        context += f"[{header}] {doc.page_content}\n"
    
    return context if context else "ไม่พบข้อมูลในคู่มือ"

def get_sql_context(query: str) -> str:
    """ ดึงข้อมูลจากฐานข้อมูล SQL (Machine Logs) """
    db = SQLDatabase.from_uri(DB_URI, include_tables=["machine_logs"])
    llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0)
    
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="zero-shot-react-description", verbose=False, handle_parsing_errors=True
    )
    
    try:
        response = agent_executor.invoke({"input": query})
        return response['output']
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหาฐานข้อมูล: {e}"

def route_query(query: str) -> str:
    """ วิเคราะห์อย่างง่ายว่าควรไปหาข้อมูลที่ไหน (Router) """
    sql_keywords = ["oee", "เครื่องจักร", "machine", "เปอร์เซ็นต์", "log", "วันที่"]
    # ถ้ามีคีย์เวิร์ดของเครื่องจักร ให้ไป SQL ถ้าไม่มีให้ไปค้นคู่มือ (Vector)
    if any(keyword in query.lower() for keyword in sql_keywords):
        return "SQL"
    return "VECTOR"

# ==========================================
#  6. Workflow หลัก 
# ==========================================
def run_super_agent(query: str):
    # print(f"\n[{datetime.now().strftime('%H:%M:%S')}] รับคำถาม: '{query}'")
    db_session = SessionLocal()
    
    try:
        now = datetime.utcnow()
        
        # --- Step A: ตรวจสอบ Cache ---
        cached_record = db_session.query(LlmPromptCache).filter(
            LlmPromptCache.input_model == query,
            LlmPromptCache.expires_date > now  
        ).first()

        if cached_record:
            # print(" [Cache Hit] ดึงคำตอบจากความจำเดิม (ไม่ใช้พลังงาน AI)")
            return json.loads(cached_record.output_model)

        # print(" [Cache Miss] เริ่มกระบวนการคิดและค้นหา...")

        # --- Step B: ตัดสินใจและดึงข้อมูลดิบ ---
        route = route_query(query)
        # print(f" วิเคราะห์เส้นทาง: วิ่งไปค้นข้อมูลที่ [{route}]")
        
        if route == "SQL":
            raw_context = get_sql_context(query)
        else:
            raw_context = get_vector_context(query)
            
        # print(f" ข้อมูลดิบที่หาได้: {raw_context[:100]}...")

# --- Step C: ส่งให้ LLM จัดรูปฟอร์แมต ---
        
        # 🟢 1. สั่งปริ้นท์ข้อมูลดิบออกทางหน้าจอ เพื่อเช็คว่า Vector DB ดึง Bullet points มาให้เราจริงไหม!
        print("\n🔍 [Debug] ข้อมูลอ้างอิงที่ส่งให้ AI อ่าน:\n" + "-"*40)
        print(raw_context)
        print("-" * 40)
        
        system_instruction = f"""คุณคือผู้ช่วยอัจฉริยะโรงงาน จงตอบคำถามโดยอ้างอิงจากข้อมูลต่อไปนี้เท่านั้น
        
กฎกติกาขั้นเด็ดขาด:
1. ดึงข้อมูลมาตอบให้ครบถ้วน 100% โดยเฉพาะส่วนที่เป็นรายการข้อๆ (Bullet points) หรือมีเครื่องหมายขีด (-) 
2. ห้ามสรุปรวบรัด ห้ามเขียนรวบเป็นย่อหน้าเดียว และห้ามตัดเนื้อหาทิ้งเด็ดขาด 
3. ต้องตอบกลับโดยรักษารูปแบบข้อๆ ไว้ตามต้นฉบับ

ข้อมูลอ้างอิง:
{raw_context}"""

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=query)
        ]
        
        llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0)
        structured_llm = llm.with_structured_output(StructuredChatResponse)
        response_object = structured_llm.invoke(messages)
        
        #  2. วิธีฆ่า Warning สีแดงถาวร: เราจะไม่ใช้ .model_dump() แล้ว แต่จะดึงค่าตรงๆ แทน
        if hasattr(response_object, 'parsed') and response_object.parsed is not None:
            ans = response_object.parsed.answer
            sent = response_object.parsed.sentiment
            conf = response_object.parsed.confidence_score
        else:
            ans = getattr(response_object, 'answer', 'ไม่มีคำตอบ')
            sent = getattr(response_object, 'sentiment', 'Neutral')
            conf = getattr(response_object, 'confidence_score', 0.0)

        final_answer_dict = {
            "answer": str(ans),
            "sentiment": str(sent),
            "confidence_score": float(conf)
        }

        # --- Step D: บันทึกความจำลง Cache ---
        new_cache = LlmPromptCache(
            model=LLM_MODEL_NAME,
            input_model=query,
            output_model=json.dumps(final_answer_dict, ensure_ascii=False), 
            condition=route,
            meta_data={"source_used": route},
            expires_date=now + timedelta(days=1) 
        )
        db_session.add(new_cache)
        db_session.commit() 

        return final_answer_dict

    except Exception as e:
        print(f" เกิดข้อผิดพลาด: {str(e)}")
        return {"answer": "ขออภัย ระบบเกิดข้อผิดพลาด", "sentiment": "Neutral", "confidence_score": 0.0}
    finally:
        db_session.close()

# ==========================================
#  7. หน้าต่างทดสอบการแชท (Terminal UI)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" เริ่มต้นระบบ(พิมพ์ 'exit' เพื่อออก)")
    print("=" * 60)
    
    while True:
        user_input = input("\n คุณ: ")
        if user_input.lower() in ['exit', 'quit', 'ออก']:
            print("บ๊ายบาย!")
            break
            
        result = run_super_agent(user_input)
        
        print("\n" + "-"*40)
        print(f" คำตอบ AI: {result.get('answer')}")
        print(f" อารมณ์คำถาม: {result.get('sentiment')}")
        print(f" ความมั่นใจ: {result.get('confidence_score')}")
        print("-" * 40)