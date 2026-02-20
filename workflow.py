import time
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------
# 1. ระบบจัดการ Cache (จำลอง Redis ด้วย Dictionary ก่อน)
# ---------------------------------------------------------
mock_redis_cache = {}
CACHE_TTL = 3600  # 1 ชั่วโมง

def get_cache_key(query: str, model_name: str, latency_mode: str) -> str:
    # นำ parameters มาผสมกันสร้างเป็น key เพื่อไม่ให้ cache ข้ามโหมดกัน
    raw_key = f"{query}_{model_name}_{latency_mode}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

# ---------------------------------------------------------
# 2. ฟังก์ชันดึงข้อมูล (Query Data Source -> Markdown)
# ---------------------------------------------------------
def get_data_as_markdown(query: str, data_type: str) -> str:
    """ดึงข้อมูลดิบแล้วแปลงเป็น Markdown"""
    if data_type == "sql":
        return "| หมวดหมู่ | รายละเอียด |\n|---|---|\n| นโยบาย | พนักงานเบิกค่าเดินทางได้ตามจริง |"
    else: # vector
        return "### ข้อมูลจากเอกสาร\nขั้นตอนการลางาน ต้องแจ้งล่วงหน้า 3 วัน"

# ---------------------------------------------------------
# 3. ฟังก์ชันจัดรูปแบบ Prompt (Prompt Formatting)
# ---------------------------------------------------------
def format_messages(query: str, context: str, model_name: str, latency_mode: str) -> list:
    """จัดเตรียมข้อความและปรับแต่ง Role ให้เข้ากับโมเดลและโหมด Latency"""
    
    # กำหนด System Prompt ตาม Latency Mode
    if latency_mode == "low":
        sys_msg = "คุณคือผู้ช่วย AI ให้ตอบกลับเป็น JSON Format ที่สั้นกระชับที่สุด"
    else:
        sys_msg = "คุณคือผู้ช่วย AI ให้ตอบกลับเป็น JSON Format อย่างละเอียด"
        if "deepseek-r1" not in model_name.lower():
            sys_msg += " จงคิดวิเคราะห์ทีละขั้นตอน (Think step by step)"

    user_content = f"Context (ข้อมูลอ้างอิงรูปแบบ Markdown):\n{context}\n\nคำถาม: {query}"

    # ปรับ Role ตามความเรื่องมากของแต่ละโมเดล
    if "deepseek-r1" in model_name.lower():
        # DeepSeek R1 ไม่ใช้ System Message
        return [HumanMessage(content=user_content)]
    elif "gemma-2" in model_name.lower():
        # Gemma 2 ให้ยุบรวม System ไปอยู่ใน User Message
        return [HumanMessage(content=f"{sys_msg}\n\n{user_content}")]
    else:
        # Llama 3 / Qwen และอื่นๆ รองรับ System Message ปกติ
        return [
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_content)
        ]

# ---------------------------------------------------------
# 4. ฟังก์ชันหลัก (Main Orchestrator)
# ---------------------------------------------------------
def run_llm_workflow(query: str, model_name: str, latency_mode: str, data_type: str) -> dict:
    """รันกระบวนการทั้งหมดตาม Flowchart"""
    
    # Step A: เช็ค Cache
    cache_key = get_cache_key(query, model_name, latency_mode)
    current_time = time.time()
    
    if cache_key in mock_redis_cache:
        cached = mock_redis_cache[cache_key]
        if current_time < cached['expires_at']:
            print(f" [Cache Hit] ดึงข้อมูลจาก Cache สำหรับ {model_name}")
            return {"source": "cache", "answer": cached['response']}
        else:
            print(f" [Cache Expired] หมดอายุ ทำการลบทิ้ง")
            del mock_redis_cache[cache_key]

    print(f" [Cache Miss] เริ่มกระบวนการ LangChain -> Model: {model_name}")

    # Step B: ดึงข้อมูลและทำ Markdown
    markdown_context = get_data_as_markdown(query, data_type)

    # Step C: จัด Prompt Messages
    messages = format_messages(query, markdown_context, model_name, latency_mode)

    # Step D: เรียกใช้ LLM ผ่าน LiteLLM Proxy Server
    # สมมติว่า LiteLLM ของคุณรันอยู่ที่พอร์ต 4000
    llm = ChatOpenAI(
        base_url="http://localhost:4000/v1", 
        api_key="sk-dummy-key", # ใส่คีย์จำลองไปก่อนถ้า LiteLLM ไม่ได้ตั้งบังคับไว้
        model=model_name,
        temperature=0.1 if latency_mode == "low" else 0.7
    )
    
    # ---  ส่วนนี้คอมเมนต์ไว้ก่อน เพื่อให้คุณรันโค้ดทดสอบได้โดยที่ยังไม่มี LiteLLM รันอยู่จริงๆ ---
    # response = llm.invoke(messages)
    # final_answer = response.content
    # --------------------------------------------------------------------------------------
    
    # ข้อมูลจำลองตอบกลับ (Mock Response)
    final_answer = f'{{"status": "success", "model_used": "{model_name}", "message": "นี่คือคำตอบจาก LLM ยืนยันว่ารับ Context แล้ว"}}'

    # Step E: เก็บลง Cache
    mock_redis_cache[cache_key] = {
        'response': final_answer,
        'expires_at': current_time + CACHE_TTL
    }

    return {"source": "llm_model", "answer": final_answer}