import time
import hashlib
import redis
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------
# 1. ระบบจัดการ Cache (Redis)
# ---------------------------------------------------------
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CACHE_TTL = 3600  # 1 ชั่วโมง

def get_cache_key(query: str, model_name: str, latency_mode: str) -> str:
    raw_key = f"{query}_{model_name}_{latency_mode}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

# ---------------------------------------------------------
# 2. ฟังก์ชันปั้น Context (Custom Markdown - ลด Token / ลดหลอน)
# ---------------------------------------------------------
def get_data_as_markdown(data_type: str, raw_data: list) -> str:
    if not raw_data:
        return "ไม่มีข้อมูลอ้างอิงที่เกี่ยวข้องกับคำถาม"

    final_markdown = ""

    if data_type == "sql":
        headers = list(raw_data[0].keys())
        final_markdown += f"| {' | '.join(headers)} |\n"
        final_markdown += f"|{'|'.join(['---'] * len(headers))}|\n"
        for row in raw_data:
            row_str = " | ".join(str(val).strip() for val in row.values())
            final_markdown += f"| {row_str} |\n"
        return final_markdown

    elif data_type == "vector":
        for i, doc in enumerate(raw_data):
            content = doc.get('page_content', '').strip()
            source = doc.get('metadata', {}).get('source', 'ไม่ระบุที่มา')
            final_markdown += f"### เอกสารอ้างอิง {i+1} (แหล่งที่มา: {source})\n"
            final_markdown += f"{content}\n---\n"
        return final_markdown

    return ""

# ---------------------------------------------------------
# 3. ฟังก์ชันจัดรูปแบบ Prompt & XML Tags
# ---------------------------------------------------------
def format_messages(query: str, context: str, model_name: str, latency_mode: str) -> list:
    if latency_mode == "low":
        sys_msg = "คุณคือผู้ช่วย AI ให้ตอบกลับเป็น JSON Format ที่สั้นกระชับที่สุด"
    else:
        sys_msg = "คุณคือผู้ช่วย AI ให้ตอบกลับเป็น JSON Format อย่างละเอียด"
        if "deepseek-r1" not in model_name.lower():
            sys_msg += " จงคิดวิเคราะห์ทีละขั้นตอน (Think step by step)"

    # ใช้ XML Tags <context> ครอบข้อมูล เพื่อให้ LLM โฟกัสได้ดีขึ้น
    user_content = f"""โปรดตอบคำถามโดยอ้างอิงจากข้อมูลด้านล่างนี้เท่านั้น:

<context>
{context}
</context>

คำถาม: {query}"""

    if "deepseek-r1" in model_name.lower():
        return [HumanMessage(content=user_content)]
    elif "gemma-2" in model_name.lower():
        return [HumanMessage(content=f"{sys_msg}\n\n{user_content}")]
    else:
        return [SystemMessage(content=sys_msg), HumanMessage(content=user_content)]

# ---------------------------------------------------------
# 4. ฟังก์ชันหลัก (Main Orchestrator)
# ---------------------------------------------------------
def run_llm_workflow(query: str, model_name: str, latency_mode: str, data_type: str) -> dict:
    
    # Step A: เช็ค Cache
    cache_key = get_cache_key(query, model_name, latency_mode)
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        print(f" [Cache Hit] ดึงข้อมูลจาก Redis สำหรับ {model_name}")
        return {"source": "redis_cache", "answer": cached_data}

    print(f" [Cache Miss] เริ่มกระบวนการ LangChain -> Model: {model_name}")

    # Step B: จำลองการดึงข้อมูลดิบ (Raw Data) และปั้นเป็น Markdown
    if data_type == "sql":
        mock_raw_data = [{'หมวดหมู่': 'นโยบาย', 'รายละเอียด': 'พนักงานเบิกค่าเดินทางได้ตามจริง'}]
    else:
        mock_raw_data = [{'page_content': 'ขั้นตอนการลางาน ต้องแจ้งล่วงหน้า 3 วัน', 'metadata': {'source': 'HR_Manual.pdf'}}]
        
    markdown_context = get_data_as_markdown(data_type, mock_raw_data)

    # Step C: จัด Prompt Messages
    messages = format_messages(query, markdown_context, model_name, latency_mode)

    # Step D: เตรียมเชื่อมต่อ LLM (รอเชื่อม LiteLLM ใน Phase 3)
    llm = ChatOpenAI(
        base_url="http://localhost:4000/v1", 
        api_key="sk-dummy-key", 
        model=model_name,
        temperature=0.1 if latency_mode == "low" else 0.7
    )
    
    # ข้อมูลจำลองตอบกลับ (Mock Response) รอเปลี่ยนเป็นของจริง
    final_answer = f'{{"status": "success", "model_used": "{model_name}", "message": "จัดรูปแบบ Context ด้วย XML Tags สำเร็จแล้ว!"}}'

    # Step E: เก็บลง Redis Cache
    redis_client.setex(cache_key, CACHE_TTL, final_answer)

    return {"source": "llm_model", "answer": final_answer}