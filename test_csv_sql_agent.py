from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os


# ==========================================
#  2. ตั้งค่า Database
# ==========================================
DB_URI = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"

# ตีกรอบให้ AI มองเห็นแค่ตาราง machine_logs ตารางเดียว จะได้ทำงานเร็วๆ
db = SQLDatabase.from_uri(DB_URI, include_tables=["machine_logs"])
print(f" ตารางที่ AI สามารถเข้าไปค้นข้อมูลได้ตอนนี้: {db.get_usable_table_names()}")

# ==========================================
#  3. เชื่อมต่อสมอง AI ผ่าน LiteLLM
# ==========================================
# เชื่อมต่อสมอง AI ผ่าน LiteLLM (ที่ชี้ไปหา Ollama อีกที)
llm = ChatOpenAI(
    model="qwen-3",              # ใช้ชื่อ model_name ตามใน config.yaml 
    api_key="sk-duQBQjTqWUwbf8wp89yvfw",           
    base_url="http://localhost:4000", # ชี้ไปที่ Port ของ LiteLLM
    temperature=0
)
# ==========================================
#  4. สร้าง Agent และทดสอบคำถาม
# ==========================================
agent_executor = create_sql_agent(
    llm, 
    db=db, 
    agent_type="zero-shot-react-description", 
    verbose=True, #โหมดบ่นพึมพำว่า AI คิดอะไรตอนค้นหา
    handle_parsing_errors=True #  สำคัญมาก! ถ้า AI พิมพ์ผิดฟอร์แมต LangChain จะสั่งให้มันแก้ตัวใหม่ ไม่ตัดจบหนี
)

query = "ขอเปอร์เซ็นต์ oee ของ machine X-27-14 วันที่ 07/12/2025"
print(f"\n คำถาม: {query}\n")

try:
    response = agent_executor.invoke({"input": query})
    print(f"\n คำตอบสุดท้าย: {response['output']}")
except Exception as e:
    print(f"\n เกิดข้อผิดพลาด: {e}")