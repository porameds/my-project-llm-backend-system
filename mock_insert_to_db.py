import os
from sqlalchemy import create_engine, text

# ==========================================
# 1. ตั้งค่า Connection ไปยัง Database ปลายทาง
# ==========================================
TRACKING_CONNECTION_STRING = "postgresql+psycopg2://postgres:8XFvLYV77O7upme@10.17.32.144:5432/ai"

def mock_insert_to_db():
    # 2. รายการไฟล์จำลองที่เราต้องการใส่เข้าไปใน DB 
    files_to_ingest = [
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC07_8D_Report.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC08_CAPA Corrective and Preventive Action (แนวทางการแก้ไขป้องกัน).md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC09 APQP.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC10 Turtle Diagram.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC11_ตัวอย่างการตอบ CAR.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC17_Cost down by VE.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC20_Teamwork and Collaboration.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC24_10 Steps for SGA.md", "dept": "Training_KC"},
        {"path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC26_5 Why Analysis.md", "dept": "Training_KC"},
    ]

    print(" กำลังเชื่อมต่อ Database...")
    engine = create_engine(TRACKING_CONNECTION_STRING)
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS ai.processed_documents (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) UNIQUE NOT NULL,
            department VARCHAR(100) NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # คำสั่ง Upsert (ถ้ามีชื่อไฟล์นี้อยู่แล้ว ให้อัปเดตข้อมูล)
    insert_query = text("""
        INSERT INTO ai.processed_documents (file_name, department, ingested_at)
        VALUES (:file_name, :department, CURRENT_TIMESTAMP)
        ON CONFLICT (file_name) DO UPDATE 
        SET department = EXCLUDED.department, 
            ingested_at = CURRENT_TIMESTAMP;
    """)

    # 4. ทำการบันทึกข้อมูล
    success_count = 0
    try:
        with engine.begin() as conn:
            # รันคำสั่งสร้างตาราง
            conn.execute(create_table_query)
            print(" ตรวจสอบและสร้างตาราง ai.processed_documents เรียบร้อย")
            print("-" * 60)
            
            # วนลูปรายชื่อไฟล์และยิงเข้า DB
            for item in files_to_ingest:
                file_name = os.path.basename(item["path"]) # ดึงเฉพาะชื่อไฟล์
                department = item["dept"]
                
                print(f"กำลังเพิ่มไฟล์: {file_name} \n -> แผนก: {department}")
                conn.execute(insert_query, {"file_name": file_name, "department": department})
                success_count += 1

        print(f" จำลองการนำข้อมูลลง DB สำเร็จทั้งหมด {success_count} รายการ!")
    except Exception as e:
        print(f"\n เกิดข้อผิดพลาดในการเชื่อมต่อหรือบันทึกข้อมูล: {e}")

if __name__ == "__main__":
    mock_insert_to_db()