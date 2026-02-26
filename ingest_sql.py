import pandas as pd
from sqlalchemy import create_engine

# 1. ตั้งค่าการเชื่อมต่อ Database ตัวเดิมของคุณ
DB_URI = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"

def ingest_csv_to_sql(csv_file_path, table_name):
    print(f" กำลังอ่านไฟล์: {csv_file_path}...")
    
    try:
        # 2. อ่านไฟล์ CSV ด้วย Pandas
        df = pd.read_csv(csv_file_path)
        
        print(f" พบข้อมูลทั้งหมด {len(df)} แถว")
        print(f" คอลัมน์ที่พบ: {', '.join(df.columns)}")
        print(" กำลังเชื่อมต่อฐานข้อมูลและสร้างตาราง...")
        
        # 3. เชื่อมต่อฐานข้อมูล
        engine = create_engine(DB_URI)
        
        # 4. นำข้อมูลลง Database (ความเจ๋งอยู่ตรงบรรทัดนี้ครับ!)
        # if_exists='replace': ถ้ามีตารางนี้อยู่แล้วให้เขียนทับ (เหมาะกับการเทส)
        # index=False: ไม่ต้องเอาเลข Index ของ Pandas ใส่ลงไปใน Database
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        
        print(f" โยนข้อมูลเข้าสู่ตาราง '{table_name}' ในฐานข้อมูลสำเร็จ 100%!")
        
    except FileNotFoundError:
        print(f" หาไฟล์ '{csv_file_path}' ไม่เจอครับ รบกวนเช็กชื่อไฟล์หรือตำแหน่งที่วางอีกทีนะ")
    except Exception as e:
        print(f" เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    # ชื่อไฟล์ CSV ที่เราเตรียมไว้
    CSV_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/smart_machine_oee_20251207.csv" 
    
    # ตั้งชื่อตารางที่ต้องการให้ไปโผล่ใน PostgreSQL
    TABLE_NAME = "machine_logs" 
    
    # สั่งลุย!
    ingest_csv_to_sql(CSV_FILE, TABLE_NAME)