import os
import re  # 🌟 เพิ่ม import re สำหรับแปลงลิงก์รูปภาพ
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
import urllib.parse

# ==========================================
#  1. ตั้งค่า Database และ Collection
# ==========================================
# 📍 1.1 DB สำหรับเก็บ Vector (Local Server)
VECTOR_CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "all_company_docs"

# 📍 1.2 DB สำหรับเก็บตาราง Tracking (Server อื่น)
# เปลี่ยนชื่อ Database ด้านหลังสุดเป็น /ai
TRACKING_CONNECTION_STRING = "postgresql+psycopg2://postgres:8XFvLYV77O7upme@10.17.32.144:5432/ai"

BASE_STATIC_URL = "http://10.17.41.116:8000/static" 

def ingest_md_to_vector(md_path, department_name):
    print(f"\n [เริ่ม] นำเข้าไฟล์: {os.path.basename(md_path)} (แผนก: {department_name})")
    
    if not os.path.exists(md_path):
        print(f" ไม่พบไฟล์ {md_path} ระบบจะข้ามไฟล์นี้ไป")
        return False

    file_name = os.path.basename(md_path)

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # ==========================================
    #  1.5 แปลง Path รูปภาพให้เป็น URL เต็ม (แบบปลอดภัยกับ Markdown)
    # ==========================================
    print(" 🔍 กำลังแปลงลิงก์รูปภาพให้อ้างอิงไปยัง FastAPI Static...")
    
    # 1. ดึงชื่อโฟลเดอร์ และแปลงช่องว่าง/ภาษาไทย ให้เป็นรหัส URL ปลอดภัย (เช่น %20)
    folder_name = file_name.replace('.md', '') 
    safe_folder_name = urllib.parse.quote(folder_name)
    
    # 2. ฟังก์ชันย่อยสำหรับแทนที่ลิงก์ทีละตัว
    def replace_url(match):
        alt_text = match.group(1)
        img_file = match.group(2)
        safe_img_file = urllib.parse.quote(img_file) # เข้ารหัสชื่อไฟล์รูปเผื่อมีช่องว่างด้วย
        
        # ⚠️ หมายเหตุ: โค้ดนี้สมมติว่ารูปภาพของคุณอยู่ในโฟลเดอร์ย่อย (เช่น marker_env/KC11.../ภาพ.jpeg)
        # ถ้าภาพของคุณกองรวมอยู่ข้างนอกโฟลเดอร์ย่อย ให้ลบ /{safe_folder_name} ออกจากบรรทัดด้านล่าง
        new_url = f"{BASE_STATIC_URL}/{safe_folder_name}/{safe_img_file}"
        
        print(f"    🖼️ แปลงแล้ว: {new_url}")
        return f"![{alt_text}]({new_url})"

    markdown_text = re.sub(
        r'!\[(.*?)\]\((.*?\.jpeg|.*?\.png|.*?\.jpg)\)', 
        replace_url, 
        markdown_text
    )
    # ==========================================

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    chunks = markdown_splitter.split_text(markdown_text)
    print(f" ✂️ สับเอกสารได้: {len(chunks)} ชิ้น")

    for chunk in chunks:
        chunk.metadata["department"] = department_name
        chunk.metadata["source_file"] = file_name
        
        parent_headers = []
        if "Header 1" in chunk.metadata:
            parent_headers.append(chunk.metadata["Header 1"])
        if "Header 2" in chunk.metadata:
            parent_headers.append(chunk.metadata["Header 2"])
            
        if parent_headers:
            header_context = " > ".join(parent_headers)
            chunk.page_content = f"[หัวข้ออ้างอิง: {header_context}]\n{chunk.page_content}"

    print(" 🧠 กำลังฝังข้อมูล (Embedding) และบันทึกลง Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_engine = create_engine(VECTOR_CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=vector_engine,
        use_jsonb=True, 
    )
    
    vector_store.add_documents(chunks)
    
    print(" 📝 กำลังอัปเดตข้อมูลไฟล์ลงตารางติดตาม...")
    tracking_engine = create_engine(TRACKING_CONNECTION_STRING)
    
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS ai.processed_documents (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) UNIQUE NOT NULL,
            department VARCHAR(100) NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    insert_query = text("""
        INSERT INTO ai.processed_documents (file_name, department, ingested_at)
        VALUES (:file_name, :department, CURRENT_TIMESTAMP)
        ON CONFLICT (file_name) DO UPDATE 
        SET department = EXCLUDED.department, 
            ingested_at = CURRENT_TIMESTAMP;
    """)

    with tracking_engine.begin() as conn:
        conn.execute(create_table_query)
        conn.execute(insert_query, {"file_name": file_name, "department": department_name})
        
    print(f" 🎉 [สำเร็จ] นำเข้า {file_name} ลงระบบเรียบร้อย!")
    return True

# ==========================================
#  2. จุดเริ่มต้นการทำงาน (Main Execution)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" ระบบนำเข้าเอกสารลง Vector Database (แยกตามแผนก)")
    print("=" * 60)

    files_to_ingest = [
        {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC10 Turtle Diagram.md", 
            "dept": "Training_KC"
        },
        {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC11_ตัวอย่างการตอบ CAR.md", 
            "dept": "Training_KC"
        },
         {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC17_Cost down by VE.md", 
            "dept": "Training_KC"
        },
          {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC20_Teamwork and Collaboration.md", 
            "dept": "Training_KC"
        },
          {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC24_10 Steps for SGA.md", 
            "dept": "Training_KC"
        },
          {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC26_5 Why Analysis.md", 
            "dept": "Training_KC"
        },
            {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC09 APQP.md", 
            "dept": "Training_KC"
        },
            {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC08_CAPA Corrective and Preventive Action (แนวทางการแก้ไขป้องกัน).md", 
            "dept": "Training_KC"
        },
            {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC07_8D_Report.md", 
            "dept": "Training_KC"
        },
    ]
    success_count = 0
    fail_count = 0

    for item in files_to_ingest:
        try:
            result = ingest_md_to_vector(item["path"], department_name=item["dept"])
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f" [Error] เกิดข้อผิดพลาดกับไฟล์ {item['path']}: {str(e)}")
            fail_count += 1 

    print("\n" + "=" * 60)
    print(f" กระบวนการเสร็จสิ้น! (สำเร็จ: {success_count} ไฟล์ | ล้มเหลว/ข้าม: {fail_count} ไฟล์)")
    print("=" * 60)


# import os
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector
# from sqlalchemy import create_engine, text  # เพิ่ม text เข้ามาเพื่อเขียนคำสั่ง SQL

# # ==========================================
# #  1. ตั้งค่า Database และ Collection
# # ==========================================
# #  1.1 DB สำหรับเก็บ Vector (Local Server ของเดิม)
# VECTOR_CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
# COLLECTION_NAME = "all_company_docs" # ใช้ Collection รวม

# #  1.2 DB สำหรับเก็บตาราง Tracking (Server อื่น)
# TRACKING_CONNECTION_STRING = "postgresql+psycopg2://postgres:8XFvLYV77O7upme@10.17.32.144:5432/ai"

# def ingest_md_to_vector(md_path, department_name):
#     print(f"\n [เริ่ม] นำเข้าไฟล์: {os.path.basename(md_path)} (แผนก: {department_name})")
    
#     if not os.path.exists(md_path):
#         print(f" ไม่พบไฟล์ {md_path} ระบบจะข้ามไฟล์นี้ไป")
#         return False

#     file_name = os.path.basename(md_path)

#     # 1. อ่านไฟล์ Markdown
#     with open(md_path, "r", encoding="utf-8") as f:
#         markdown_text = f.read()

#     # 2. ตั้งค่าการหั่นตาม Markdown Header
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         # ("###", "Header 3"),
#     ]
#     markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
#     # 3. สับเนื้อหา
#     chunks = markdown_splitter.split_text(markdown_text)
#     print(f" สับเอกสารได้: {len(chunks)} ชิ้น")

#     # 4. วนลูปแปะ Metadata และ "อัดบริบท (Context) เข้าไปในเนื้อหา"
#     for chunk in chunks:
#         chunk.metadata["department"] = department_name
#         chunk.metadata["source_file"] = file_name
        
#         # ดึงหัวข้อจาก Metadata มาสร้างเป็นบริบทนำหน้า
#         parent_headers = []
#         if "Header 1" in chunk.metadata:
#             parent_headers.append(chunk.metadata["Header 1"])
#         if "Header 2" in chunk.metadata:
#             parent_headers.append(chunk.metadata["Header 2"])
            
#         # ถ้าย่อหน้านั้นมีหัวข้อหลักกำกับอยู่ ให้เอามาแปะหน้าเนื้อหาเลย
#         if parent_headers:
#             header_context = " > ".join(parent_headers)
#             chunk.page_content = f"[หัวข้ออ้างอิง: {header_context}]\n{chunk.page_content}"

#     print(f" แปะ Metadata และบริบท (department={department_name}) สำเร็จ")

#     # 5. โหลด Embedding Model และบันทึกลง PGVector (บน Local Server)
#     print(" กำลังฝังข้อมูล (Embedding) และบันทึกลง Vector Database...")
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
#     vector_engine = create_engine(VECTOR_CONNECTION_STRING) # ใช้ Engine ของ Vector DB
    
#     vector_store = PGVector(
#         embeddings=embeddings,
#         collection_name=COLLECTION_NAME,
#         connection=vector_engine,
#         use_jsonb=True, 
#     )
    
#     vector_store.add_documents(chunks)
    
#     # ==========================================
#     #  6. อัปเดตข้อมูลไฟล์ลงตาราง Tracking อีกอัน (บน Server อื่น)
#     # ==========================================
#     print(" กำลังอัปเดตข้อมูลไฟล์ลงตารางติดตาม (Tracking Table บน Server แยก)...")
    
#     # สร้าง Engine ตัวใหม่ที่ชี้ไปยัง Server ที่สอง
#     tracking_engine = create_engine(TRACKING_CONNECTION_STRING)
    
#     # คำสั่ง SQL สร้างตาราง (ถ้ายังไม่มี)
#     create_table_query = text("""
#         CREATE TABLE IF NOT EXISTS ai.processed_documents (
#             id SERIAL PRIMARY KEY,
#             file_name VARCHAR(255) UNIQUE NOT NULL,
#             department VARCHAR(100) NOT NULL,
#             ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     """)
    
#     # คำสั่ง SQL นำข้อมูลเข้า (ใช้ Upsert: ถ้าชื่อไฟล์ซ้ำให้อัปเดตแผนกและเวลาแทน)
#     insert_query = text("""
#         INSERT INTO ai.processed_documents (file_name, department, ingested_at)
#         VALUES (:file_name, :department, CURRENT_TIMESTAMP)
#         ON CONFLICT (file_name) DO UPDATE 
#         SET department = EXCLUDED.department, 
#             ingested_at = CURRENT_TIMESTAMP;
#     """)

#     # ใช้ tracking_engine.begin() เพื่อทำ Transaction กับ Server ที่สอง
#     with tracking_engine.begin() as conn:
#         conn.execute(create_table_query)
#         conn.execute(insert_query, {"file_name": file_name, "department": department_name})
        
#     print(f"  [ตารางติดตาม] บันทึกชื่อไฟล์ '{file_name}' เข้าแผนก '{department_name}' เรียบร้อย!")
#     # ==========================================

#     print(f" [สำเร็จ] นำเข้า {file_name} ลงระบบเรียบร้อย!")
#     return True

# # ==========================================
# #  2. จุดเริ่มต้นการทำงาน (Main Execution)
# # ==========================================
# if __name__ == "__main__":
#     print("=" * 60)
#     print(" ระบบนำเข้าเอกสารลง Vector Database (แยกตามแผนก)")
#     print("=" * 60)

#     # เพิ่มไฟล์และแผนกที่ต้องการนำเข้าตรงนี้ได้เลยในอนาคต!
#     files_to_ingest = [
#         {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC10 Turtle Diagram.md", 
#             "dept": "Training_KC"
#         },
#         {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC11_ตัวอย่างการตอบ CAR.md", 
#             "dept": "Training_KC"
#         },
#          {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC17_Cost down by VE.md", 
#             "dept": "Training_KC"
#         },
#           {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC20_Teamwork and Collaboration.md", 
#             "dept": "Training_KC"
#         },
#           {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC24_10 Steps for SGA.md", 
#             "dept": "Training_KC"
#         },
#           {
#             "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC26_5 Why Analysis.md", 
#             "dept": "Training_KC"
#         },
#     ]

#     success_count = 0
#     fail_count = 0

#     # วนลูปนำเข้าทีละไฟล์
#     for item in files_to_ingest:
#         try:
#             result = ingest_md_to_vector(item["path"], department_name=item["dept"])
#             if result:
#                 success_count += 1
#             else:
#                 fail_count += 1
#         except Exception as e:
#             print(f" [Error] เกิดข้อผิดพลาดกับไฟล์ {item['path']}: {str(e)}")
#             fail_count += 1  # แก้ไขตรงนี้ (เดิมพิมพ์ค้างไว้แค่ +=)

#     # สรุปผลการทำงาน
#     print("\n" + "=" * 60)
#     print(f" กระบวนการเสร็จสิ้น! (สำเร็จ: {success_count} ไฟล์ | ล้มเหลว/ข้าม: {fail_count} ไฟล์)")
#     print("=" * 60)