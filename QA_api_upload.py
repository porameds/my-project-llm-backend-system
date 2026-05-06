import os
import re
import time
import urllib.parse
import pandas as pd
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

# Langchain & Vector DB
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

router = APIRouter()

# ==========================================
#  1. การตั้งค่าต่างๆ (Configuration)
# ==========================================
router = APIRouter(tags=["FACA Upload"])

VECTOR_CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@host.docker.internal/llm_db"
COLLECTION_NAME = "all_company_docs"
TRACKING_CONNECTION_STRING = "postgresql+psycopg2://postgres:8XFvLYV77O7upme@10.17.32.144:5432/ai"

BASE_STATIC_URL = "http://10.17.41.116:8000/static" 

PDF_UPLOAD_DIR = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_pdf"
EXCEL_UPLOAD_DIR = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_csv"
MD_OUTPUT_DIR = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_md"

os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)
os.makedirs(EXCEL_UPLOAD_DIR, exist_ok=True)
os.makedirs(MD_OUTPUT_DIR, exist_ok=True)

# ==========================================
#  2. ฟังก์ชันหลัก (Core Logic)
# ==========================================

def process_excel_to_single_markdown(excel_filepath: str, output_directory: str) -> str:
    try:
        df = pd.read_excel(excel_filepath)
        df.columns = df.columns.str.strip()
        df = df.fillna("ไม่มีข้อมูล (N/A)")

        has_topic_col = 'Topic' in df.columns
        all_md_content = "" 

        for index, row in df.iterrows():
            topic = row['Topic'] if has_topic_col and str(row['Topic']) != "ไม่มีข้อมูล (N/A)" else f'Record_{index+1}'
            
            md_content = f"""# Topic: {topic}
## Information
{row.get('Information', 'N/A')}
## Occurrence
{row.get('Occurrence', 'N/A')}
## Containment Action
{row.get('Containment Action', 'N/A')}
## Root cause analysis
{row.get('Root cause analysis', 'N/A')}
## Corrective action
{row.get('Corrective action', 'N/A')}
## Preventive action
{row.get('Preventive action', 'N/A')}
## Verification of effectiveness
{row.get('Verification of effectiveness', 'N/A')}

---

"""
            all_md_content += md_content
            
        base_excel_name = os.path.basename(excel_filepath).replace(".xlsx", "").replace(".xls", "")
        safe_filename = "".join([c for c in base_excel_name if c.isalnum() or c=='_']).rstrip()
        final_filename = f"{safe_filename}.md"
        filepath = os.path.join(output_directory, final_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(all_md_content)
            
        print(f" แปลง Excel เป็นไฟล์ Markdown ไฟล์เดียวสำเร็จ: {final_filename}")
        return filepath
    except Exception as e:
        print(f" Error in process_excel_to_single_markdown: {e}")
        raise e


def ingest_md_to_vector(md_path: str, department_name: str) -> bool:
    if not os.path.exists(md_path):
        return False

    file_name = os.path.basename(md_path)

    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    folder_name = file_name.replace('.md', '') 
    safe_folder_name = urllib.parse.quote(folder_name)
    
    def replace_url(match):
        alt_text = match.group(1)
        raw_img_path = match.group(2)
        safe_img_file = urllib.parse.quote(os.path.basename(raw_img_path))
        new_url = f"{BASE_STATIC_URL}/folder_for_img_path_url/{safe_folder_name}/{safe_img_file}"
        return f"![{alt_text}]({new_url})"

    markdown_text = re.sub(
        r'!\[(.*?)\]\((.*?\.jpeg|.*?\.png|.*?\.jpg)\)', 
        replace_url, 
        markdown_text
    )

    headers_to_split_on = [
        ("#", "Header 1"),  
        ("##", "Header 2")  
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    chunks = markdown_splitter.split_text(markdown_text)
    
    print(f"  สับเอกสารได้ทั้งหมด: {len(chunks)} ชิ้น (จาก {file_name})")

    topic_product_map = {}

    for chunk in chunks:
        h1 = chunk.metadata.get("Header 1")
        h2 = chunk.metadata.get("Header 2")

        if h1 and h2 == "Information":
            match = re.search(r'Product\s*:\s*([^\n\r]+)', chunk.page_content, re.IGNORECASE)
            if match:
                topic_product_map[h1] = match.group(1).strip()

    for chunk in chunks:
        chunk.metadata["department"] = department_name
        chunk.metadata["source_file"] = file_name

        h1 = chunk.metadata.get("Header 1")
        product_name = topic_product_map.get(h1, "ไม่ระบุรุ่น")
        
        if product_name != "ไม่ระบุรุ่น":
            chunk.metadata["product"] = product_name

        parent_headers = []
        if "Header 1" in chunk.metadata: parent_headers.append(chunk.metadata["Header 1"])
        if "Header 2" in chunk.metadata: parent_headers.append(chunk.metadata["Header 2"])
            
        if parent_headers:
            header_context = " > ".join(parent_headers)
            chunk.page_content = f"[รุ่นสินค้า: {product_name} | หัวข้ออ้างอิง: {header_context}]\n{chunk.page_content}"

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_engine = create_engine(VECTOR_CONNECTION_STRING)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=vector_engine,
        use_jsonb=True, 
    )
    vector_store.add_documents(chunks)
    
    return True

def update_pdf_tracking_table(file_name: str, department_name: str):
    try:
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
            SET department = EXCLUDED.department, ingested_at = CURRENT_TIMESTAMP;
        """)

        with tracking_engine.begin() as conn:
            conn.execute(create_table_query)
            conn.execute(insert_query, {"file_name": file_name, "department": department_name})
            
        print(f" อัปเดต Tracking Database สำหรับ PDF สำเร็จ: {file_name}")
    except Exception as e:
        print(f" เกิดข้อผิดพลาดในการอัปเดต Tracking Database: {e}")


def update_is_check_status(original_filename: str, safe_filename: str):
    """ ค้นหาชื่อไฟล์จาก JSON โดยหาแค่ "ชื่อไฟล์" เพื่อหลีกเลี่ยงปัญหาเรื่องเว้นวรรคใน SQL """
    try:
        time.sleep(3)
        tracking_engine = create_engine(TRACKING_CONNECTION_STRING)
        
        # ใช้ LIKE ค้นหาโดยครอบด้วย Double Quote ตาม format JSON 
        update_query = text("""
            UPDATE ai.smart_llm_upload_dept 
            SET is_check = TRUE, update_date = CURRENT_TIMESTAMP
            WHERE file_data::text LIKE :search_original 
               OR file_data::text LIKE :search_safe;
        """)
        
        # ส่งพารามิเตอร์แบบมี Double Quote ครอบ (เช่น %"ชื่อไฟล์.pdf"%)
        with tracking_engine.begin() as conn:
            result = conn.execute(update_query, {
                "search_original": f'%"{original_filename}"%',
                "search_safe": f'%"{safe_filename}"%'
            })
            
            if result.rowcount > 0:
                print(f" Stamp 'is_check=TRUE' ให้ไฟล์ {safe_filename} เรียบร้อยแล้ว ({result.rowcount} records)")
            else:
                print(f" ไม่พบไฟล์ {original_filename} หรือ {safe_filename} ในตาราง smart_llm_upload_dept")
                
    except Exception as e:
        print(f" เกิดข้อผิดพลาดในการอัปเดต is_check: {e}")

# ==========================================
#  3. FastAPI Endpoints
# ==========================================

def pipeline_task(excel_filepath: str, department_name: str):
    try:
        single_md_file = process_excel_to_single_markdown(excel_filepath, MD_OUTPUT_DIR)
        
        if ingest_md_to_vector(single_md_file, department_name):
            print(f" Pipeline สำเร็จ! นำเข้าข้อมูลจากไฟล์ {os.path.basename(single_md_file)} เรียบร้อย")
        else:
            print(" Pipeline ล้มเหลวในขั้นตอนการนำเข้า VectorDB")
            
    except Exception as e:
        print(f" Pipeline Failed: {e}")

@router.post("/api/upload-qa_faca/")
async def upload_and_process_file(
    background_tasks: BackgroundTasks,
    file: List[UploadFile] = File(...),  # <-- เปลี่ยนให้รองรับการอัปโหลดแบบ Array 
    department_name: str = Form("QA_FACA")
):
    upload_results = []

    # วนลูปจัดการทุกไฟล์ที่ถูกอัปโหลดเข้ามาพร้อมกัน
    for f in file:
        original_filename = f.filename
        safe_filename = original_filename.replace(" ", "_") 
        file_extension = os.path.splitext(safe_filename)[1].lower()

        if file_extension in ['.xlsx', '.xls']:
            # -----------------------------------
            # Flow 1: สำหรับไฟล์ Excel
            # -----------------------------------
            save_path = os.path.join(EXCEL_UPLOAD_DIR, safe_filename)
            with open(save_path, "wb") as buffer:
                content = await f.read()
                buffer.write(content)

            # ส่ง safe_filename ไปเข้า Pipeline
            background_tasks.add_task(pipeline_task, save_path, department_name)
            
            # ส่งไป Stamp is_check=TRUE 
            background_tasks.add_task(update_is_check_status, original_filename, safe_filename)

            upload_results.append({
                "file": safe_filename, 
                "status": "processing", 
                "type": "excel"
            })

        elif file_extension == '.pdf':
            # -----------------------------------
            # Flow 2: สำหรับไฟล์ PDF
            # -----------------------------------
            save_path = os.path.join(PDF_UPLOAD_DIR, safe_filename)
            with open(save_path, "wb") as buffer:
                content = await f.read()
                buffer.write(content)

            # สั่งอัปเดต Tracking Table 
            background_tasks.add_task(update_pdf_tracking_table, safe_filename, department_name)
            
            # ส่งไป Stamp is_check=TRUE 
            background_tasks.add_task(update_is_check_status, original_filename, safe_filename)

            upload_results.append({
                "file": safe_filename, 
                "status": "processing", 
                "type": "pdf"
            })

        else:
            upload_results.append({
                "file": safe_filename, 
                "status": "failed", 
                "reason": "Unsupported format"
            })

    # ส่งผลลัพธ์กลับไปให้ Frontend ทราบสถานะของทุกไฟล์
    return {
        "status": "success",
        "message": f"ได้รับไฟล์จำนวน {len(file)} ไฟล์ และกำลังประมวลผล",
        "details": upload_results,
        "department": department_name
    }