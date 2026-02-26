import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

#  ตั้งค่า Database
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents_md_3" 

def ingest_md_to_vector(md_path):
    print(f"\n เริ่มกระบวนการนำเข้าไฟล์ {md_path} ลง Vector DB...")
    
    if not os.path.exists(md_path):
        print(f" ไม่พบไฟล์ {md_path} กรุณารันไฟล์ 1_pdf_to_md.py ก่อนครับ")
        return

    # 1. อ่านไฟล์ Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. ตั้งค่าการหั่นตาม Markdown Header
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # 3. สับเนื้อหา
    chunks = markdown_splitter.split_text(markdown_text)
    print(f" สับเอกสารตามหัวข้อได้ทั้งหมด {len(chunks)} ชิ้น\n")

    # 4. โหลด Embedding Model (bge-m3)
    print(" กำลังโหลด Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 5. บันทึกลง PostgreSQL (pgvector)
    print(" กำลังบันทึกข้อมูลลง Database...")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )
    
    vector_store.add_documents(chunks)
    print(" นำเข้าเอกสารลง Vector DB เสร็จสมบูรณ์! พร้อมใช้งานในระบบ RAG แล้ว!")

if __name__ == "__main__":
    MD_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/ai_generated_doc.md"  # 
    ingest_md_to_vector(MD_FILE)