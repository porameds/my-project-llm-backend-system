import os
from pypdf import PdfReader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

# ⚙️ ตั้งค่าพื้นฐาน
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents_md" 

# 📄 ตั้งชื่อไฟล์
PDF_FILE_PATH = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/QAI-N1-SMF-090-2803.pdf"      # ชื่อไฟล์ PDF ต้นฉบับที่คุณมี
MD_FILE_PATH = "QAI-N1-SMF-090-2803.md"     # ชื่อไฟล์ชั่วคราวที่จะให้คุณเข้าไปแก้ Manual

def extract_pdf_to_md(pdf_path, md_path):
    print(f"\n📄 [Step 1] กำลังอ่านไฟล์ PDF: {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            # ดึงข้อความออกมาทีละหน้า
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        
        # เซฟเป็นไฟล์ Markdown ดิบๆ ให้ User ไปแก้ต่อ
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ ดูดข้อความสำเร็จ! บันทึกไฟล์ดิบไว้ที่ '{md_path}'")
        return True
    except FileNotFoundError:
        print(f"❌ หาไฟล์ PDF '{pdf_path}' ไม่เจอครับ เอามาวางในโฟลเดอร์รึยัง?")
        return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอ่าน PDF: {e}")
        return False

def ingest_vector(md_path):
    print("\n🚀 [Step 2] เริ่มกระบวนการหั่นและนำเข้า Vector DB...")
    
    # 1. อ่านไฟล์ Markdown ที่คุณเพิ่งจัดฟอร์แมตเสร็จ
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. ตั้งค่าการหั่นตาม Markdown Header (ดักไว้ 3 ระดับเลย)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # สับเนื้อหา
    chunks = markdown_splitter.split_text(markdown_text)
    print(f"✂️ สับเอกสารตามหัวข้อ # ได้ทั้งหมด {len(chunks)} ชิ้น\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"📦 ชิ้นที่ {i}:")
        print(f"   - Metadata (หมวดหมู่): {chunk.metadata}")
        print(f"   - เนื้อหา: {chunk.page_content[:50]}...\n")

    # 3. โหลด Embedding Model (แปลง Text เป็น Vector)
    print("🧠 กำลังโหลด Embedding Model (bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 4. บันทึกลง PostgreSQL
    print("💾 กำลังบันทึกข้อมูลลงฐานข้อมูล...")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )
    
    vector_store.add_documents(chunks)
    print("🎉 บันทึกเอกสารลง Vector DB เรียบร้อย พร้อมให้ AI ค้นหาแล้ว!")

if __name__ == "__main__":
    # เช็คว่ามีไฟล์ Markdown ดราฟต์อยู่แล้วหรือยัง
    # ถ้ายังไม่มี ให้ดูดจาก PDF สร้างขึ้นมาใหม่
    if not os.path.exists(MD_FILE_PATH):
        success = extract_pdf_to_md(PDF_FILE_PATH, MD_FILE_PATH)
        if not success:
            exit()
    
    # จังหวะหยุดพักให้คุณทำงาน Manual
    print("\n=======================================================")
    print(f"🛑 โปรแกรมหยุดชั่วคราว: กรุณาไปเปิดไฟล์ '{MD_FILE_PATH}'")
    print("   👉 จัดหน้า ลบคำผิด ใส่ # สำหรับหัวข้อหลัก และ ## สำหรับหัวข้อย่อย")
    print("   👉 กด Save ไฟล์ให้เรียบร้อย")
    print("=======================================================")
    
    input("กดปุ่ม [Enter] เพื่อยืนยันว่าคุณแก้ไขไฟล์เสร็จแล้ว และพร้อมไปต่อ...")
    
    # เมื่อกด Enter จะไปสู่กระบวนการนำเข้า Vector DB
    ingest_vector(MD_FILE_PATH)