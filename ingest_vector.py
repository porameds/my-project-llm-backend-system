from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
# เปลี่ยนชื่อ Collection 
COLLECTION_NAME = "company_documents_md" 

def ingest_vector():
    print(" เริ่มกระบวนการนำเข้าเอกสารด้วย Markdown Splitter...")

    # 1. จัด Format เอกสารใหม่ให้เป็น Markdown 
    mock_markdown_text = """
# ระเบียบการเบิกค่าล่วงเวลา (OT)
พนักงานที่ทำงานล่วงเวลาเกิน 18:00 น. สามารถเบิกค่าอาหารได้ 100 บาท 
และหากทำงานเกิน 22:00 น. สามารถเบิกค่าแท็กซี่กลับบ้านได้ตามจริงแต่ไม่เกิน 300 บาท

# ขั้นตอนการเปิดระบบเซิร์ฟเวอร์
1. ตรวจสอบไฟสถานะ UPS ว่าเป็นสีเขียว
2. กดปุ่ม Power ที่เครื่อง Server หลัก (Rack 1)
3. รอ 5 นาทีจนกว่าจะได้ยินเสียงปี๊บ 1 ครั้ง
4. ล็อกอินเข้าระบบด้วยสิทธิ์ Admin เพื่อตรวจสอบ Service
5. ทำซ้ำขั้นตอนที่ 1
6. ทำซ้ำขั้นตอนที่ 2 
7. ทำซ้ำขั้นตอนที่ 3
8. ทำซ้ำขั้นตอนที่ 4
    """

    # 2. ตั้งค่าการหั่นตาม Markdown Header (บอกให้ระบบรู้ว่า # คือหัวข้อระดับ 1)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # สับเนื้อหา (มันจะหั่นตรงที่มี # เท่านั้น)
    chunks = markdown_splitter.split_text(mock_markdown_text)
    print(f" สับเอกสารตามหัวข้อได้ทั้งหมด {len(chunks)} ชิ้น\n")
    
    # แอบดูว่าระบบมันหั่นออกมาหน้าตาเป็นยังไง
    for i, chunk in enumerate(chunks, 1):
        print(f" ชิ้นที่ {i}:")
        print(f"   - Metadata (ป้ายกำกับ): {chunk.metadata}")
        print(f"   - เนื้อหา: {chunk.page_content[:50]}...\n")

    # 3. ตั้งค่าตัวแปลงข้อความเป็นตัวเลข
    print(" กำลังโหลด Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 4. บันทึกลง PostgreSQL (pgvector)
    print(" กำลังบันทึกข้อมูลพิกัดลง Vector DB...")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )
    
    vector_store.add_documents(chunks)
    print(" บันทึกเอกสารแบบ Markdown ลง Vector DB เสร็จสมบูรณ์!")

if __name__ == "__main__":
    ingest_vector()