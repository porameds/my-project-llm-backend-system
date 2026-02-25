from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

# 1. ตั้งค่าการเชื่อมต่อ Database (ใช้ URL เดิมของคุณเลย)
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents" # ชื่อตาราง/คอลเลกชันที่จะเก็บเอกสาร

def ingest_data_to_vector_db():
    print(" เริ่มกระบวนการนำเข้าเอกสารสู่ Vector DB...")

    # 2. จำลองข้อความเอกสาร (ในของจริงตรงนี้อาจจะเป็นการอ่านจากไฟล์ PDF หรือ Markdown)
    mock_document_text = """
    ระเบียบการเบิกค่าล่วงเวลา (OT):
    พนักงานที่ทำงานล่วงเวลาเกิน 18:00 น. สามารถเบิกค่าอาหารได้ 100 บาท 
    และหากทำงานเกิน 22:00 น. สามารถเบิกค่าแท็กซี่กลับบ้านได้ตามจริงแต่ไม่เกิน 300 บาท
    
    ขั้นตอนการเปิดระบบเซิร์ฟเวอร์:
    1. ตรวจสอบไฟสถานะ UPS ว่าเป็นสีเขียว
    2. กดปุ่ม Power ที่เครื่อง Server หลัก (Rack 1)
    3. รอ 5 นาทีจนกว่าจะได้ยินเสียงปี๊บ 1 ครั้ง
    4. ล็อกอินเข้าระบบด้วยสิทธิ์ Admin เพื่อตรวจสอบ Service
    5. ทำซ้ำขั้นตอนที่ 1
    6. ทำซ้ำขั้นตอนที่ 2 
    7. ทำซ้ำขั้นตอนที่ 3
    8. ทำซ้ำขั้นตอนที่ 4
    """

    # 3. สับเอกสาร (Chunking) เป็นชิ้นเล็กๆ
    print(" กำลังสับเนื้อหาเอกสาร...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # หั่นทีละ 300 ตัวอักษร
        chunk_overlap=50 # ให้มีเนื้อหาเหลื่อมกัน 50 ตัวอักษรกันความหมายขาดช่วง
    )
    
    # สร้างก้อนเนื้อหา (Chunks)
    chunks = text_splitter.create_documents([mock_document_text])
    print(f" สับเอกสารได้ทั้งหมด {len(chunks)} ชิ้น")

    # 4. ตั้งค่าตัวแปลงข้อความเป็นตัวเลข (Embedding Model)
    # ใช้ bge-m3 ซึ่งเป็นโมเดลฟรีที่เก่งภาษาไทยและภาษาอังกฤษมากๆ (รันบนเครื่องคุณเอง)
    print(" กำลังโหลด Embedding Model (อาจใช้เวลาสักครู่ในครั้งแรก)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 5. นำก้อนเนื้อหา + แปลงเป็นตัวเลข -> บันทึกลง PostgreSQL (pgvector)
    print(" กำลังบันทึกข้อมูลพิกัด (Vector) ลงฐานข้อมูล PostgreSQL...")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True, # เก็บ Metadata (เช่น ชื่อไฟล์, หน้าที่) ในรูปแบบ JSONB
    )
    
    # คำสั่งนี้จะทำการแปลง Text -> Vector และ Insert ลง DB ให้อัตโนมัติ!
    vector_store.add_documents(chunks)
    
    print(" บันทึกเอกสารลง Vector DB เสร็จสมบูรณ์!")

if __name__ == "__main__":
    ingest_data_to_vector_db()