from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents_md"

# โหลด Embedding Model ตัวเดิมที่ใช้ตอนเซฟ
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
engine = create_engine(CONNECTION_STRING)

# เชื่อมต่อกับ Vector DB
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=engine,
)

#  ทดลองตั้งคำถาม
query = "ขั้นตอนการเปิดระบบเซิร์ฟเวอร์มีอะไรบ้าง"
print(f" คำถาม: {query}\n")

# สั่งให้ค้นหาข้อความที่คล้ายคลึงที่สุด 2 อันดับแรก (Top-K)
results = vector_store.similarity_search(query, k=2)

print(" ผลลัพธ์ที่ค้นพบจาก Vector DB:")
for i, doc in enumerate(results, 1):
    print(f"\n--- อันดับที่ {i} ---")
    print(doc.page_content)