from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

#  ตั้งค่าพื้นฐาน (ต้องตรงกับตอนที่นำเข้าข้อมูล)
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "company_documents_md_4" 

def test_retrieval():
    print(" กำลังโหลด Embedding Model (bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    print(" กำลังเชื่อมต่อ Vector Database...")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )

    print("\n ระบบพร้อมแล้ว! พิมพ์คำถามเพื่อทดสอบได้เลย (พิมพ์ 'exit' เพื่อออก)")
    print("=" * 60)

    while True:
        query = input("\n คำถามของคุณ: ")
        if query.lower() in ['exit', 'quit', 'ออก']:
            print(" จบการทดสอบ")
            break
            
        print(" กำลังค้นหาข้อมูลที่เกี่ยวข้องที่สุด 3 อันดับแรก...\n")
        
        # ค้นหาเอกสาร 3 ชิ้น (k=3) ที่ความหมายใกล้เคียงกับคำถามมากที่สุด
        # similarity_search_with_score จะคืนค่ามาเป็น (Document, Score)
        results = vector_store.similarity_search_with_score(query, k=5)
        
        for i, (doc, score) in enumerate(results, 1):
            # หมายเหตุ: สำหรับ pgvector ค่า score คือ "ระยะห่าง" (Distance)
            # ยิ่งค่าน้อย (เข้าใกล้ 0) แปลว่า "ยิ่งเหมือน/ยิ่งแม่นยำ" ครับ
            print(f"[อันดับที่ {i}]  ความแม่นยำ (ระยะห่าง): {score:.4f}")
            print(f"หมวดหมู่ (Metadata): {doc.metadata}")
            print(f"เนื้อหาที่ค้นเจอ: {doc.page_content}")
            print("-" * 60)

if __name__ == "__main__":
    test_retrieval()