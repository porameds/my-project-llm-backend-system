import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

# ==========================================
#  1. ตั้งค่า Database และ Collection
# ==========================================
CONNECTION_STRING = "postgresql+psycopg2://postgres:User%40FujikuraN1@localhost/llm_db"
COLLECTION_NAME = "all_company_docs" # ใช้ Collection รวม

def ingest_md_to_vector(md_path, department_name):
    print(f"\n [เริ่ม] นำเข้าไฟล์: {os.path.basename(md_path)} (แผนก: {department_name})")
    
    if not os.path.exists(md_path):
        print(f" ไม่พบไฟล์ {md_path} ระบบจะข้ามไฟล์นี้ไป")
        return False

    file_name = os.path.basename(md_path)

    # 1. อ่านไฟล์ Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. ตั้งค่าการหั่นตาม Markdown Header
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        # ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # 3. สับเนื้อหา
    chunks = markdown_splitter.split_text(markdown_text)
    print(f" สับเอกสารได้: {len(chunks)} ชิ้น")

    # 4. วนลูปแปะ Metadata
    for chunk in chunks:
        chunk.metadata["department"] = department_name
        chunk.metadata["source_file"] = file_name
    
    print(f" แปะ Metadata (department={department_name}) สำเร็จ")

    # 5. โหลด Embedding Model และบันทึกลง PGVector
    print(" กำลังฝังข้อมูล (Embedding) และบันทึกลง Database...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    engine = create_engine(CONNECTION_STRING)
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True, 
    )
    
    vector_store.add_documents(chunks)
    print(f" [สำเร็จ] นำเข้า {file_name} ลง Vector DB เรียบร้อย!")
    return True

# ==========================================
#  2. จุดเริ่มต้นการทำงาน (Main Execution)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" ระบบนำเข้าเอกสารลง Vector Database (แยกตามแผนก)")
    print("=" * 60)

    # เพิ่มไฟล์และแผนกที่ต้องการนำเข้าตรงนี้ได้เลยในอนาคต!
    files_to_ingest = [
        {
            "path": "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC07_8D Report (การรายงานผลโดย 8D).md", 
            "dept": "QA"
        },
        # {"path": "/path/to/HR_Policy_2026.md", "dept": "HR"},
        # {"path": "/path/to/IT_Network_Setup.md", "dept": "IT"},
        # {"path": "/path/to/Production_Manual.md", "dept": "Production"}
    ]

    success_count = 0
    fail_count = 0

    # วนลูปนำเข้าทีละไฟล์
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

    # สรุปผลการทำงาน
    print("\n" + "=" * 60)
    print(f" กระบวนการเสร็จสิ้น! (สำเร็จ: {success_count} ไฟล์ | ล้มเหลว/ข้าม: {fail_count} ไฟล์)")
    print("=" * 60)