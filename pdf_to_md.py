import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

#  ตั้งค่าโมเดล Qwen-3 (ผ่าน LiteLLM)
llm = ChatOpenAI(
    model="qwen-3", 
    api_key="sk-hXu_Q9kM5BWMeMVbrpYsdg", 
    base_url="http://localhost:4000", 
    temperature=0  
)

def extract_text_from_pdf(pdf_path):
    print(f" กำลังอ่านข้อความจากไฟล์ {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n\n"
    return text

def generate_markdown_with_ai(raw_text, output_md_path):
    print(" กำลังส่งให้ AI (Qwen-3) จัดหน้า Markdown... (อาจใช้เวลาสักครู่)")
    
    # Prompt สั่งงานให้ AI จัดหน้าตาเอกสารให้ถูกต้องตามหลัก RAG
    system_prompt = """
    คุณคือผู้เชี่ยวชาญด้านการจัดเอกสาร (Document Parser)
    หน้าที่ของคุณคือ นำข้อความดิบที่ดึงมาจากไฟล์ PDF (ซึ่งมักจะเละและจัดหน้าผิด) 
    มาเรียบเรียงใหม่ให้อยู่ในรูปแบบ Markdown ที่สวยงาม ถูกต้อง และมีโครงสร้างชัดเจน

    กฎสำคัญ:
    1. ใช้ # สำหรับชื่อคู่มือหรือหัวข้อใหญ่สุด (Header 1)
    2. ใช้ ## สำหรับหมวดหมู่หลัก (Header 2)
    3. ใช้ ### สำหรับหัวข้อย่อย (Header 3)
    4. ห้ามสรุปหรือตัดเนื้อหาทิ้งเด็ดขาด ต้องคงข้อมูลไว้ให้ครบถ้วน 100%
    5. ซ่อมแซมคำผิดหรือสระที่ลอย (ถ้ามี) ให้อ่านรู้เรื่อง
    6. ถ้าเนื้อหามีการระบุว่าเป็น "ตาราง" ให้พยายามจัดฟอร์แมตเป็นตาราง Markdown
    7. ตอบกลับมาเฉพาะเนื้อหา Markdown เท่านั้น ห้ามมีคำอธิบายอื่นผสม
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "นี่คือข้อความดิบจาก PDF:\n\n{raw_text}")
    ])
    
    # สร้าง Chain และสั่งรัน
    chain = prompt | llm
    response = chain.invoke({"raw_text": raw_text})
    
    # บันทึกผลลัพธ์ลงไฟล์ .md
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(response.content)
        
    print(f" AI จัดหน้าเสร็จสมบูรณ์! บันทึกไฟล์ไว้ที่: {output_md_path}")
    print(" คำแนะนำ: เปิดไฟล์นี้ดูและตรวจสอบ/แก้ไขความถูกต้องเล็กน้อยก่อนนำเข้า Vector DB นะครับ")

if __name__ == "__main__":
    PDF_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/QAI-N1-SMF-090-2803.pdf"  # ใส่ชื่อไฟล์ PDF ของคุณ
    MD_FILE = "ai_generated_doc.md"       # ชื่อไฟล์ Markdown ขาออก
    
    if os.path.exists(PDF_FILE):
        raw_text = extract_text_from_pdf(PDF_FILE)
        # ส่งข้อความดิบไปให้ AI จัดหน้า
        generate_markdown_with_ai(raw_text, MD_FILE)
    else:
        print(f" ไม่พบไฟล์ {PDF_FILE}")