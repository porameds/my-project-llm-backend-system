import os
import time
from docling.document_converter import DocumentConverter

# ==========================================
#  1. ตั้งค่าไฟล์
# ==========================================
PDF_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/KC07_8D.pdf"  # 👈 เปลี่ยนเป็นชื่อไฟล์ PDF ของคุณ
OUTPUT_MD_FILE = "8D_Report_Docling.md"

print(f" กำลังส่งไฟล์ {PDF_FILE} ให้ Docling วิเคราะห์โครงสร้างและตาราง...")
start_time = time.time()

# เช็คว่าไฟล์มีอยู่จริงไหม
if not os.path.exists(PDF_FILE):
    print(f" ไม่พบไฟล์ '{PDF_FILE}' กรุณาตรวจสอบชื่อไฟล์")
    exit()

try:
    # ==========================================
    #  2. เรียกใช้งาน Docling
    # ==========================================
    # DocumentConverter จะทำหน้าที่จำลองการมองเห็นหน้ากระดาษ และดึงโครงสร้างออกมา
    converter = DocumentConverter()
    
    # เริ่มสกัดข้อมูล
    result = converter.convert(PDF_FILE)
    
    # แปลงผลลัพธ์เป็น Markdown
    markdown_text = result.document.export_to_markdown()

    # ==========================================
    #  3. บันทึกไฟล์
    # ==========================================
    with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    end_time = time.time()
    print(f"\n เสร็จสมบูรณ์! ใช้เวลาไป {end_time - start_time:.2f} วินาที")
    print(f" บันทึกไฟล์ไว้ที่: {OUTPUT_MD_FILE}")

except Exception as e:
    print(f"\n เกิดข้อผิดพลาด: {e}")