import os
import time
from llama_parse import LlamaParse

# ==========================================
#  1. ใส่ API Key ของคุณ
# ==========================================

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-iErFAtd52GFZzL6eg2ICAUa8KBOLCIRsPKqE2rkRbebjtEe0"

#  เปลี่ยนชื่อไฟล์ตรงนี้ให้ตรงกับชื่อไฟล์ที่คุณ Rename ใหม่
PDF_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/KC07_8D.pdf"  
OUTPUT_MD_FILE = "8D_Report_LlamaParse_Perfect.md"

# ==========================================
#  2. เช็คก่อนว่าไฟล์มีอยู่จริงไหม
# ==========================================
if not os.path.exists(PDF_FILE):
    print(f" ระบบหาไฟล์ '{PDF_FILE}' ไม่เจอ กรุณาเช็คชื่อไฟล์อีกครั้ง")
    exit()

print(f" กำลังส่งไฟล์ {PDF_FILE} ให้ LlamaParse (โหมด Premium)...")
start_time = time.time()

custom_instruction = """
เอกสารนี้คือ 'คู่มือและรายงาน 8D Report' ภาษาไทย ซึ่งอยู่ในรูปแบบสไลด์นำเสนอ
กฎในการสกัดข้อมูล:
1. ภาษาไทย: สกัดอักษร สระ และวรรณยุกต์ให้ถูกต้อง 100% ห้ามมีสระลอย
2. หัวข้อสไลด์: ให้ตั้งชื่อหัวข้อของแต่ละหน้าโดยใช้ ## หรือ ### เสมอ
3. การจัดการตาราง (สำคัญมาก!): 
   - แปลงตารางทุกอันให้เป็น Markdown Table 
   - หากตารางมีการ "ผสานเซลล์ (Merge Cells)" ให้คุณคัดลอกข้อความไปเติมในเซลล์ที่ว่างให้ครบทุกช่อง ห้ามปล่อยให้ช่องว่าง
4. เก็บข้อมูลทุกอย่างให้ครบ ห้ามย่อความ ห้ามสรุปเอง
"""

parser = LlamaParse(
    result_type="markdown",
    language="th",
    premium_mode=True,
    parsing_instruction=custom_instruction,
)

try:
    documents = parser.load_data(PDF_FILE)
    
    #  เช็คว่า LlamaParse ส่งข้อมูลกลับมาไหม
    if len(documents) == 0:
        print(" LlamaParse คืนค่าว่างเปล่า! อาจเกิดจาก API Key ผิด หรือเน็ตเวิร์คบล็อกการส่งไฟล์")
        exit()

    with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text + "\n\n")

    end_time = time.time()
    print(f"\n เสร็จสมบูรณ์แบบ Perfect! ใช้เวลาไป {end_time - start_time:.2f} วินาที")
    print(f" บันทึกไฟล์ไว้ที่: {OUTPUT_MD_FILE}")

except Exception as e:
    print(f"\n เกิดข้อผิดพลาดระหว่างส่งไป LlamaParse: {e}")