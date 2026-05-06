import os
import time
import subprocess

# ==========================================
#  1. ตั้งค่าไฟล์
# ==========================================
PDF_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/KC07_8D.pdf"           # ชื่อไฟล์ PDF ที่เราเปลี่ยนเป็นภาษาอังกฤษแล้ว
OUTPUT_DIR = "8D_Marker_Output"      # ชื่อ "โฟลเดอร์" ที่จะเก็บผลลัพธ์

print(f" กำลังส่งไฟล์ {PDF_FILE} ให้ Marker...")
print(f" โหมดรีดพลัง GPU: เตรียมตัวประมวลผลด้วย RTX 4500!")
start_time = time.time()

# เช็คไฟล์ก่อน
if not os.path.exists(PDF_FILE):
    print(f" ไม่พบไฟล์ '{PDF_FILE}' กรุณาตรวจสอบชื่อไฟล์")
    exit()

try:
    # ==========================================
    #  2. เรียกใช้งาน Marker ผ่าน Command
    # ==========================================
    # สั่ง batch_multiplier=2 เพื่อสั่งให้ Marker กิน VRAM เพิ่มขึ้น (ประมวลผลเร็วขึ้น)
    # VRAM 24GB ของคุณรับไหวสบายๆ ครับ
    command = f"marker_single {PDF_FILE} {OUTPUT_DIR} --batch_multiplier 2"
    
    # รันคำสั่ง
    subprocess.run(command, shell=True, check=True)

    end_time = time.time()
    print(f"\n เสร็จสมบูรณ์แบบสุดยอด! ใช้เวลาไป {end_time - start_time:.2f} วินาที")
    print(f" เข้าไปดูไฟล์ .md ได้ในโฟลเดอร์: {OUTPUT_DIR}")

except subprocess.CalledProcessError as e:
    print(f"\n เกิดข้อผิดพลาดในการรัน Marker: {e}")