import os
import glob
import time
import subprocess
import requests
import json
import re

# ==========================================
#  1. ตั้งค่าไฟล์และโฟลเดอร์
# ==========================================
PDF_FILE = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/KC11_ตัวอย่างการตอบ CAR  (หา Root Cause ทำ Corrective Action).pdf"
MARKER_OUT_DIR = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/Marker_Output"
FINAL_MD_FILE = "KC11_ตัวอย่างการตอบ CAR.md" # ชื่อไฟล์ผลลัพธ์สุดท้ายที่คลีนแล้ว

# ==========================================
#  2. ตั้งค่า LLM (LiteLLM / Ollama)
# ==========================================
LLM_BASE_URL = "http://localhost:4000/v1/chat/completions" 
# LLM_BASE_URL = "http://localhost:11434/v1/chat/completions"
LLM_API_KEY = "sk-b-_s7oHZUViziY1nKAPAtg" #key qwen for clean Thai language
# LLM_API_KEY = "sk-hXu_Q9kM5BWMeMVbrpYsdg"
LLM_MODEL_NAME = "qwen-3" 

def is_thai_document(text):
    """
    ตรวจสอบว่ามีตัวอักษรภาษาไทยอยู่ในข้อความหรือไม่
    """
    thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
    return len(thai_chars) > 50

def run_marker():
    print(f"\n[Step 1] กำลังให้ Marker สกัดโครงสร้างตารางจาก PDF...")
    # หากรัน Marker ไปแล้ว และมีไฟล์ .md อยู่แล้ว สามารถคอมเมนต์ 2 บรรทัดนี้เพื่อข้ามได้เลยครับ
    command = f'marker_single "{PDF_FILE}" --output_dir "{MARKER_OUT_DIR}"'
    subprocess.run(command, shell=True, check=True)
    print(" Marker ทำงานเสร็จสิ้น!")

def find_generated_md():
    md_files = glob.glob(f"{MARKER_OUT_DIR}/**/*.md", recursive=True)
    if not md_files:
        raise FileNotFoundError("หาไฟล์ .md ที่ Marker สร้างไม่เจอครับ")
    return md_files[0]

def clean_thai_text_with_llm(raw_md_text):
    print(f"\n[Step 2] เตรียมส่งข้อความให้ {LLM_MODEL_NAME} คลีนภาษาไทย...")
    
    # 1. System Prompt (ภาษาอังกฤษ) เพื่อป้องกัน AI สับสนและตอบเป็นภาษาจีน
    system_prompt = """You are an expert Thai language proofreader. Your ONLY task is to correct OCR errors, fix floating vowels (สระลอย), and correct typos in the provided Thai text.

CRITICAL RULES:
1. DO NOT translate. Keep Thai as Thai, English as English.
2. DO NOT output any Chinese characters. This is strictly forbidden.
3. DO NOT change the Markdown structure, tables (|...|), or headers (#).
4. DO NOT add any conversational filler. Output ONLY the corrected text.
"""

    # 2. Chunking: หั่นข้อความเป็นส่วนๆ (แบ่งตามย่อหน้า)
    paragraphs = raw_md_text.split('\n\n')
    chunks = []
    current_chunk = ""
    max_chunk_size = 1500 # จำกัด 1,500 ตัวอักษรต่อรอบ ไม่ให้ AI ค้าง
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    print(f" หั่นเอกสารออกเป็น {len(chunks)} ส่วน เพื่อป้องกันระบบค้าง\n")

    # 3. วนลูปส่งให้ AI คลีนทีละก้อน
    final_cleaned_text = ""
    
    for i, chunk_text in enumerate(chunks):
        print(f"    -> กำลังคลีนส่วนที่ {i+1}/{len(chunks)} ... ", end="", flush=True)
        
        user_prompt = f"Please fix the OCR errors in the following text. DO NOT output Chinese. Output ONLY the corrected Markdown.\n\n<text>\n{chunk_text}\n</text>"

        payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1 # ตัดความครีเอทีฟออก 100%
        }

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            start_chunk_time = time.time()
            # บังคับให้ AI ต้องตอบกลับภายใน 120 วินาที ไม่งั้นถือว่า Timeout
            response = requests.post(LLM_BASE_URL, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                print(f"\n [API Error] แจ้งเตือนจาก LiteLLM: {response.text}")
            response.raise_for_status()
            
            result = response.json()
            cleaned_chunk = result["choices"][0]["message"]["content"]
            
            # ล้าง Tag <text> 
            cleaned_chunk = cleaned_chunk.replace("<text>", "").replace("</text>", "").strip()
            # ล้าง Markdown block ที่อาจติดมาด้วย Regex เพื่อความแม่นยำ
            cleaned_chunk = re.sub(r'^```[a-zA-Z]*\n', '', cleaned_chunk)
            cleaned_chunk = re.sub(r'\n```$', '', cleaned_chunk)
            cleaned_chunk = cleaned_chunk.replace("```", "")
            
            final_cleaned_text += cleaned_chunk.strip() + "\n\n"
            print(f" (เสร็จใน {time.time() - start_chunk_time:.1f} วิ)")
            
        except requests.exceptions.Timeout:
            print(f" [Timeout] AI ตอบกลับช้าเกินไป ข้ามส่วนนี้ไปใช้ข้อความต้นฉบับ")
            final_cleaned_text += chunk_text + "\n\n"
        except Exception as e:
            # ถ้า AI ค้างหรือมี Error จะข้ามไปใช้ข้อความต้นฉบับแทน เพื่อให้โปรแกรมเดินต่อได้
            print(f" [Error] ข้ามส่วนนี้เนื่องจาก Error: {str(e)[:50]}...")
            final_cleaned_text += chunk_text + "\n\n" 

    return final_cleaned_text.strip()

# ==========================================
#  เริ่มการทำงานหลัก
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    try:
        # 1. รัน Marker แปลง PDF -> MD
        run_marker()
        
        # # 2. หาไฟล์ .md
        # raw_md_path = find_generated_md()
        # print(f"\nเจอไฟล์ Markdown ต้นฉบับที่: {raw_md_path}")
        
        # with open(raw_md_path, "r", encoding="utf-8") as f:
        #     raw_text = f.read()
            
        # # 3. ตรวจสอบและคลีนภาษาไทย
        # if is_thai_document(raw_text):
        #     perfect_text = clean_thai_text_with_llm(raw_text)
        # else:
        #     print(f"\n[Step 2] ข้ามการใช้ LLM เนื่องจากเอกสารส่วนใหญ่เป็นภาษาอังกฤษ")
        #     perfect_text = raw_text 
        
        # # 4. เซฟเป็นไฟล์ใหม่
        # with open(FINAL_MD_FILE, "w", encoding="utf-8") as f:
        #     f.write(perfect_text)
            
        print(f"\n [เสร็จสมบูรณ์] ใช้เวลาไปทั้งหมด {time.time() - start_time:.2f} วินาที")
        print(f" บันทึกไฟล์ที่คลีนแล้วชื่อ: '{FINAL_MD_FILE}'")
        print(f"   คุณสามารถนำไฟล์ '{FINAL_MD_FILE}' ไปฝังลง Vector Database ได้เลยครับ!")

    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดรุนแรง: {e}")