import pandas as pd
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- ส่วนของ Logic การแปลงไฟล์ (คงเดิมแต่ปรับให้รับ parameter รายไฟล์) ---
def process_excel_to_markdown(excel_filepath, output_directory):
    if not excel_filepath.endswith(('.xlsx', '.xls')):
        return

    print(f"Detected new file: {os.path.basename(excel_filepath)}")
    
    try:
        # รอสักครู่เพื่อให้ระบบบันทึกไฟล์เสร็จสมบูรณ์ (ป้องกันไฟล์ถูก lock)
        time.sleep(1) 
        df = pd.read_excel(excel_filepath)
        df.columns = df.columns.str.strip()
        df = df.fillna("ไม่มีข้อมูล (N/A)")

        has_topic_col = 'Topic' in df.columns

        for index, row in df.iterrows():
            topic = row['Topic'] if has_topic_col and str(row['Topic']) != "ไม่มีข้อมูล (N/A)" else f'Record_{index+1}'
            
            md_content = f"""# Topic: {topic}
## Information
{row.get('Information', 'N/A')}
## Occurrence
{row.get('Occurrence', 'N/A')}
## Containment Action
{row.get('Containment Action', 'N/A')}
## Root cause analysis
{row.get('Root cause analysis', 'N/A')}
## Corrective action
{row.get('Corrective action', 'N/A')}
## Preventive action
{row.get('Preventive action', 'N/A')}
## Verification of effectiveness
{row.get('Verification of effectiveness', 'N/A')}
"""
            safe_filename = "".join([c for c in str(topic) if c.isalnum() or c==' ']).rstrip()
            filename = f"FACA_{os.path.basename(excel_filepath)}_{index+1}_{safe_filename[:20].strip()}.md"
            filepath = os.path.join(output_directory, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
                
        print(f" Successfully converted: {os.path.basename(excel_filepath)}")
    except Exception as e:
        print(f" Error processing {excel_filepath}: {e}")

# --- ส่วนของ Watcher ---
class ExcelHandler(FileSystemEventHandler):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_created(self, event):
        # ทำงานเมื่อมีการสร้างไฟล์ใหม่
        if not event.is_directory:
            process_excel_to_markdown(event.src_path, self.output_dir)

    def on_moved(self, event):
        # ทำงานเมื่อมีการย้ายไฟล์เข้ามาใน folder
        if not event.is_directory:
            process_excel_to_markdown(event.dest_path, self.output_dir)

if __name__ == "__main__":
    # กำหนด Path
    input_folder = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_csv"
    output_folder = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_md."

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ตั้งค่า Watcher
    event_handler = ExcelHandler(output_folder)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)

    print(f" Monitoring folder: {input_folder}")
    print("Waiting for new Excel files... (Press Ctrl+C to stop)")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped monitoring.")
    observer.join()

# import pandas as pd
# import os

# def excel_to_markdown_for_vectordb(excel_filepath, output_directory):
#     # สร้างโฟลเดอร์สำหรับเก็บไฟล์ Markdown หากยังไม่มี
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     try:
#         # ใช้ pd.read_excel() สำหรับไฟล์ .xlsx
#         df = pd.read_excel(excel_filepath)
#         print(" อ่านไฟล์ Excel สำเร็จ!")
#     except Exception as e:
#         print(f" ไม่สามารถอ่านไฟล์ Excel ได้: {e}")
#         return []

#     # คลีนชื่อคอลัมน์ (ลบช่องว่างส่วนเกิน)
#     df.columns = df.columns.str.strip()

#     # แทนที่ค่าว่าง (NaN) ด้วยข้อความที่เหมาะสม
#     df = df.fillna("ไม่มีข้อมูล (N/A)")

#     documents = []
#     has_topic_col = 'Topic' in df.columns

#     # วนลูปอ่านข้อมูลทีละแถว
#     for index, row in df.iterrows():
#         topic = row['Topic'] if has_topic_col and str(row['Topic']) != "ไม่มีข้อมูล (N/A)" else f'Record_{index+1}'
        
#         md_content = f"""# Topic: {topic}

# ## Information
# {row.get('Information', 'N/A')}

# ## Occurrence
# {row.get('Occurrence', 'N/A')}

# ## Containment Action
# {row.get('Containment Action', 'N/A')}

# ## Root cause analysis
# {row.get('Root cause analysis', 'N/A')}

# ## Corrective action
# {row.get('Corrective action', 'N/A')}

# ## Preventive action
# {row.get('Preventive action', 'N/A')}

# ## Verification of effectiveness
# {row.get('Verification of effectiveness', 'N/A')}
# """
#         documents.append(md_content)
        
#         # ตั้งชื่อไฟล์โดยลบอักขระพิเศษ
#         safe_filename = "".join([c for c in str(topic) if c.isalnum() or c==' ']).rstrip()
#         filename = f"FACA_Record_{index+1}_{safe_filename[:20].strip()}.md"
#         filepath = os.path.join(output_directory, filename)
        
#         # บันทึกเป็นไฟล์ Markdown (.md)
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write(md_content)
            
#     print(f" สร้างไฟล์ Markdown สำเร็จทั้งหมด {len(documents)} ไฟล์ ไปที่โฟลเดอร์ '{output_directory}'")
#     return documents

# # ระบุ Path ไฟล์ของคุณ (แบบ Absolute Path ใช้งานได้เลย)
# excel_file = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_csv/FACA.xlsx" 
# output_dir = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/all_md_file"

# # เรียกใช้ฟังก์ชัน
# markdown_docs = excel_to_markdown_for_vectordb(excel_file, output_dir)