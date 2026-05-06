import os
import urllib.parse
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

BASE_PDF_DIR = "/home/smf-llm-ai/llm_backend_system/my_llm_backend_system/marker_env/QA_FACA_pdf"

@router.get("/api/pdf/list")
async def list_all_pdfs(request: Request):
    if not os.path.exists(BASE_PDF_DIR):
        return {"data": []}

    folders_dict = {}
    
    # ดึง URL หลักของเซิร์ฟเวอร์แบบ Dynamic
    base_url = str(request.base_url).rstrip("/")

    try:
        for root, dirs, files in os.walk(BASE_PDF_DIR):
            for file in files:
                # เปลี่ยน Filter เป็น .pdf
                if file.lower().endswith('.pdf'):
                    
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, BASE_PDF_DIR)
                    folder_name = os.path.basename(root)
                    
                    # แปลง Path ให้ปลอดภัยสำหรับ URL
                    safe_path = urllib.parse.quote(relative_path)
                    
                    # ปรับ URL ให้ชี้ไปที่ static endpoint ของ PDF
                    pdf_url = f"{base_url}/static/pdfs/{safe_path}"
                    
                    if folder_name not in folders_dict:
                        folders_dict[folder_name] = []
                        
                    folders_dict[folder_name].append({
                        "relative_path": relative_path,
                        "filename": file,
                        "url": pdf_url
                    })

        formatted_data = []
        for folder, pdf_list in folders_dict.items():
            formatted_data.append({
                "folderName": folder,
                "fileCount": len(pdf_list),
                "pdfs": pdf_list
            })

        return {
            "data": formatted_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการดึงรายการ PDF: {str(e)}")