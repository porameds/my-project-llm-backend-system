from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from workflow import run_llm_workflow 

app = FastAPI(title="LLM Backend API")

# 1. กำหนดรูปแบบ Parameter ที่รับเข้ามา (อ้างอิงสเต็ป "Get Parameter")
class ChatRequest(BaseModel):
    query: str
    model_name: str = "llama-3" # ค่าเริ่มต้น
    latency_mode: str = "low"   # ค่าเริ่มต้น (low หรือ high)
    data_type: str = "vector"   # บอกว่าให้ไปหาข้อมูลที่ sql หรือ vector

# 2. สร้าง API Endpoint สำหรับรับ Request
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"ได้รับ Request: {request.query} | Model: {request.model_name}")
        
        # 3. ส่งข้อมูลไปให้ LangChain จัดการ 
        response = run_llm_workflow(
            query=request.query,
            model_name=request.model_name,
            latency_mode=request.latency_mode,
            data_type=request.data_type
        )
        
        # จำลองผลลัพธ์ไปก่อน
        response = {"answer": "นี่คือคำตอบจำลอง ระบบได้รับคำสั่งคุณแล้ว!"}
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)