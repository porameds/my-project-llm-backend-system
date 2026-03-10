FROM python:3.10-slim

WORKDIR /app

# 1. ก๊อปปี้ requirements.txt จากโฟลเดอร์ docker 
COPY docker/requirements.txt .

# 2. ติดตั้ง Library
RUN pip install --no-cache-dir -r requirements.txt

# 3. ก๊อปปี้ไฟล์และโฟลเดอร์ทั้งหมดจากโฟลเดอร์โปรเจกต์หลัก เข้าไปใน Docker
COPY . .

EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]