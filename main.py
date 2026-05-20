# ==========================================
# IMPORTS
# ==========================================
from venv import logger

import uvicorn
import pandas as pd
import uuid
import json
import os
import re

from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_img_list_for_user_delete import router as image_router
from api_img_list_for_user_delete import router as pdfs_router
from QA_api_upload import router as qa_faca

# ==========================================
# LANGCHAIN / DB
# ==========================================
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    Integer
)

from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

# ==========================================
# CONFIG
# ==========================================
PROMPT_DB_URL = (
    "postgresql+psycopg2://postgres:"
    "8XFvLYV77O7upme@10.17.32.144:5432/ai"
)

prompt_engine = create_engine(
    PROMPT_DB_URL,
    pool_pre_ping=True,
    pool_recycle=300
)

PromptSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=prompt_engine
)

PromptBase = declarative_base()

CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:"
    "User%40FujikuraN1@host.docker.internal/llm_db"
)

COLLECTION_NAME = "all_company_docs"

LLM_MODEL_NAME = "qwen-3"

LLM_API_KEY = "sk-laW9Aq8Xv3MH85OSRZHhMQ"

LLM_BASE_URL = "http://host.docker.internal:4000/v1"

# ==========================================
# FASTAPI
# ==========================================
app = FastAPI(
    title="Company Super Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ==========================================
# STATIC
# ==========================================
app.mount(
    "/static/pdfs",
    StaticFiles(
        directory="/home/smf-llm-ai/llm_backend_system/"
                  "my_llm_backend_system/marker_env/QA_FACA_pdf"
    ),
    name="pdfs"
)

app.mount(
    "/static",
    StaticFiles(
        directory="/home/smf-llm-ai/llm_backend_system/"
                  "my_llm_backend_system/marker_env"
    ),
    name="static"
)

# ==========================================
# CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ROUTERS
# ==========================================
app.include_router(image_router)
app.include_router(pdfs_router)
app.include_router(qa_faca)

# ==========================================
# DATABASE
# ==========================================
engine = create_engine(
    CONNECTION_STRING,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# ==========================================
# REQUEST MODELS
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    department: Optional[str] = None
    user: str = "guest"


# ==========================================
# CACHE TABLE
# ==========================================
class LlmPromptCache(Base):
    __tablename__ = "llm_prompt_cache"
    __table_args__ = {"schema": "public"}

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    model = Column(String, index=True)
    input_model = Column(Text, index=True)
    output_model = Column(Text)

    condition = Column(Text, nullable=True)

    meta_data = Column(
        "meta",
        JSONB,
        nullable=True
    )

    expires_date = Column(DateTime)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ==========================================
# CHAT HISTORY
# ==========================================
class ChatHistoryDB(PromptBase):
    __tablename__ = "chat_history"
    __table_args__ = {"schema": "ai"}

    id = Column(Integer, primary_key=True)

    dept = Column(String)
    user_message = Column(Text)
    bot_reply = Column(Text)
    created_at = Column(DateTime)
    user = Column(String)


# ==========================================
# SUGGESTED PROMPTS
# ==========================================
class SuggestedPrompt(PromptBase):
    __tablename__ = "suggested_prompts"
    __table_args__ = {"schema": "ai"}

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    department = Column(String, index=True)

    prompt_text = Column(Text, nullable=False)

    is_active = Column(
        DateTime,
        default=datetime.utcnow
    )

    priority = Column(Integer, default=0)


# ==========================================
# CREATE TABLES
# ==========================================
print("Creating tables...")

Base.metadata.create_all(bind=engine)
PromptBase.metadata.create_all(bind=prompt_engine)

# ==========================================
# EMBEDDING
# ==========================================
print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True
)

# ==========================================
# LLM
# ==========================================
print("Connecting LLM...")

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,

    temperature=0,

    max_tokens=4000,

    model_kwargs={
        "stop": [
            "Observation:",
            "\nObservation:",
            "Observation",
            "中文",
            "Chinese:",
            "User:"
        ]
    }
)

# ==========================================
# SQL AGENT
# ==========================================
db = SQLDatabase.from_uri(
    CONNECTION_STRING,
    include_tables=["machine_logs"]
)

PREFIX = """
คุณคือ AI ผู้เชี่ยวชาญด้านฐานข้อมูลโรงงาน
ตอบสั้น กระชับ แม่นยำ
"""

sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    prefix=PREFIX,
    max_iterations=5,
    top_k=10
)

# ==========================================
# UTILITIES
# ==========================================

CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]+')


def remove_chinese_text(text: str) -> str:
    """
    ลบภาษาจีนออกทั้งหมด
    """
    return CHINESE_PATTERN.sub('', text)


def clean_context(text: str) -> str:
    """
    ทำความสะอาด context
    """

    # ลบ markdown image
    text = re.sub(
        r'!\[.*?\]\((http[s]?://[^\)]+)\)',
        '',
        text
    )

    # ลบภาษาจีน
    text = remove_chinese_text(text)

    # ลบ space เยอะ
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_PATTERN.search(text))


def extract_product_from_query(
    query: str,
    llm_model
) -> Optional[str]:

    match = re.search(
        r'[A-Z0-9]{3,}-[A-Z0-9-]{3,}',
        query.upper()
    )

    if match:
        return match.group(0)

    strict_prompt = f"""
Extract ONLY product model name.

TEXT:
{query}

RULE:
- Return ONLY model name
- If not found return NONE
"""

    try:
        res = llm_model.invoke(strict_prompt).content.strip()

        if "NONE" in res.upper():
            return None

        if len(res) > 30:
            return None

        return remove_chinese_text(res)

    except Exception as e:
        logger.error(f"extract_product error: {e}")
        return None


def transform_user_query(
    query: str,
    history: List[Any]
) -> str:

    """
    Smart follow-up merge
    ไม่ใช้ LLM
    """

    query = query.strip()

    # -----------------------------
    # ถ้า query เป็น product model
    # -----------------------------
    product_match = re.search(
        r'[A-Z0-9]{3,}-[A-Z0-9-]{3,}',
        query.upper()
    )

    if product_match:

        # หา context ล่าสุด
        last_user_text = ""

        for msg in reversed(history):

            if isinstance(msg, HumanMessage):

                last_user_text = msg.content
                break

        return f"{last_user_text} {query}"

    # -----------------------------
    # ถ้าเป็น topic follow-up
    # -----------------------------
    topics = [
        "information",
        "occurrence",
        "root cause",
        "corrective",
        "preventive",
        "verification"
    ]

    lower_query = query.lower()

    if lower_query in topics:

        last_context = ""

        for msg in reversed(history):

            if isinstance(msg, HumanMessage):

                content = msg.content

                if re.search(
                    r'[A-Z0-9]{3,}-[A-Z0-9-]{3,}',
                    content.upper()
                ):
                    last_context = content
                    break

        return f"{last_context} {query}"

    # -----------------------------
    # default
    # -----------------------------
    return query


# ==========================================
# API
# ==========================================
@app.post("/api/chat")
async def chat_with_company_bot(request: ChatRequest):

    print("\n" + "=" * 60)
    print("[REQUEST]")
    print(f"USER: {request.user}")
    print(f"QUERY: {request.query}")
    print(f"DEPT: {request.department}")
    print("=" * 60)

    db_session = SessionLocal()

    try:

        now = datetime.utcnow()

        cache_condition = (
            request.department
            if request.department
            else "ALL_DEPARTMENTS"
        )

        target_db = (
            "SQL_DB"
            if request.department == "dataInsights"
            else "VECTOR_DB"
        )

        # ==========================================
        # LOAD HISTORY
        # ==========================================
        db_prompt_session = PromptSessionLocal()

        chat_history_langchain = []

        try:

            recent_db_chats = (
                db_prompt_session
                .query(ChatHistoryDB)
                .filter(
                    ChatHistoryDB.user == request.user,
                    ChatHistoryDB.dept == request.department
                )
                .order_by(
                    ChatHistoryDB.created_at.desc()
                )
                .limit(3)
                .all()
            )

            recent_db_chats.reverse()

            for chat in recent_db_chats:

                if chat.user_message:

                    clean_user = remove_chinese_text(
                        chat.user_message
                    )

                    chat_history_langchain.append(
                        HumanMessage(content=clean_user)
                    )

                if chat.bot_reply:

                    try:
                        reply_data = json.loads(chat.bot_reply)

                        bot_text = reply_data.get(
                            "answer",
                            ""
                        )

                    except:
                        bot_text = chat.bot_reply

                    bot_text = remove_chinese_text(bot_text)

                    chat_history_langchain.append(
                        AIMessage(content=bot_text)
                    )

        except Exception as e:
            logger.error(f"history error: {e}")

        finally:
            db_prompt_session.close()

        # ==========================================
        # QUERY TRANSFORM
        # ==========================================
        standalone_query = transform_user_query(
            request.query,
            chat_history_langchain
        )

        standalone_query = remove_chinese_text(
            standalone_query
        )

        print(f"[TRANSFORM] {standalone_query}")

        # ==========================================
        # CACHE CHECK
        # ==========================================
        cached_record = (
            db_session.query(LlmPromptCache)
            .filter(
                LlmPromptCache.model == LLM_MODEL_NAME,
                LlmPromptCache.input_model == standalone_query,
                LlmPromptCache.condition == cache_condition,
                LlmPromptCache.expires_date > now
            )
            .first()
        )

        if cached_record:

            logger.info(
                f"[CACHE HIT] {standalone_query}"
            )

            return json.loads(
                cached_record.output_model
            )

        # ==========================================
        # SQL MODE
        # ==========================================
        if target_db == "SQL_DB":

            smart_prompt = (
                f"คำถามจากผู้ใช้: {standalone_query}"
            )

            response = sql_agent.invoke({
                "input": smart_prompt
            })

            final_response_dict = {
                "answer": remove_chinese_text(
                    response.get(
                        "output",
                        "ไม่พบข้อมูล"
                    )
                ),
                "sentiment": "Neutral",
                "confidence_score": 0.95,
                "sources": [
                    "SQL Database"
                ]
            }

        # ==========================================
        # VECTOR MODE
        # ==========================================
        else:

            extracted_product = extract_product_from_query(
                standalone_query,
                llm
            )

            search_kwargs = {
                "k": 10,
                "filter": {}
            }

            if (
                request.department
                and request.department != "dataInsights"
            ):
                search_kwargs["filter"]["department"] = (
                    request.department
                )

            if extracted_product:
                search_kwargs["filter"]["product"] = (
                    extracted_product
                )

            if not search_kwargs["filter"]:
                del search_kwargs["filter"]

            results = (
                vector_store.similarity_search_with_score(
                    standalone_query,
                    **search_kwargs
                )
            )

            # ==========================================
            # NO RESULTS
            # ==========================================
            if not results:

                final_response_dict = {
                    "answer": "ไม่พบข้อมูล",
                    "images": [],
                    "sentiment": "Neutral",
                    "confidence_score": 0.0,
                    "sources": []
                }

            else:

                raw_context = ""
                source_files = []
                unique_products = set()

                for doc, score in results:

                    file_name = doc.metadata.get(
                        "source_file",
                        "unknown"
                    )

                    prod_name = doc.metadata.get(
                        "product"
                    )

                    page_content = remove_chinese_text(
                        doc.page_content
                    )

                    raw_context += (
                        f"[{file_name}]\n"
                        f"{page_content}\n\n"
                    )

                    if file_name not in source_files:
                        source_files.append(file_name)

                    if prod_name:
                        unique_products.add(prod_name)

                clean_text_context = clean_context(
                    raw_context
                )

                # ==========================================
                # TOPIC DETECTION
                # ==========================================
                topics_list = [
                    "information",
                    "occurrence",
                    "root cause",
                    "corrective",
                    "preventive",
                    "verification"
                ]

                current_query_topic = next(
                    (
                        t for t in topics_list
                        if t in request.query.lower()
                    ),
                    None
                )

                if current_query_topic is None:

                    current_query_topic = next(
                        (
                            t for t in topics_list
                            if t in standalone_query.lower()
                        ),
                        None
                    )

                has_product = extracted_product is not None

                product_list_str = "\n".join([
                    f"Product: {p}"
                    for p in unique_products
                ])

                direct_answer = None

                # ==========================================
                # ASK PRODUCT
                # ==========================================
                if not has_product:

                    direct_answer = (
                        "พบข้อมูลใน Product:\n\n"
                        f"{product_list_str}\n\n"
                        "กรุณาระบุรุ่นสินค้า"
                    )

                # ==========================================
                # ASK TOPIC
                # ==========================================
                elif (
                    has_product
                    and current_query_topic is None
                ):

                    direct_answer = (
                        f"คุณต้องการหัวข้อใดของ "
                        f"{extracted_product} ?\n\n"
                        "เช่น:\n"
                        "- Information\n"
                        "- Occurrence\n"
                        "- Root cause\n"
                        "- Corrective\n"
                        "- Preventive"
                    )

                # ==========================================
                # GENERATE ANSWER
                # ==========================================
                else:

                    topic_name = current_query_topic.upper()

                    system_instruction = f"""
คุณคือ QA Engineer AI

กฎ:
1. ตอบภาษาไทยเท่านั้น
2. ห้ามใช้ภาษาจีน
3. ใช้ข้อมูลจาก context เท่านั้น
4. อธิบายให้ครบทุกประเด็น
5. ห้ามสรุปสั้น ให้นำคำตอบมาตอบอย่างครบถ้วน
6. หาก context มี step/process ต้องอธิบายทุก step
7. หากมี risk level ต้องระบุทั้งหมด
8. หากมี root cause หลายข้อ ต้องตอบทุกข้อ

รูปแบบคำตอบ:

หัวข้อ:
รายละเอียด:
"""

                    task_instruction = f"""
ตอบเฉพาะหัวข้อ:
{topic_name}

Product:
{extracted_product}
"""

                    messages = [
                        SystemMessage(
                            content=system_instruction
                        ),

                        HumanMessage(
                            content=(
                                f"""
CONTEXT:
{clean_text_context}

USER QUESTION:
{standalone_query}

TASK:
{task_instruction}
"""
                            )
                        )
                    ]

                    response_object = llm.invoke(messages)

                    final_answer_text = (
                        response_object.content
                    )

                    final_answer_text = remove_chinese_text(
                        final_answer_text
                    )

                    direct_answer = final_answer_text

                # ==========================================
                # FINAL RESPONSE
                # ==========================================
                final_response_dict = {
                    "answer": direct_answer,
                    "images": [],
                    "sentiment": "Neutral",
                    "confidence_score": 0.90,
                    "sources": source_files
                }

        # ==========================================
        # CACHE SAVE
        # ==========================================
        answer_text = final_response_dict.get(
            "answer",
            ""
        )

        if not contains_chinese(answer_text):

            new_cache = LlmPromptCache(
                model=LLM_MODEL_NAME,
                input_model=standalone_query,
                output_model=json.dumps(
                    final_response_dict,
                    ensure_ascii=False
                ),
                condition=cache_condition,
                meta_data={
                    "department_requested":
                        request.department,

                    "routed_to":
                        target_db
                },
                expires_date=(
                    now + timedelta(days=7)
                )
            )

            db_session.add(new_cache)
            db_session.commit()

        return final_response_dict

    except Exception as e:

        logger.error(f"[API ERROR] {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        db_session.close()

@app.get("/api/get-suggested-prompts")
async def get_prompts(department: str):
    # เปลี่ยนมาใช้ PromptSessionLocal
    db_session = PromptSessionLocal()
    try:
        prompts = db_session.query(SuggestedPrompt).filter(
            SuggestedPrompt.department == department
        ).order_by(SuggestedPrompt.priority.asc()).all()
        
        return {
            "status": "success",
            "prompts": [p.prompt_text for p in prompts]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()



# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )