from __future__ import annotations

import os
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .agent.builder import RagService
    from .agent.runtime import context as research_context, init_runtime, set_progress_callback
    from .kb_manager import KnowledgeBaseManager
    from .llm_factory import create_default_llm, get_default_llm_service
    from .pdf_processor import PDFProcessor
    from .rag_system import setup_rag_system
except ImportError:
    from agent.builder import RagService
    from agent.runtime import context as research_context, init_runtime, set_progress_callback
    from kb_manager import KnowledgeBaseManager
    from llm_factory import create_default_llm, get_default_llm_service
    from pdf_processor import PDFProcessor
    from rag_system import setup_rag_system


DEFAULT_UPLOAD_DIR = "./paper_results"


class QuestionRequest(BaseModel):
    question: str


class LLMConfigRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float | None = None


@dataclass
class BackendServices:
    tasks_db: dict[str, Any] = field(default_factory=dict)
    rag_service_cls: Any = RagService
    pdf_processor: PDFProcessor | None = None
    rag_system: Any = None
    kb_manager: KnowledgeBaseManager | None = None
    llm_service: Any = field(default_factory=get_default_llm_service)
    llm: Any = None
    runtime_initialized: bool = False


def initialize_services(services: BackendServices) -> BackendServices:
    if services.runtime_initialized and services.kb_manager is not None:
        return services

    print("Initializing RAG runtime...")
    if services.pdf_processor is None:
        services.pdf_processor = PDFProcessor(output_dir="./md", lang="en", dpi=220)
    if services.rag_system is None:
        services.rag_system = setup_rag_system()
    if services.llm is None:
        services.llm = services.llm_service.create_llm() if services.llm_service is not None else create_default_llm()

    init_runtime(
        rag_system=services.rag_system,
        pdf_processor=services.pdf_processor,
        llm=services.llm,
    )
    if services.kb_manager is None:
        services.kb_manager = KnowledgeBaseManager(
            md_dir="./md",
            pdf_root_dir="./paper_results",
            upload_dir=DEFAULT_UPLOAD_DIR,
            rag_system=services.rag_system,
            pdf_processor=services.pdf_processor,
        )
    services.kb_manager.refresh_state(rebuild_if_needed=True)
    services.runtime_initialized = True
    print("RAG runtime initialized.")
    return services


def create_app(
    *,
    services: BackendServices | None = None,
    initialize_on_startup: bool = True,
) -> FastAPI:
    app = FastAPI(title="RAG Backend API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.services = services or BackendServices()

    def ensure_services() -> BackendServices:
        current_services: BackendServices = app.state.services
        if not current_services.runtime_initialized:
            initialize_services(current_services)
        return current_services

    if initialize_on_startup:
        @app.on_event("startup")
        def startup_event() -> None:
            ensure_services()

    def run_rag_task(task_id: str, question: str) -> None:
        current_services = ensure_services()
        current_services.tasks_db[task_id]["status"] = "processing"

        def timeline_cb(message: str) -> None:
            current_services.tasks_db[task_id]["timeline"].append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": message,
                }
            )

        try:
            research_context.reset()
            set_progress_callback(timeline_cb)
            timeline_cb("Agent 开始处理查询。")

            active_llm = current_services.llm
            if active_llm is None:
                active_llm = (
                    current_services.llm_service.create_llm()
                    if current_services.llm_service is not None
                    else create_default_llm()
                )
                current_services.llm = active_llm

            agent_service = current_services.rag_service_cls(llm=active_llm)
            result = agent_service.run(query=question, thread_id=task_id)

            timeline_cb("Agent 执行完成，正在提取结构化答案。")

            final_evidence_items = [
                dict(item)
                for item in research_context.final_evidence_items
                if isinstance(item, dict)
            ]
            structured_answer = agent_service.parse_final_response(
                result,
                final_evidence_items=final_evidence_items,
            )

            evidence_lookup = {
                int(item.get("index") or 0): item
                for item in final_evidence_items
                if int(item.get("index") or 0) > 0
            }
            referenced_evidence = [
                evidence_lookup[index]
                for index in structured_answer.evidence_list
                if index in evidence_lookup
            ]

            timeline_cb(f"提取到 {len(referenced_evidence)} 条被引用证据。")

            current_services.tasks_db[task_id]["status"] = "completed"
            current_services.tasks_db[task_id]["result"] = {
                "answer": structured_answer.answer or "Agent 未能生成有效答案",
                "evidence_list": structured_answer.evidence_list,
            }
        except Exception as exc:
            current_services.tasks_db[task_id]["status"] = "failed"
            timeline_cb(f"任务执行失败: {exc}")
            traceback.print_exc()

    @app.post("/api/task/create")
    async def create_task(req: QuestionRequest, bg_tasks: BackgroundTasks) -> dict[str, Any]:
        current_services = ensure_services()
        task_id = str(uuid.uuid4())
        current_services.tasks_db[task_id] = {
            "id": task_id,
            "status": "pending",
            "timeline": [],
            "result": None,
        }
        bg_tasks.add_task(run_rag_task, task_id, req.question)
        return {"task_id": task_id, "message": "任务创建成功"}

    @app.get("/api/task/{task_id}")
    async def get_task_status(task_id: str) -> dict[str, Any]:
        current_services: BackendServices = app.state.services
        if task_id not in current_services.tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")
        return current_services.tasks_db[task_id]

    @app.post("/api/llm/config")
    async def configure_llm(req: LLMConfigRequest) -> JSONResponse:
        current_services: BackendServices = app.state.services
        llm_service = current_services.llm_service or get_default_llm_service()
        current_services.llm_service = llm_service

        api_key = str(req.api_key or "").strip()
        base_url = str(req.base_url or "").strip()
        model = str(req.model or "").strip()
        has_any_field = any([api_key, base_url, model, req.temperature is not None])
        has_full_api_config = all([api_key, base_url, model])

        success = False
        if has_full_api_config:
            success = llm_service.switch_to_api(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=req.temperature,
            )
        elif not api_key and has_any_field:
            success = llm_service.switch_to_remote(
                req.temperature,
                base_url=base_url or None,
                model=model or None,
            )

        if not success:
            return JSONResponse(
                status_code=400,
                content={"message": "参数错误，请正确填写"},
            )

        current_services.llm = llm_service.create_llm()
        if (
            current_services.runtime_initialized
            and current_services.rag_system is not None
            and current_services.pdf_processor is not None
        ):
            init_runtime(
                rag_system=current_services.rag_system,
                pdf_processor=current_services.pdf_processor,
                llm=current_services.llm,
            )

        return JSONResponse(
            status_code=200,
            content={"message": "设置已保存并正常连接"},
        )

    @app.get("/api/kb/papers")
    async def list_kb_papers(keyword: str | None = Query(default=None)) -> dict[str, Any]:
        current_services = ensure_services()
        if current_services.kb_manager is None:
            raise HTTPException(status_code=500, detail="Knowledge base manager is not initialized.")
        return current_services.kb_manager.list_papers(keyword=keyword)

    @app.post("/api/kb/papers/upload")
    async def upload_kb_paper(file: UploadFile = File(...)) -> dict[str, Any]:
        current_services = ensure_services()
        kb_manager = current_services.kb_manager
        if kb_manager is None:
            raise HTTPException(status_code=500, detail="Knowledge base manager is not initialized.")

        original_filename = os.path.basename(file.filename or "").strip()
        if not original_filename:
            raise HTTPException(status_code=400, detail="上传文件名不能为空。")
        if not original_filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持上传 PDF 文件。")

        os.makedirs(kb_manager.upload_dir, exist_ok=True)
        stored_source_file, saved_path = kb_manager.build_upload_pdf_path(original_filename)
        existing_record = kb_manager.find_paper_by_source_file(stored_source_file)
        if existing_record is not None:
            raise HTTPException(
                status_code=409,
                detail=f"文件已存在于知识库中: {stored_source_file}",
            )

        try:
            with open(saved_path, "wb") as output_file:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
        finally:
            await file.close()

        try:
            result = kb_manager.ingest_pdf(saved_path, source_file=stored_source_file)
        except Exception as exc:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "message": "文件已完成加载",
            **result,
        }

    @app.delete("/api/kb/papers/{paper_id}")
    async def delete_kb_paper(paper_id: int) -> dict[str, Any]:
        current_services = ensure_services()
        kb_manager = current_services.kb_manager
        if kb_manager is None:
            raise HTTPException(status_code=500, detail="Knowledge base manager is not initialized.")

        try:
            result = kb_manager.delete_paper(paper_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "message": "文件已删除",
            **result,
        }

    return app


app = create_app()
