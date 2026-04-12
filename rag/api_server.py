from datetime import datetime
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.builder import RagService
from agent.runtime import context as research_context, init_runtime, set_progress_callback
from llm_factory import create_default_llm
from pdf_processor import PDFProcessor
from rag_system import setup_rag_system


app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks_db = {}

print("Initializing RAG runtime...")
pdf_processor = PDFProcessor(output_dir="./md", lang="en", dpi=220)
rag_system = setup_rag_system()
init_runtime(
    rag_system=rag_system,
    pdf_processor=pdf_processor,
    llm=create_default_llm(),
)
print("RAG runtime initialized.")


class QuestionRequest(BaseModel):
    question: str


def run_rag_task(task_id: str, question: str):
    """Run the agent-based RAG workflow in the background."""
    tasks_db[task_id]["status"] = "processing"

    def timeline_cb(message: str):
        tasks_db[task_id]["timeline"].append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message,
            }
        )

    try:
        research_context.reset()
        set_progress_callback(timeline_cb)
        timeline_cb("Agent 开始处理查询。")

        agent_service = RagService()
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

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = {
            "answer": structured_answer.answer or "Agent 未能生成有效答案",
            "evidence_list": structured_answer.evidence_list,
        }
    except Exception as exc:
        tasks_db[task_id]["status"] = "failed"
        timeline_cb(f"任务执行失败: {exc}")
        import traceback

        traceback.print_exc()


@app.post("/api/task/create")
async def create_task(req: QuestionRequest, bg_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "id": task_id,
        "status": "pending",
        "timeline": [],
        "result": None,
    }
    bg_tasks.add_task(run_rag_task, task_id, req.question)
    return {"task_id": task_id, "message": "任务创建成功"}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]
