from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from datetime import datetime
from main_controller import OCRRAGController
from rag_system import setup_rag_system
from agent.builder import RagService
from agent.runtime import context as research_context, init_runtime, set_progress_callback
from llm_factory import create_default_llm

app = FastAPI(title="RAG Backend API")

# 允许跨域请求（方便前端联调）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存任务数据库：生产环境中建议替换为 Redis + Celery
tasks_db = {}

# 全局初始化 Controller
controller = OCRRAGController("./pdf")
print("正在系统启动时初始化 RAG 环境...")
controller.setup_pdf_processor()
controller.setup_arxiv_crawler()
controller.rag_system = setup_rag_system()

# 初始化 Agent 工具依赖
init_runtime(
    rag_system=controller.rag_system,
    pdf_processor=controller.process_pdf_folder,
    llm=create_default_llm(),
)
print("系统初始化完成！")


# 定义前端请求的数据结构
class QuestionRequest(BaseModel):
    question: str


def run_rag_task(task_id: str, question: str):
    """后台执行 RAG 主流程的函数 - 使用 Agent 架构"""
    tasks_db[task_id]["status"] = "processing"

    # 闭包回调函数：用于收集 Agent 的中间状态并附加时间戳
    def timeline_cb(message: str):
        tasks_db[task_id]["timeline"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message
        })

    try:
        # 重置研究上下文
        research_context.reset()
        # 设置进度回调
        set_progress_callback(timeline_cb)
        timeline_cb("Agent 开始处理查询...")

        # 创建并运行 Agent 服务
        agent_service = RagService()
        result = agent_service.run(query=question, thread_id=task_id)

        timeline_cb("Agent 执行完成，正在提取结果...")

        # 提取最终答案：取最后一条无 tool_calls 的 AI 消息
        messages = result.get("messages", [])
        final_answer = ""
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            tool_calls = getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None)
            if content and not tool_calls:
                final_answer = content
                break

        # 提取 sources：从 context.papers（检索和爬虫阶段都会写入）
        seen = set()
        formatted_docs = []
        for doc in research_context.papers:
            if hasattr(doc, 'page_content'):
                source = doc.metadata.get("source", "未知文件")
                content = doc.page_content[:300]
            elif isinstance(doc, dict):
                source = doc.get("source") or doc.get("metadata", {}).get("title") or "未知文件"
                content = str(doc.get("content", ""))[:300]
            else:
                continue

            if source not in seen:
                formatted_docs.append({
                    "content": content,
                    "source": source
                })
                seen.add(source)

        timeline_cb(f"提取到 {len(formatted_docs)} 个来源")

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = {
            "answer": final_answer or "Agent 未能生成有效答案",
            "sources": formatted_docs
        }

    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        timeline_cb(f"任务执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


# 接口 1: 提交问题 (Create Task)
@app.post("/api/task/create")
async def create_task(req: QuestionRequest, bg_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # 初始化任务状态
    tasks_db[task_id] = {
        "id": task_id,
        "status": "pending",  # pending | processing | completed | failed
        "timeline": [],
        "result": None
    }

    # 将任务推入后台队列执行，不会阻塞当前请求
    bg_tasks.add_task(run_rag_task, task_id, req.question)

    return {"task_id": task_id, "message": "任务创建成功"}


# 接口 2: 获取任务状态与详情 (Get Task Status)
@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]

# 启动命令: uvicorn api_server:app --host 0.0.0.0 --port 8000
