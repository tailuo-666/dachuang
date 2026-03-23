# RAG Agent 系统架构文档

## 📋 目录
1. [系统概述](#系统概述)
2. [架构图](#架构图)
3. [模块详解](#模块详解)
4. [数据流](#数据流)
5. [技术栈](#技术栈)

---

## 系统概述

这是一个基于 **Agent 智能驱动** 的学术论文 RAG 问答系统，核心特点：

- 🤖 **自主决策**：Agent 根据问题复杂度自动规划执行策略
- 🔍 **混合检索**：BM25 + 向量检索，提升召回率
- 📊 **相关性评估**：BERT 语义相似度判断是否需要补充信息
- 🌐 **智能回落**：本地知识不足时自动爬取 ArXiv 论文
- 📄 **PDF 处理**：OCR + 布局分析，支持学术论文提取

---

## 架构图

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户/前端                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP API
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    api_server.py (API 层)                        │
│  - FastAPI 服务器                                                 │
│  - 任务管理 (tasks_db)                                            │
│  - 进度回调 (timeline)                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │ run_rag_task()
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   engine.py (Agent 核心)                         │
│  RagService:                                                     │
│  - LLM: qwen-plus (云端决策)                                      │
│  - Agent: 自主规划和执行                                          │
│  - SYSTEM_PROMPT: 执行策略                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ invoke 6 tools
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   tools.py (工具层)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │query_        │  │retriever     │  │value_        │          │
│  │transform     │  │              │  │evaluator     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │web_deep_     │  │save_sub_     │  │report_       │          │
│  │research      │  │task_result   │  │generator     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  依赖注入: init_tools(rag_system, pdf_processor, llm)            │
└────────────────────────┬────────────────────────────────────────┘
                         │ 调用底层能力
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    底层能力模块                                   │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ rag_system.py    │  │ checkDecompo-    │                     │
│  │ - RAGSystem      │  │ sition.py        │                     │
│  │ - 混合检索        │  │ - 查询分解        │                     │
│  │ - 答案生成        │  └──────────────────┘                     │
│  └──────────────────┘                                           │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ value.py         │  │ arxiv_crawler_   │                     │
│  │ - 相关性评估      │  │ integrated.py    │                     │
│  │ - BERT 语义相似度 │  │ - 论文爬取        │                     │
│  └──────────────────┘  └──────────────────┘                     │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ pdf_processor.py │  │ main_controller  │                     │
│  │ - PDF 提取       │  │ .py              │                     │
│  │ - OCR 处理       │  │ - 资源管理器      │                     │
│  └──────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Agent 工作流

```
用户提问
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 自主决策循环                                            │
│                                                              │
│  1. Plan (规划)                                              │
│     └─→ query_transform: 分解为子问题 [Q1, Q2, Q3]           │
│                                                              │
│  2. Loop (循环处理每个子问题)                                 │
│     ├─→ Act: retriever(Q1) - 检索本地知识库                  │
│     ├─→ Check: value_evaluator(Q1, docs) - 评估相关性        │
│     ├─→ Decide:                                              │
│     │   ├─ YES → 基于文档生成答案                             │
│     │   └─ NO → web_deep_research(Q1)                        │
│     │            ├─ 爬取 ArXiv 论文                           │
│     │            ├─ 下载 PDF                                  │
│     │            ├─ OCR 处理                                  │
│     │            ├─ 更新向量库                                │
│     │            └─ 重新检索                                  │
│     └─→ Save: save_sub_task_result(Q1, answer)              │
│                                                              │
│  3. Finish (完成)                                            │
│     └─→ report_generator() - 综合所有子答案生成最终报告       │
└─────────────────────────────────────────────────────────────┘
    ↓
返回答案 + 来源
```

---

## 模块详解

### 1️⃣ api_server.py - API 服务层

**职责**：HTTP 接口、任务管理、进度追踪

**关键功能**：
- `POST /api/task/create`: 创建问答任务
- `GET /api/task/{task_id}`: 查询任务状态
- `run_rag_task()`: 后台执行 Agent 流程

**启动初始化**：
```python
controller = OCRRAGController("./pdf")
controller.setup_pdf_processor()
controller.setup_arxiv_crawler()
controller.rag_system = setup_rag_system()
init_tools(controller.rag_system, controller.process_pdf_folder, agent_llm)
```

**响应格式**：
```json
{
  "status": "completed",
  "timeline": [
    {"time": "10:30:15", "message": "Agent 开始处理查询..."},
    {"time": "10:30:18", "message": "正在检索: 什么是RAG？"}
  ],
  "result": {
    "answer": "RAG (Retrieval-Augmented Generation) 是...",
    "sources": [
      {"content": "...", "source": "paper1.pdf"}
    ]
  }
}
```

---

### 2️⃣ engine.py - Agent 核心

**职责**：Agent 决策引擎

**关键组件**：
- `RagService`: Agent 服务类
- `llm`: qwen-plus (DashScope 云端)
- `SYSTEM_PROMPT`: 定义执行策略

**执行策略**：
```
Plan → Loop(Act → Check → Decide → Save) → Finish
```

**代码示例**：
```python
agent_service = RagService()
result = agent_service.run(query="如何优化RAG检索质量？")
# 返回: {"messages": [...]}
```

---

### 3️⃣ tools.py - 工具层

**职责**：6个 Agent 工具 + 依赖注入

#### 工具列表

| 工具 | 功能 | 调用对象 |
|------|------|---------|
| `query_transform` | 查询分解 | `QueryDecomposerAgent(_llm)` |
| `retriever` | 本地检索 | `_rag_system.retriever.invoke()` |
| `value_evaluator` | 相关性评估 | `RelevanceGraderTool._run()` |
| `web_deep_research` | 爬虫+更新 | `ArxivCrawler + PDFProcessor + RAGSystem` |
| `save_sub_task_result` | 保存答案 | `context.qa_pairs.append()` |
| `report_generator` | 生成报告 | `context.qa_pairs` |

#### 依赖注入机制

```python
# 模块级变量
_rag_system = None
_pdf_processor = None
_llm = None
_progress_callback = None

# 启动时注入
def init_tools(rag_system, pdf_processor, llm):
    global _rag_system, _pdf_processor, _llm
    _rag_system = rag_system
    _pdf_processor = pdf_processor
    _llm = llm
```

#### ResearchContext（全局上下文）

```python
class ResearchContext:
    qa_pairs = []          # 子问题答案对
    original_query = ""    # 原始查询
    sub_questions = []     # 子问题列表
    papers = []            # 检索到的文档（用于提取 sources）
```

---

### 4️⃣ rag_system.py - RAG 核心

**职责**：向量检索、混合检索、答案生成

**关键配置**：
- **LLM**: DeepSeek (vLLM `localhost:8001`)
- **Embeddings**: BGE-m3 (本地 `/root/.cache/modelscope/hub/models/BAAI/bge-m3`)
- **混合检索**: BM25 (40%) + Vector (60%)
- **向量库**: FAISS

**核心方法**：
```python
# 被 tools.py 调用
docs = rag_system.retriever.invoke("什么是RAG？")

# 更新向量库（新 PDF 处理后）
rag_system.update_rag_system()
```

---

### 5️⃣ checkDecomposition.py - 查询分解

**职责**：判断是否需要分解查询

**策略**：
1. spaCy NER 提取实体
2. 计算复杂度评分
3. 边界情况使用 LLM 轻量检查

**返回**：
```python
needs_decomposition, reason, sub_questions = decomposer.route_query(query)
# 示例: True, "复杂查询", ["子问题1", "子问题2", "子问题3"]
```

---

### 6️⃣ value.py - 相关性评估

**职责**：评估检索文档是否足以回答问题

**策略**：
- **BERT 策略**（默认）：BGE-m3 语义相似度，阈值 0.75
- **LLM 策略**：大模型深度理解（可选）

**返回**：
```python
result = grader._run(query, documents, strategy="bert")
# {"action": "use_context", "score": 0.82, "reason": "semantic_similarity_pass"}
# 或
# {"action": "call_crawler", "score": 0.65, "reason": "low_similarity"}
```

**模型配置**：
```python
# 优先使用本地 BGE-m3
local_embedding_path = "/root/.cache/modelscope/hub/models/BAAI/bge-m3"
# 回退到在线模型
fallback = "sentence-transformers/all-MiniLM-L6-v2"
```

---

### 7️⃣ arxiv_crawler_integrated.py - 论文爬虫

**职责**：爬取、下载、格式化 ArXiv 论文

**核心方法**：
```python
crawler = ArxivCrawlerIntegrated("./paper_results")
papers = crawler.crawl_papers("RAG optimization", max_pages=3)
crawler.download_papers(papers, max_downloads=3)
```

---

### 8️⃣ pdf_processor.py - PDF 处理

**职责**：PDF 文本提取（OCR + 布局分析）

**技术**：
- PyMuPDF (fitz)
- PaddleOCR
- pdftotext

**输出**：Markdown 文件到 `./md/`

---

### 9️⃣ main_controller.py - 资源管理器

**职责**：管理 PDF/爬虫/RAG 资源

**保留方法**（被 tools.py 调用）：
- `setup_pdf_processor()`
- `setup_arxiv_crawler()`
- `process_all_pdfs()`

**弃用方法**（已被 Agent 替代）：
- `ask_question_with_fallback()`
- `evaluate_relevance()`

---

### 🔟 query_processor.py - 查询处理

**职责**：文档评分、答案生成

**被调用者**：`rag_system.py` 的 `enhanced_ask_question()`

---

## 数据流

### 场景 1：简单查询（本地知识充足）

```
用户: "什么是RAG？"
  ↓
api_server.py: 创建任务
  ↓
engine.py: Agent 启动
  ↓
tools.py: query_transform → 不分解（单一问题）
  ↓
tools.py: retriever → 检索本地向量库
  ↓ (返回 5 个文档)
tools.py: value_evaluator → BERT 评分 0.85 (PASS)
  ↓
tools.py: save_sub_task_result → 保存答案
  ↓
tools.py: report_generator → 生成最终报告
  ↓
api_server.py: 提取答案和来源
  ↓
返回: {"answer": "RAG是...", "sources": [...]}
```

### 场景 2：复杂查询（需要爬虫）

```
用户: "对比 DeepSeek 和 ChatGPT 的推理能力和成本"
  ↓
api_server.py: 创建任务
  ↓
engine.py: Agent 启动
  ↓
tools.py: query_transform → 分解为 3 个子问题
  - Q1: "DeepSeek 的推理能力如何？"
  - Q2: "ChatGPT 的推理能力如何？"
  - Q3: "两者的成本对比"
  ↓
【循环处理 Q1】
tools.py: retriever(Q1) → 检索本地向量库
  ↓ (返回 3 个文档，但相关性低)
tools.py: value_evaluator(Q1) → BERT 评分 0.62 (FAIL)
  ↓
tools.py: web_deep_research(Q1)
  ├─ ArxivCrawler: 爬取 3 篇论文
  ├─ 下载 PDF 到 ./paper_results/
  ├─ PDFProcessor: OCR 处理 → ./md/
  ├─ RAGSystem: update_rag_system() 更新向量库
  └─ retriever(Q1): 重新检索 → 返回新文档
  ↓
tools.py: save_sub_task_result(Q1, answer)
  ↓
【循环处理 Q2, Q3...】
  ↓
tools.py: report_generator → 综合 3 个子答案
  ↓
api_server.py: 提取答案和来源
  ↓
返回: {"answer": "综合分析...", "sources": [新下载的论文...]}
```

---

## 技术栈

### 后端框架
- **FastAPI**: API 服务器
- **LangChain**: Agent 框架、工具链

### LLM 配置

| 组件 | 模型 | 位置 | 用途 |
|------|------|------|------|
| engine.py | qwen-plus | DashScope 云端 | Agent 决策、推理 |
| rag_system.py | DeepSeek-R1 | vLLM localhost:8001 | 文档问答生成 |
| checkDecomposition.py | qwen-plus | 注入自 engine.py | 查询分解 |
| value.py | BGE-m3 (BERT) | 本地 | 相关性评估 |

### Embedding 模型
- **BGE-m3**: `/root/.cache/modelscope/hub/models/BAAI/bge-m3`
- **维度**: 1024
- **用途**: 文档向量化、语义相似度计算

### 向量数据库
- **FAISS**: 本地向量库 (`./faiss/`)
- **BM25**: 关键词检索
- **混合检索**: BM25 (40%) + Vector (60%)

### NLP 工具
- **spaCy**: 中文 NER (`zh_core_web_sm`)
- **PaddleOCR**: PDF OCR 处理

### 爬虫
- **requests + BeautifulSoup**: ArXiv 爬取
- **PyMuPDF (fitz)**: PDF 解析

---

## 目录结构

```
D:\pycharmcode\llm\
├── api_server.py              # API 服务层
├── engine.py                  # Agent 核心
├── tools.py                   # 工具层
├── rag_system.py             # RAG 核心
├── checkDecomposition.py     # 查询分解
├── value.py                  # 相关性评估
├── arxiv_crawler_integrated.py # 论文爬虫
├── pdf_processor.py          # PDF 处理
├── main_controller.py        # 资源管理器
├── query_processor.py        # 查询处理
│
├── pdf/                      # PDF 输入目录
├── md/                       # Markdown 输出目录
├── faiss/                    # FAISS 向量库
├── paper_results/            # 爬虫输出目录
└── documents/                # 文档存储
```

---

## 关键改进（集成后）

### 改进前（线性流程）
```
用户提问 → 检索 → 简单评估 → 爬虫（如果需要） → 返回答案
```
- ❌ 决策逻辑固定
- ❌ 无法处理复杂查询
- ❌ 评估标准简单

### 改进后（Agent 驱动）
```
用户提问 → Agent 自主规划 → 动态执行 → 智能回落 → 综合报告
```
- ✅ 自主决策（Plan-Act-Check-Decide）
- ✅ 查询分解（处理复杂问题）
- ✅ BERT 语义评估（精准判断）
- ✅ 自动爬虫+更新（知识增量）
- ✅ 进度追踪（实时反馈）

---

## 配置文件

### 环境变量
```bash
# DashScope API Key (qwen-plus)
DASHSCOPE_API_KEY=sk-e4b7b6386950428bb71c658d47da47ef

# vLLM 服务地址
VLLM_BASE_URL=http://localhost:8001/v1

# 本地模型路径
BGE_M3_PATH=/root/.cache/modelscope/hub/models/BAAI/bge-m3
```

### 启动命令
```bash
# 启动 vLLM 服务（DeepSeek）
python -m vllm.entrypoints.openai.api_server \
  --model ../llm/DeepSeek-R1-0528-Qwen3-8B \
  --port 8001

# 启动 API 服务
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## 测试验证

### 1. 启动验证
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
观察日志：
- ✅ "Tools initialized with dependencies"
- ✅ "使用本地 BGE-m3 模型"
- ✅ "系统初始化完成！"

### 2. 简单查询测试
```bash
curl -X POST http://localhost:8000/api/task/create \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？"}'
  预期：返回 task_id，轮询后获得答案
```

### 3. 复杂查询测试
```bash
curl -X POST http://localhost:8000/api/task/create \
  -H "Content-Type: application/json" \
  -d '{"question": "对比DeepSeek和ChatGPT的推理能力和成本"}'
  预期：timeline 中出现"正在分解查询"、多次"正在检索"
```

### 4. 爬虫回落测试
提交本地库无相关内容的问题，验证：
- timeline 出现 "正在执行深度爬虫"
- timeline 出现 "正在处理下载的PDF并更新向量库"
- sources 包含新下载的论文

---

## 维护日志

### 2025-01-XX - Agent 架构集成
- ✅ 修复 engine.py Bug（未定义变量）
- ✅ 实现 tools.py 依赖注入机制
- ✅ 重写 api_server.py 使用 Agent 架构
- ✅ 完成 value.py TODO（使用本地 BGE-m3 模型）
- ✅ 保持 main_controller.py 向后兼容

---

## 联系方式

如有问题，请参考：
- 计划文档: `C:\Users\admin123\.claude\plans\replicated-honking-toucan.md`
- 内存文档: `C:\Users\admin123\.claude\projects\D--pycharmcode-llm\memory\MEMORY.md`
