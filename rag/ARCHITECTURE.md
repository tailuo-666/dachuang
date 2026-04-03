# RAG Architecture

## 目录位置

当前架构文档位于：

`C:\Users\jack\Desktop\demo\rag\ARCHITECTURE.md`

当前 `rag/` 目录结构如下：

```text
rag/
├── agent/
├── crawlers/
├── query/
├── retrieval/
├── testing/
├── api_server.py
├── llm_factory.py
├── main_controller.py
├── pdf_processor.py
├── rag_system.py
├── schemas.py
└── ARCHITECTURE.md
```

## 当前主流程

现在系统已经不再使用“查询分解 -> 子问题循环 -> 汇总”的旧链路。

当前正式流程是：

```text
用户输入
  -> QueryPlan 生成
  -> 本地知识库检索
  -> hook 评估相关性
  -> 如有必要则调用学术爬虫
  -> 基于证据生成最终答案
```

这个流程是单查询工作流，不再维护子问题列表，也不再保存 `qa_pairs` 一类中间结果。

## 核心模块

### 1. 查询规划

文件：

- `rag/query/optimizer.py`
- `rag/query/service.py`

职责：

- 将用户原始问题转成单一的学术查询计划
- 生成中英文检索语句
- 提取中英文关键词
- 做术语规范化

核心数据结构：

- `AcademicQueryPlan`

字段包括：

- `original_query`
- `normalized_query_zh`
- `retrieval_query_zh`
- `retrieval_query_en`
- `crawler_query_en`
- `keywords_zh`
- `keywords_en`

说明：

- `retrieval_query_zh` 用于本地知识库检索
- `crawler_query_en` 和 `keywords_en` 用于学术站点爬虫
- 查询规划不是 agent 工具，而是在 agent 启动前由 middleware 注入

### 2. 检索与相关性评估

文件：

- `rag/agent/tools_impl.py`
- `rag/retrieval/evaluator.py`

职责：

- `retrieve_local_kb` 负责调用本地 retriever
- 检索结果统一转换为标准文档结构
- 检索后立即做非 LLM 的规则式相关性评估

当前评估方式：

- 基于 `keywords_en` 做关键词覆盖率计算
- 统计 top 文档覆盖率和来源数
- 判断当前本地证据是否足够回答

输出结果写入：

- `retrieval_sufficient`
- `relevance_score`
- `relevance_reason`
- `crawl_required`

### 3. 学术爬虫

文件：

- `rag/crawlers/arxiv.py`
- `rag/agent/tools_impl.py`

职责：

- `crawl_academic_sources` 根据 `crawler_query_en` 与 `keywords_en` 爬取学术内容
- 当前默认站点是 arXiv
- 即使没有 PDF 下载和入库，也会立即返回 title/abstract 级证据

爬虫返回统一结构：

- `papers`
- `evidence_docs`
- `downloaded_count`
- `indexed_doc_count`

说明：

- 当前爬虫不再接收子问题
- 当前爬虫不再依赖旧的英文正则拆词主逻辑
- 中文输入的适配通过 QueryPlan 阶段完成

### 4. Agent 与 Hook

文件：

- `rag/agent/builder.py`
- `rag/agent/middleware.py`
- `rag/agent/runtime.py`

职责：

- 用 LangChain v1 `create_agent(...)` 构建正式 agent
- 用 middleware 约束工作流，而不是把流程控制交给模型自由发挥

当前只保留两个正式工具：

- `retrieve_local_kb`
- `crawl_academic_sources`

middleware 的作用：

- `before_agent`
  - 读取用户问题
  - 生成 `AcademicQueryPlan`
  - 初始化运行态
- `wrap_model_call`
  - 动态限制模型当前可见工具
  - 初始阶段只允许调用本地检索
  - 检索不足时才允许调用爬虫
  - 检索足够后禁止继续爬虫
- `wrap_tool_call`
  - 检索工具返回后自动评估相关性
  - 爬虫工具返回后写入补充证据

## 数据模型

文件：

- `rag/schemas.py`

当前共享模型包括：

- `AcademicQueryPlan`
- `NormalizedDocument`
- `RetrievalPayload`
- `CrawlPaper`
- `CrawlPayload`
- `RelevanceEvaluation`
- `ResearchState`

这些结构的作用是统一：

- agent state
- 工具返回格式
- 检索与爬虫证据格式
- API 层可消费的数据结构

## API 层与运行时

文件：

- `rag/api_server.py`
- `rag/main_controller.py`
- `rag/rag_system.py`
- `rag/agent/runtime.py`

职责划分：

- `api_server.py`
  - 提供 FastAPI 接口
  - 创建任务
  - 启动 agent
  - 汇总 timeline 和最终 answer
- `main_controller.py`
  - 保留 OCR / PDF / RAG 主系统初始化入口
  - 保留论文抓取和 PDF 处理的外围调度
- `rag_system.py`
  - 管理向量库、BM25、混合检索器、基础问答链
  - 在非 agent 场景下也能使用单查询 QueryPlan
- `runtime.py`
  - 保存当前运行期上下文
  - 为 API 层提供 `context.papers`

## LLM 配置入口

文件：

- `rag/llm_factory.py`

职责：

- `llm_factory.py` 提供统一 LLM 构造入口

说明：

- 当前默认走服务器本机直连 vLLM OpenAI 兼容接口
- 默认地址为 `http://127.0.0.1:8001/v1`
- 可通过 `RAG_LLM_BASE_URL`、`RAG_LLM_HOST`、`RAG_LLM_PORT`、`RAG_LLM_MODEL`、`RAG_LLM_API_KEY` 覆盖
- 后续如果切换模型或切换部署方式，应优先改 `llm_factory.py`
- 业务代码不应到处直接写死模型构造逻辑

## 当前保留的测试

文件：

- `rag/testing/fixtures.py`
- `rag/testing/test_single_query_flow.py`

覆盖内容：

- 中文 QueryPlan 生成
- 中英混合 QueryPlan 生成
- 检索充分时跳过爬虫
- 检索不足时触发爬虫
- arXiv 查询串构建
- 无 PDF 时证据文档生成
- middleware 的工具过滤逻辑

## 已删除的旧链路

以下旧链路文件和目录已经从正式主流程中移除：

- 旧查询分解逻辑
- 旧子问题驱动工具链
- 旧兼容壳文件
- `查询优化与判断相关性` 目录

当前项目中不再存在这些正式入口：

- `query_transform`
- `value_evaluator`
- `web_deep_research`
- `save_sub_task_result`
- `report_generator`
- `checkDecomposition.py`

## 后续维护建议

建议后续继续保持下面这条原则：

- 所有“查询 -> 检索 -> 判断 -> 爬虫 -> 回答”的逻辑，只放在 `rag/` 这一套正式模块里
- 不再引入第二套平行实验链路
- 如果要扩展站点爬虫，只扩展 `rag/crawlers/`
- 如果要调整判断逻辑，只改 `rag/retrieval/evaluator.py` 和 `rag/agent/middleware.py`
- 如果要调整提示词和查询规划，只改 `rag/query/`
