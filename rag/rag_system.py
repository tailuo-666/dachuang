from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from query import AcademicQueryService
try:
    from llm_factory import create_default_llm
except ImportError:
    from .llm_factory import create_default_llm
import torch
import os
import glob
import re

# 固定的MD文档路径（与OCR处理器一致）
MD_OUTPUT_FOLDER = "./md"


class RAGSystem:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.query_service = None

    def setup_llm(self, model_path="../llm/DeepSeek-R1-0528-Qwen3-8B"):
        """设置语言模型"""
        print("正在加载 DashScope qwen-plus 语言模型...")
        self.llm = create_default_llm()
        print("语言模型加载完成")

        # 初始化查询处理器
        self.setup_query_service()

    def setup_query_service(self):
        """设置查询处理器"""
        print("正在初始化查询处理器...")
        self.query_service = AcademicQueryService(self.llm)
        print("查询处理器初始化完成")

    def setup_embeddings(self, embedding_model_path="/root/.cache/modelscope/hub/models/BAAI/bge-m3"):
        """设置嵌入模型"""
        print("正在加载嵌入模型...")

        try:
            if os.path.exists(embedding_model_path):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_path,
                    model_kwargs={
                        'device': 'cuda:1',  # 指定使用第二张GPU
                        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'trust_remote_code': True,
                    },
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 2,
                    }
                )
                # 测试嵌入模型
                test_emb = self.embeddings.embed_documents(["测试"])
                if test_emb and len(test_emb[0]) > 0:
                    print("嵌入模型加载成功")
                    return

            print("使用随机嵌入")
            self.embeddings = FakeEmbeddings(size=384)

        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            self.embeddings = FakeEmbeddings(size=384)

    def load_md_documents(self):
        """从固定路径加载所有MD文档"""
        print("正在加载MD文档...")

        if not os.path.exists(MD_OUTPUT_FOLDER):
            print(f"MD文档文件夹不存在: {MD_OUTPUT_FOLDER}")
            return []

        md_files = glob.glob(os.path.join(MD_OUTPUT_FOLDER, "*.md"))
        if not md_files:
            print(f"在 '{MD_OUTPUT_FOLDER}' 中未找到MD文件")
            return []

        print(f"找到 {len(md_files)} 个MD文件")
        all_documents = []

        for md_file in md_files:
            try:
                loader = TextLoader(md_file, encoding='utf-8')
                documents = loader.load()

                for doc in documents:
                    doc.metadata['source'] = os.path.basename(md_file)

                all_documents.extend(documents)
                print(f"成功加载: {os.path.basename(md_file)}")

            except Exception as e:
                print(f"加载失败 {md_file}: {e}")
                continue

        return all_documents

    def retriever_vector_store(self):
        """判断向量库是否存在"""
        if os.path.exists("./faiss"):
            try:
                from langchain_community.vectorstores import FAISS
                self.vectorstore = FAISS.load_local(
                    "./faiss",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"加载FAISS失败，将重新创建: {e}")

    def _is_academic_content(self, text):
        """判断是否为有效的学术内容"""
        # 过滤掉页码、页眉页脚等
        exclusion_patterns = [
            r'^\d+$',  # 纯数字
            r'^[ivxlcdm]+$',  # 罗马数字
            r'^(abstract|keywords|references?)$',  # 章节标题单独成块
            r'^.*\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{4}.*$',  # 日期
        ]

        for pattern in exclusion_patterns:
            if re.match(pattern, text.lower().strip()):
                return False

        # 应该包含足够的实质性内容
        words = len(text.split())
        return words >= 5  # 至少5个词

    def setup_vector_store_optimized_fallback(self, documents):
        """针对学术论文的优化传统切块（回退方案）"""

        def clean_text_content(text):
            if not text or not text.strip():
                return ""
            cleaned = re.sub(r'\s+', ' ', text.strip())
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_text_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        # 学术论文需要更大的块大小
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增大以容纳完整段落
            chunk_overlap=200,  # 增加重叠保证上下文
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "．", "；", "，", " "]
        )

        texts = text_splitter.split_documents(documents)
        processed_texts = []

        for text in texts:
            content = clean_text_content(text.page_content)
            if content and len(content.strip()) >= 50 and self._is_academic_content(content):
                text.page_content = content
                processed_texts.append(text)

        print(f"传统切块完成，共 {len(processed_texts)} 个文本块")

        try:
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("向量数据库创建成功")
            return vectorstore
        except Exception as e:
            print(f"向量数据库创建失败: {e}")
            return None

    def setup_vector_store_semantic_arxiv(self, documents):
        """专门优化arXiv论文的语义切块"""
        if not documents:
            print("没有文档可处理")
            return None,[]

        # 针对学术论文的文本清理
        def clean_arxiv_content(text):
            if not text or not text.strip():
                return ""

            # 保留数学公式标记
            cleaned = re.sub(r'\s+', ' ', text.strip())
            # 保留常见的数学和环境标记
            cleaned = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', lambda m: m.group(0).replace('\n', ' '), cleaned)
            cleaned = re.sub(r'\$\$.*?\$\$', lambda m: m.group(0).replace('\n', ' '), cleaned)
            cleaned = re.sub(r'\$.*?\$', lambda m: m.group(0).replace('\n', ' '), cleaned)

            # 清理控制字符但保留重要标记
            cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_arxiv_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        try:
            # 修复：使用正确的SemanticChunker参数
            text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,  # 控制块大小的主要参数
            )

            texts = text_splitter.split_documents(documents)

            # 学术论文特定的后处理 - 手动控制块大小
            processed_texts = []
            for text in texts:
                content = clean_arxiv_content(text.page_content)
                if not content:
                    continue

                # 手动过滤过小的块
                if len(content.strip()) < 50:  # 学术文本需要更长的最小长度
                    continue

                # 手动处理过大的块 - 进行二次分割
                if len(content) > 1200:
                    # 对过大的块使用传统分割器进行二次分割
                    secondary_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        separators=["\n", "。", "！", "？", "；", "，", " "]
                    )
                    sub_chunks = secondary_splitter.split_text(content)
                    for i, sub_chunk in enumerate(sub_chunks):
                        if len(sub_chunk.strip()) >= 50 and self._is_academic_content(sub_chunk):
                            sub_doc = text.copy()
                            sub_doc.page_content = sub_chunk
                            processed_texts.append(sub_doc)
                else:
                    # 检查是否是有效的学术内容
                    if self._is_academic_content(content):
                        text.page_content = content
                        processed_texts.append(text)

            print(f"语义切块完成，共 {len(processed_texts)} 个语义块")

            # 分析块大小分布
            sizes = [len(t.page_content) for t in processed_texts]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                print(f"平均块大小: {avg_size:.0f} 字符")
                print(f"最小/最大块: {min(sizes)}/{max(sizes)} 字符")

            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("向量数据库创建成功")
            return vectorstore

        except Exception as e:
            print(f"语义切块失败: {e}")
            # 回退到优化的传统方法
            return self.setup_vector_store_optimized_fallback(documents)

    def get_all_documents_from_faiss(self):
        """从FAISS获取所有文档块，用于BM25检索器"""
        if self.vectorstore is None:
            print("FAISS向量库未初始化")
            return []

        documents = []
        # 遍历FAISS的docstore获取所有文档
        for doc_id, doc in self.vectorstore.docstore._dict.items():
            documents.append(doc)

        print(f"从FAISS提取了 {len(documents)} 个文档块")
        return documents

    def create_bm25_retriever_from_faiss(self, k=5):
        """基于FAISS中的文档创建BM25检索器"""
        documents = self.get_all_documents_from_faiss()
        if not documents:
            print("无法创建BM25检索器：没有文档")
            return None

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25_retriever.k = k
            print("BM25检索器创建成功（基于FAISS文档）")
            return bm25_retriever
        except Exception as e:
            print(f"BM25检索器创建失败: {e}")
            return None

    def setup_hybrid_retriever(self,bm25_weight=0.4, vector_weight=0.6):
        """设置BM25+余弦相似度的混合检索器"""
        print("正在设置混合检索器...")

        try:
            # 创建BM25检索器
            bm25_retriever = self.create_bm25_retriever_from_faiss(k=5)

            vector_retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # 使用余弦相似度
                search_kwargs={"k": 5, "score_threshold": 0.7}
            )

            # 3. 创建混合检索器
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, vector_weight]  # 可调整权重
            )

            print("混合检索器设置完成")
            return self.ensemble_retriever

        except Exception as e:
            print(f"混合检索器设置失败: {e}")
            # 回退到向量检索
            return self.setup_fallback_retriever()

    def setup_fallback_retriever(self):
        """回退检索器设置"""
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def setup_rag_chain(self):
        """设置RAG链"""
        rag_prompt_template = """基于以下提供的多个文档内容，请用中文回答用户的问题。回答应该准确、详细且基于文档事实。

        文档内容：
        {context}

        用户问题：{question}

        请严格按照以下要求回答：
        1. 基于提供的文档内容提供答案，不要编造信息
        2. 如果文档内容不足以回答问题，请明确说明
        3. 答案应该结构清晰，分段表述
        4. 引用文档中的具体信息来支持你的回答
        5. 如果信息来自多个文档，请说明信息来源

        回答："""

        prompt = PromptTemplate.from_template(rag_prompt_template)

        def format_docs(docs):
            if not docs:
                return "未找到相关文档内容"
            formatted = []
            for i, doc in enumerate(docs, 1):
                clean_content = doc.page_content.replace('\n', ' ').strip()
                if not clean_content:
                    continue
                source = doc.metadata.get('source', '未知文件')
                formatted.append(f"[文档 {i} - {source}]\n{clean_content}")
            return "\n\n".join(formatted) if formatted else "未找到相关文档内容"

        self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def enhanced_ask_question(self, question, show_docs=True):
        """增强的问题回答方法，使用单查询 QueryPlan 进行检索与回答。"""
        print("正在处理单查询学术检索...")

        plan = self.query_service.build_query_plan(question)
        search_query = plan.retrieval_query_zh or question
        relevant_docs = self.retriever.invoke(search_query)

        if show_docs:
            if relevant_docs:
                print(f"检索到 {len(relevant_docs)} 个相关文档:")
                for j, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('source', '未知文件')
                    content_preview = doc.page_content.replace('\n', ' ')[:120]
                    print(f"   文档 {j} [{source}]: {content_preview}...")
            else:
                print("未检索到相关文档")

        if not relevant_docs:
            return self.ask_question(question, show_docs=show_docs, use_enhanced=False)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        answer = self.query_service.generate_answer(context, question)
        return answer, relevant_docs

    def ask_question(self, question, show_docs=True, use_enhanced=True):
        """回答问题"""
        if use_enhanced and self.query_service:
            return self.enhanced_ask_question(question, show_docs)

        # 原始逻辑保持不变
        try:
            if show_docs:
                search_query = question
                if self.query_service:
                    try:
                        search_query = self.query_service.build_query_plan(question).retrieval_query_zh or question
                    except Exception:
                        search_query = question
                relevant_docs = self.retriever.invoke(search_query)
                if relevant_docs:
                    print(f"检索到 {len(relevant_docs)} 个相关文档:")
                    for j, doc in enumerate(relevant_docs, 1):
                        source = doc.metadata.get('source', '未知文件')
                        content_preview = doc.page_content.replace('\n', ' ')[:120]
                        print(f"   文档 {j} [{source}]: {content_preview}...")
                else:
                    print("未检索到相关文档")

            answer = self.rag_chain.invoke(question)
            return answer, relevant_docs if 'relevant_docs' in locals() else []

        except Exception as e:
            print(f"处理问题时出错: {e}")
            return None, None

    def initialize(self,hybrid_weights=(0.4, 0.6)):
        """初始化整个RAG系统"""
        self.setup_llm()
        self.setup_embeddings()

        # documents = self.load_md_documents()
        # if not documents:
        #     print("没有加载到文档，RAG系统将使用空数据库")
        #     return False
        self.retriever_vector_store()

        if self.vectorstore is None:
            return False

        self.retriever = self.setup_hybrid_retriever(
            bm25_weight=hybrid_weights[0],
            vector_weight=hybrid_weights[1]
        )

        self.setup_rag_chain()
        return True

    def update_rag_system(self,chunk_strategy = "semantic_arxiv",hybrid_weights=(0.4, 0.6)):
        #处理md文件更新faiss库
        documents = self.load_md_documents()
        if chunk_strategy == "semantic_arxiv":
            self.vectorstore = self.setup_vector_store_semantic_arxiv(documents)
        elif chunk_strategy == "optimized":
            self.vectorstore= self.setup_vector_store_optimized_fallback(documents)
        else:
            self.vectorstore = self.setup_vector_store_semantic_arxiv(documents)  # 默认
        if self.vectorstore is None:
            return False

        self.retriever = self.setup_hybrid_retriever(
            bm25_weight=hybrid_weights[0],
            vector_weight=hybrid_weights[1]
        )

        self.setup_rag_chain()


# # 创建全局RAG系统实例
_global_rag_system = None


def setup_rag_system():
    """设置并返回RAG系统实例"""
    global _global_rag_system
    if _global_rag_system is None:
        _global_rag_system = RAGSystem()
        success = _global_rag_system.initialize()
        if not success:
            print("RAG系统初始化失败")
            return None
    return _global_rag_system



# def get_rag_system():
#     """获取RAG系统实例"""
#     return _global_rag_system
