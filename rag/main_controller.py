# main_controller.py (修改后的版本)
import os
import time
from pdf_processor import PDFProcessor
from rag_system import setup_rag_system
# from arxiv_crawler import create_arxiv_crawler
from arxiv_crawler_integrated import create_arxiv_crawler_integrated

class OCRRAGController:
    def __init__(self, pdf_folder_path):
        self.pdf_folder_path = pdf_folder_path
        self.md_folder_path = "./md"
        self.rag_system = None
        self.last_processed_time = 0
        self.arxiv_crawler = None
        self.last_crawl_time = 0
        self.crawl_cooldown = 3600  # 1小时冷却时间，避免频繁爬取
        self.use_enhanced = False
        self.process_pdf_folder = None

    def setup_pdf_processor(self):
        """设置ocr模型"""
        self.process_pdf_folder = PDFProcessor(
            output_dir=self.md_folder_path,  # 输出目录
            lang='en',  # 语言设置 ('ch' 或 'en')
            dpi=220  # OCR分辨率
        )

    def setup_arxiv_crawler(self):
        """设置Arxiv爬虫"""
        # self.arxiv_crawler = create_arxiv_crawler(self.pdf_folder_path)
        self.arxiv_crawler = create_arxiv_crawler_integrated(self.pdf_folder_path)

    def fetch_latest_papers(self,question, max_papers=5):
        """获取最新论文并下载"""
        if self.arxiv_crawler is None:
            self.setup_arxiv_crawler()

        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_crawl_time < self.crawl_cooldown:
            print("爬虫功能冷却中，请稍后再试")
            return False

        try:
            print("=" * 60)
            print("开始从Arxiv获取最新论文...")
            print("=" * 60)

            # 爬取论文信息
            papers = self.arxiv_crawler.crawl_papers(question,max_pages=2)
            if not papers:
                print("未获取到论文信息")
                return False
            # self.arxiv_crawler.save_to_csv(papers)  # 保存CSV
            # self.arxiv_crawler.save_formatted_papers()  # 保存TXT
            # 下载论文PDF
            success_count = self.arxiv_crawler.download_papers(
                papers=papers,
                max_downloads=max_papers
            )

            if success_count > 0:
                print(f"成功下载 {success_count} 篇新论文")
                self.last_crawl_time = current_time
                return True
            else:
                print("没有成功下载任何论文")
                return False

        except Exception as e:
            print(f"获取论文失败: {e}")
            return False

    def check_for_pdfs_and_mds(self):
        """检查是否有新的PDF文件需要处理"""
        pdf_folder_exit = True
        md_folder_exit = True
        if not os.path.exists(self.pdf_folder_path):
            pdf_folder_exit = False
            print(f"PDF文件夹不存在: {self.pdf_folder_path}")
        if not os.path.exists(self.md_folder_path):
            md_folder_exit = False
            print(f"MD文件夹不存在: {self.md_folder_path}")
        if not pdf_folder_exit or not md_folder_exit:
            return [],[]

        pdf_files = [f for f in os.listdir(self.pdf_folder_path)
                     if f.lower().endswith('.pdf')]
        md_files = [f for f in os.listdir(self.md_folder_path)
                     if f.lower().endswith('.md')]
        return pdf_files,md_files

    def process_all_pdfs(self):
        """处理所有PDF文件并更新RAG系统"""
        print("开始处理PDF文件夹...")

        # 处理PDF文件夹
        processed_files = self.process_pdf_folder.process_pdf_folder(self.pdf_folder_path)

        if processed_files:
            print("PDF处理完成，正在更新RAG系统...")
            # 重新初始化RAG系统
            self.rag_system.update_rag_system()
            self.last_processed_time = time.time()
            print("RAG系统更新完成!")
        else:
            print("没有找到可处理的PDF文件")

        return processed_files


    ##需要改进，相关性的改进
    def evaluate_relevance(self, retrieved_docs):
        """评估检索到的文档与问题的相关性"""
        if not retrieved_docs:
            return False, "未检索到相关文档"

        # 简单评估：检查文档数量和质量
        relevant_count = 0
        total_content_length = 0

        for doc in retrieved_docs:
            content = doc.page_content
            # 简单的关键词匹配（可以替换为更复杂的语义匹配）
            if len(content) > 50:  # 确保文档有足够内容
                total_content_length += len(content)
                relevant_count += 1

        # 判断标准：至少有1个相关文档，且总内容长度足够
        is_relevant = relevant_count >= 1 and total_content_length > 200

        reason = f"检索到 {relevant_count} 个相关文档，总内容长度 {total_content_length} 字符"

        return is_relevant, reason

    def ask_question_with_fallback(self, question, show_docs=True):
        """带回落机制的问答：如果相关性不足，自动触发爬虫"""
        if self.rag_system is None:
            print("RAG系统未初始化")
            return "系统未就绪，请先初始化RAG系统", []
        try:
            # 第一次尝试：使用现有知识库回答
            print("正在检索现有文档...")
            answer, relevant_docs = self.rag_system.ask_question(
                question,
                show_docs=show_docs,
                use_enhanced=self.use_enhanced
            )

            # 评估相关性
            is_relevant, reason = self.evaluate_relevance(relevant_docs)
            is_relevant = True
            if is_relevant:
                print(f"文档相关性评估：通过 ({reason})")
                return answer, relevant_docs
            else:
                print(f"文档相关性评估：不足 ({reason})")
                print("正在尝试获取最新研究论文...")

                # 触发爬虫获取新论文
                if self.fetch_latest_papers(question,max_papers=3):
                    # 处理新论文并更新RAG系统
                    self.process_all_pdfs()

                    # 第二次尝试：使用更新后的知识库回答
                    print("使用更新后的知识库重新回答问题...")
                    answer, docs = self.rag_system.ask_question(
                        question,
                        show_docs=show_docs,
                        use_enhanced=self.use_enhanced
                    )

                    # 再次评估
                    is_relevant, reason = self.evaluate_relevance(question, docs)
                    if is_relevant:
                        print(f"更新后文档相关性评估：通过 ({reason})")
                    else:
                        print(f"更新后文档相关性评估：仍然不足 ({reason})")
                        answer = f"{answer}\n\n注意：即使获取了最新论文，相关信息仍然有限。"

                    return answer, docs
                else:
                    return "无法获取最新论文，请检查网络连接或稍后重试", []

        except Exception as e:
            print(f"处理问题时出错: {e}")
            return f"处理问题时出错: {e}", []

    def run_interactive(self):
        """运行交互式问答系统"""
        # 初始化爬虫
        self.setup_arxiv_crawler()
        # 初始化ocr
        self.setup_pdf_processor()
        # 检查PDF文件夹状态
        pdf_files,md_files = self.check_for_pdfs_and_mds()
        if not pdf_files and not md_files:
            print("未找到pdf和md文件，需要从Arxiv获取初始论文库")
            self.fetch_latest_papers("multimodal self-growing system", max_papers=5)
        elif not pdf_files:
            print(f"PDF文件夹为空，但找到 {len(md_files)} 个MD文件，将使用现有MD文件")
        elif not md_files:
            print(f"MD文件夹为空，但找到 {len(pdf_files)} 个PDF文件，将处理这些PDF")


        # 初始化RAG系统
        if self.rag_system is None:
            print("正在初始化RAG系统...")
            self.rag_system = setup_rag_system()
            self.last_processed_time = time.time()

        if self.rag_system is None:
            print("RAG系统初始化失败，无法启动问答")
            return

        print("\n" + "=" * 60)
        print("输入 '退出' 来结束程序")
        # print("输入 '手动更新' 来强制获取最新论文")
        # print("输入 '刷新' 来重新处理现有PDF文件")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n请输入问题: ").strip()

                if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                    print("感谢使用!")
                    break
                # elif user_input.lower() in ['手动更新', 'update']:
                #     print("开始手动更新论文库...")
                #     if self.fetch_latest_papers(max_papers=5):
                #         self.process_all_pdfs()
                #     continue
                # elif user_input.lower() in ['刷新', 'refresh', 'reload']:
                #     print("重新处理PDF文件...")
                #     self.process_all_pdfs()
                #     continue
                # elif user_input.lower() in ['模式', 'mode']:
                #     enhanced_mode = not enhanced_mode
                #     mode_name = "增强模式" if enhanced_mode else "简单模式"
                #     print(f"已切换到{mode_name}")
                #     continue

                if not user_input:
                    continue

                # 使用智能问答系统（自动触发爬虫回落机制）
                print("正在分析问题并生成答案...")
                answer, docs = self.ask_question_with_fallback(user_input, show_docs=True)
                print(f"答案:\n{answer}")

            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")


def main():
    # 配置PDF文件夹路径
    PDF_FOLDER = "./pdf"

    # 确保PDF文件夹存在
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs("./documents", exist_ok=True)

    # 创建控制器
    controller = OCRRAGController(PDF_FOLDER)

    # 启动系统
    controller.run_interactive()


if __name__ == "__main__":
    main()