import os
import requests
import re
import math
import csv
from bs4 import BeautifulSoup
import time
from lxml import html
from typing import List, Dict, Optional


class ArxivCrawlerIntegrated:
    """集成式Arxiv论文爬虫管理器"""

    def __init__(self, output_dir: str = "./paper_results"):
        """初始化爬虫

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.all_papers = []  # 存储爬取的所有论文
        self._ensure_directories()

    def _ensure_directories(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_search_query(self, user_question: str) -> str:
        """根据用户问题生成搜索查询

        Args:
            user_question: 用户问题或关键词

        Returns:
            生成的搜索查询字符串
        """
        # 简单的关键词提取（可以替换为更复杂的NLP方法）
        keywords = re.findall(r'\b[a-zA-Z]{4,}\b', user_question)
        if keywords:
            # 取前3个关键词
            main_keywords = keywords[:3]
            query = " OR ".join(main_keywords)
        else:
            # 默认查询
            query = "multimodal self-growing system"

        return query

    def build_search_url(self, query: str, start: int = 0, size: int = 50) -> str:
        """构建搜索URL

        Args:
            query: 搜索查询
            start: 起始位置
            size: 每页大小

        Returns:
            完整的搜索URL
        """
        encoded_query = requests.utils.quote(query)
        return f"https://arxiv.org/search/?query={encoded_query}&searchtype=abstract&abstracts=show&order=-announced_date_first&size={size}&start={start}"

    def get_total_results(self, url: str) -> int:
        """获取总结果数

        Args:
            url: 搜索URL

        Returns:
            总结果数
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            tree = html.fromstring(response.content)
            result_string = ''.join(tree.xpath('//*[@id="main-container"]/div[1]/div[1]/h1/text()')).strip()
            match = re.search(r'of ([\d,]+) results', result_string)
            return int(match.group(1).replace(',', '')) if match else 0
        except Exception as e:
            print(f"获取总结果数失败: {e}")
            return 0

    def fetch_paper_info(self, url: str) -> List[Dict]:
        """获取单页论文信息

        Args:
            url: 页面URL

        Returns:
            论文信息列表
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = []

            for article in soup.find_all('li', class_='arxiv-result'):
                try:
                    title = article.find('p', class_='title').text.strip()
                    authors_text = article.find('p', class_='authors').text.replace('Authors:', '').strip()
                    authors = [author.strip() for author in authors_text.split(',')]
                    abstract = article.find('span', class_='abstract-full').text.strip()
                    submitted = article.find('p', class_='is-size-7').text.strip()
                    submission_date = submitted.split(';')[0].replace('Submitted', '').strip()

                    pdf_link_element = article.find('a', string='pdf')
                    pdf_link = pdf_link_element['href'] if pdf_link_element else 'No PDF link found'

                    papers.append({
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'submission_date': submission_date,
                        'pdf_link': pdf_link
                    })
                except Exception as e:
                    print(f"解析单篇论文信息失败: {e}")
                    continue

            return papers
        except Exception as e:
            print(f"获取论文信息失败: {e}")
            return []

    def crawl_papers(self, user_question: str, max_pages: int = 5) -> List[Dict]:
        """根据用户问题爬取相关论文

        Args:
            user_question: 用户问题或关键词
            max_pages: 最大爬取页数

        Returns:
            论文信息列表
        """
        print("正在根据问题搜索相关论文...")

        # 生成搜索查询
        search_query = self.generate_search_query(user_question)
        print(f"生成的搜索查询: {search_query}")

        base_url = self.build_search_url(search_query)
        total_results = self.get_total_results(base_url)

        if total_results == 0:
            print("未找到相关论文，尝试使用默认查询...")
            search_query = "multimodal self-growing system"
            base_url = self.build_search_url(search_query)
            total_results = self.get_total_results(base_url)
            if total_results == 0:
                print("默认查询也未找到论文")
                return []

        total_pages = min(math.ceil(total_results / 50), max_pages)
        self.all_papers = []

        for page in range(total_pages):
            start = page * 50
            print(f"爬取页面 {page + 1}/{total_pages}, 起始位置: {start}")

            page_url = self.build_search_url(search_query, start=start)
            papers = self.fetch_paper_info(page_url)
            self.all_papers.extend(papers)

            time.sleep(2)  # 礼貌性延迟

        print(f"爬取完成！共获取 {len(self.all_papers)} 篇相关论文")
        return self.all_papers

    def save_to_csv(self, papers: List[Dict] = None, filename: str = "paper_result.csv") -> bool:
        """将论文信息保存到CSV文件

        Args:
            papers: 论文列表，默认为self.all_papers
            filename: 文件名

        Returns:
            保存是否成功
        """
        if papers is None:
            papers = self.all_papers

        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'authors', 'abstract', 'submission_date', 'pdf_link']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in papers:
                    writer.writerow(paper)
            print(f"论文信息已保存到 {filepath}")
            return True
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            return False

    def read_csv(self, filename: str) -> List[Dict]:
        """从CSV文件读取论文信息

        Args:
            filename: 文件名

        Returns:
            论文信息列表
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            papers = []
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    title = row['title']
                    submission = row['submission_date']
                    pdf_link = row['pdf_link']
                    papers.append({
                        'title': title,
                        'submission_date': submission,
                        'pdf_link': pdf_link
                    })
            return papers
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return []

    def extract_year(self, submission: str) -> str:
        """从提交日期中提取年份

        Args:
            submission: 提交日期字符串

        Returns:
            年份字符串
        """
        match = re.search(r'\d{4}', submission)
        return match.group(0) if match else 'Unknown'

    def format_paper(self, paper: Dict) -> str:
        """格式化单篇论文信息

        Args:
            paper: 论文信息字典

        Returns:
            格式化后的字符串
        """
        title = paper['title']
        year = self.extract_year(paper['submission_date'])
        pdf_link = paper['pdf_link']
        return f"+ {title}, arxiv {year}, [[paper]]({pdf_link})."

    def generate_paper_list(self, filename: str) -> List[str]:
        """从CSV文件生成格式化论文列表

        Args:
            filename: CSV文件名

        Returns:
            格式化后的论文列表
        """
        papers = self.read_csv(filename)
        return [self.format_paper(paper) for paper in papers]

    def save_formatted_papers(self, papers: List[str] = None,
                              filename: str = "formatted_papers.txt") -> bool:
        """保存格式化后的论文列表到文件

        Args:
            papers: 论文列表，默认为self.all_papers的格式化结果
            filename: 输出文件名

        Returns:
            保存是否成功
        """
        if papers is None:
            papers = self.generate_paper_list("paper_result.csv")

        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                for paper in papers:
                    f.write(paper + '\n')
            print(f"格式化论文列表已保存到 {filepath}")
            return True
        except Exception as e:
            print(f"保存格式化论文失败: {e}")
            return False

    def extract_papers_from_file(self, filename: str) -> List[Dict]:
        """从文本文件中提取论文信息

        Args:
            filename: 文件名

        Returns:
            论文信息列表
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            paper_data = []
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # 匹配模式：标题和论文链接
                pattern = r'\+\s(.+?),\s(?:arxiv|arXiv)\s\d{4},\s\[\[paper\]\]\((https://arxiv\.org/pdf/[^)]+)\)\.'
                matches = re.findall(pattern, content)
                for title, paper_link in matches:
                    paper_data.append({
                        'title': title.strip(),
                        'paper_link': paper_link
                    })
            print(f"从文件中成功提取 {len(paper_data)} 篇论文")
            return paper_data
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return []

    def _clean_filename(self, filename: str) -> str:
        """清理文件名

        Args:
            filename: 原始文件名

        Returns:
            清理后的文件名
        """
        illegal_chars = r'[<>:"/\\|?*]'
        clean_name = re.sub(illegal_chars, '', filename)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        # 限制文件名长度
        if len(clean_name) > 80:
            clean_name = clean_name[:80]
        return clean_name

    def download_paper(self, paper_link: str, filepath: str) -> bool:
        """下载单个论文文件

        Args:
            paper_link: 论文PDF链接
            filepath: 保存路径

        Returns:
            下载是否成功
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            # 设置超时
            response = requests.get(paper_link, headers=headers, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"下载失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"下载出错: {e}")
            return False

    def download_papers(self, papers: List[Dict] = None,
                        max_downloads: int = 10,
                        source: str = "memory") -> int:
        """批量下载论文PDF

        Args:
            papers: 论文列表，默认为self.all_papers
            max_downloads: 最大下载数量
            source: 论文来源，可选"memory"（内存）或"file"（从格式化文件读取）

        Returns:
            成功下载的论文数量
        """
        if papers is None:
            if source == "file":
                papers = self.extract_papers_from_file("formatted_papers.txt")
            else:
                papers = self.all_papers[:max_downloads]
        else:
            papers = papers[:max_downloads]

        if not papers:
            print("没有论文可下载")
            return 0

        print(f"准备下载 {len(papers)} 篇论文")
        success_count = 0

        for i, paper in enumerate(papers, 1):
            # 确定论文链接字段名
            paper_link = paper.get('pdf_link') or paper.get('paper_link')
            if not paper_link or paper_link == 'No PDF link found':
                print(f"跳过无PDF链接的论文: {paper['title']}")
                continue

            print(f"\n[{i}/{len(papers)}] 正在下载: {paper['title']}")

            # 清理文件名
            clean_title = self._clean_filename(paper['title'])
            filename = f"{clean_title}.pdf"
            filepath = os.path.join(self.output_dir, filename)

            # 检查文件是否已存在
            if os.path.exists(filepath):
                print(f"文件已存在，跳过: {filename}")
                success_count += 1
                continue

            # 下载论文
            start_time = time.time()
            if self.download_paper(paper_link, filepath):
                end_time = time.time()
                download_time = end_time - start_time
                print(f"✓ 下载成功: {filename} (耗时: {download_time:.2f}秒)")
                success_count += 1
            else:
                print(f"✗ 下载失败: {filename}")

            time.sleep(1)  # 下载间隔

        print(f"\n下载完成！成功下载 {success_count}/{len(papers)} 篇论文")
        return success_count

    def run_full_workflow(self, user_question: str,
                          max_pages: int = 5,
                          max_downloads: int = 10) -> Dict[str, any]:
        """运行完整工作流程：爬取→格式化→下载

        Args:
            user_question: 用户问题或关键词
            max_pages: 最大爬取页数
            max_downloads: 最大下载数量

        Returns:
            包含各步骤结果的字典
        """
        result = {}

        # 第一步：爬取论文
        papers = self.crawl_papers(user_question, max_pages)
        result['crawled_papers'] = len(papers)

        # 保存到CSV
        if papers:
            self.save_to_csv(papers)

            # 第二步：格式化论文
            formatted_papers = self.generate_paper_list("paper_result.csv")
            self.save_formatted_papers(formatted_papers)
            result['formatted_papers'] = len(formatted_papers)

            # 第三步：下载论文
            success_downloads = self.download_papers(max_downloads=max_downloads)
            result['downloaded_papers'] = success_downloads

        return result


def create_arxiv_crawler_integrated(output_dir: str = "./paper_results") -> ArxivCrawlerIntegrated:
    """创建集成式Arxiv爬虫实例

    Args:
        output_dir: 输出目录路径

    Returns:
        ArxivCrawlerIntegrated实例
    """
    return ArxivCrawlerIntegrated(output_dir)


# 使用示例
if __name__ == "__main__":
    # 创建爬虫实例
    crawler = create_arxiv_crawler_integrated("./my_papers")

    # 方法1：运行完整流程
    result = crawler.run_full_workflow(
        user_question="multimodal self-growing system",
        max_pages=2,
        max_downloads=5
    )
    print(f"完整流程结果: {result}")

    # 方法2：分步执行
    # 1. 爬取论文
    # papers = crawler.crawl_papers("machine learning", max_pages=3)
    #
    # 2. 保存到CSV
    # crawler.save_to_csv(papers, "ml_papers.csv")
    #
    # 3. 格式化论文
    # formatted = crawler.generate_paper_list("ml_papers.csv")
    # crawler.save_formatted_papers(formatted, "formatted_ml_papers.txt")
    #
    # 4. 下载论文
    # success = crawler.download_papers(max_downloads=3)
    # print(f"成功下载 {success} 篇论文")