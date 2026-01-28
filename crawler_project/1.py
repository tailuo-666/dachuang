import argparse
import time
import os
import re
import requests
from langchain_community.retrievers import ArxivRetriever

# ================= 配置区 =================
SUMMARY_DIR = "arxiv_summary_notes"
PDF_DIR = r"D:\论文爬取"


def setup_directories():
    """创建需要的文件夹"""
    for d in [SUMMARY_DIR, PDF_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📁 已创建文件夹: {d}")


def sanitize_filename(name):
    """清洗文件名"""
    name = re.sub(r'\s+', '_', name)
    return re.sub(r'[\\/*?:"<>|]', "", name)


def extract_arxiv_id(entry_id):
    """从Entry ID中提取arXiv ID"""
    if not entry_id:
        return None

    entry_str = str(entry_id)

    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arXiv:(\d+\.\d+)',
        r'arxiv\.org/abs/([a-z\-]+/\d+\.\d+)',
        r'arxiv\.org/pdf/([a-z\-]+/\d+\.\d+)',
        r'arXiv:([a-z\-]+/\d+\.\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, entry_str)
        if match:
            arxiv_id = match.group(1)
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
            return arxiv_id

    return None


def download_pdf(arxiv_id, title):
    """下载PDF文件"""
    if not arxiv_id:
        return False, None

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    clean_arxiv_id = arxiv_id.replace('/', '_')
    safe_title = sanitize_filename(title)[:80]
    filename = f"{clean_arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(PDF_DIR, filename)

    print(f"   📥 下载PDF: {arxiv_id}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf"
        }

        response = requests.get(pdf_url, headers=headers, timeout=60)

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)

            file_size = os.path.getsize(filepath) / 1024
            print(f"   ✅ PDF下载完成 ({file_size:.1f}KB)")
            return True, filepath
        else:
            print(f"   ❌ HTTP错误 {response.status_code}")
            return False, None

    except Exception as e:
        print(f"   ❌ 下载失败: {str(e)}")
        return False, None


def save_summary(doc, query, download_pdf_option=False):
    """保存摘要笔记"""
    safe_query = sanitize_filename(query)

    # 获取元数据
    title = doc.metadata.get('Title', 'No Title').replace('\n', ' ')
    authors = doc.metadata.get('Authors', 'Unknown')
    published = doc.metadata.get('Published', 'Unknown')
    entry_id = doc.metadata.get('Entry ID', '')

    # 提取arXiv ID
    arxiv_id = extract_arxiv_id(entry_id)

    safe_title = sanitize_filename(title)[:50]
    filename = f"{safe_query}_{safe_title}_summary.md"
    filepath = os.path.join(SUMMARY_DIR, filename)

    # PDF下载
    pdf_saved = False
    pdf_path = ""
    if download_pdf_option and arxiv_id:
        pdf_success, pdf_path = download_pdf(arxiv_id, title)
        pdf_saved = pdf_success

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# ArXiv Summary Note: {title}\n\n")
        f.write(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Authors:** {authors}\n")
        f.write(f"- **Published:** {published}\n")

        # ========== 原始Entry ID展示 ==========
        f.write(f"\n**📌 原始Entry ID (直接从ArxivRetriever获取):**\n")
        if entry_id:
            f.write(f"> `{entry_id}`\n\n")

            # 显示Entry ID的基本信息
            f.write(f"**Entry ID分析:**\n")
            f.write(f"- **长度:** {len(entry_id)} 字符\n")
            f.write(f"- **类型:** `{type(entry_id).__name__}`\n")
        else:
            f.write(f"> `(空字符串或无Entry ID)` ⚠️\n")

        # arXiv标识符
        f.write(f"\n**📌 论文标识符:**\n")
        if arxiv_id:
            f.write(f"- **arXiv ID:** `{arxiv_id}`\n")
            f.write(f"- **摘要链接:** [arxiv.org/abs/{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n")
            f.write(f"- **PDF链接:** [arxiv.org/pdf/{arxiv_id}.pdf](https://arxiv.org/pdf/{arxiv_id}.pdf)\n")
        else:
            f.write(f"- **arXiv ID:** 未找到\n")
            f.write(f"- **原始Entry ID:** {entry_id[:100] if entry_id else '无'}\n")

        # PDF状态
        if download_pdf_option:
            f.write(f"\n**📂 PDF下载状态:**\n")
            if pdf_saved and pdf_path:
                abs_path = os.path.abspath(pdf_path)
                file_size = os.path.getsize(pdf_path) / 1024
                f.write(f"- ✅ 已下载: `{os.path.basename(pdf_path)}` ({file_size:.1f}KB)\n")
                f.write(f"- 📍 路径: `{abs_path}`\n")
            else:
                f.write(f"- ❌ 下载失败\n")

        f.write("\n---\n\n")
        f.write("### 📝 Abstract (摘要)\n\n")
        f.write(f"> {doc.page_content}\n")

        # 统计
        char_count = len(doc.page_content)
        word_count = len(doc.page_content.split())
        f.write(f"\n*统计: {word_count} 词, {char_count} 字符*\n")

    return filepath, pdf_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--download-pdf', action='store_true', help="是否下载PDF")
    args = parser.parse_args()

    setup_directories()

    print("\n" + "=" * 50)
    print("🦜 ArXiv 论文摘要下载器")
    print("=" * 50)
    print(f"📁 摘要笔记: {SUMMARY_DIR}")
    print(f"📁 PDF保存: {PDF_DIR}")
    print("=" * 50)

    # 交互输入
    if not args.query:
        args.query = input("\n👉 搜索关键词: ").strip()
        if not args.query:
            print("❌ 关键词不能为空")
            return

    if not args.download_pdf:
        pdf_input = input("👉 是否下载PDF? (y/n, 默认n): ").strip().lower()
        args.download_pdf = (pdf_input == 'y')

    print(f"\n🚀 正在搜索: '{args.query}'")
    if args.download_pdf:
        print(f"📥 PDF下载: ✅ 开启")

    try:
        # 获取摘要
        retriever = ArxivRetriever(
            load_max_docs=args.count,
            get_full_documents=False
        )
        summary_docs = retriever.invoke(args.query)

        if not summary_docs:
            print("❌ 未找到相关论文。")
            return

        print(f"✅ 找到 {len(summary_docs)} 篇论文")

        # 保存摘要和PDF
        pdf_count = 0
        for i, doc in enumerate(summary_docs, 1):
            title = doc.metadata.get('Title', 'No Title')[:70]
            print(f"  [{i}] 处理: {title}")

            path, pdf_success = save_summary(doc, args.query, args.download_pdf)
            if pdf_success:
                pdf_count += 1
            print(f"    摘要已保存: {os.path.basename(path)}")

        # 完成信息
        print(f"\n" + "=" * 50)
        print("🎉 完成!")
        print(f"📄 摘要笔记目录: {os.path.abspath(SUMMARY_DIR)}")

        if args.download_pdf:
            print(f"📂 PDF保存目录: {os.path.abspath(PDF_DIR)}")
            print(f"📥 成功下载: {pdf_count}/{len(summary_docs)} 个PDF")

        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()