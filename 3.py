import os
import requests
import re
import time

# 目标文件夹
output_dir = r"C:\Users\温晋堂\Desktop\论文爬取"


# 从文本文件中提取论文信息
def extract_papers_from_file(file_path):
    """从文本文件中提取论文信息"""
    paper_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 匹配模式：标题和论文链接
        pattern = r'\+\s(.+?),\s(?:arxiv|arXiv)\s\d{4},\s\[\[paper\]\]\((https://arxiv\.org/pdf/[^)]+)\)'

        matches = re.findall(pattern, content)

        for title, paper_link in matches:
            paper_data.append({
                'title': title.strip(),
                'paper_link': paper_link
            })

        print(f"从文件中成功提取 {len(paper_data)} 篇论文")

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")

    return paper_data


# 清理文件名
def clean_filename(filename):
    """清理文件名，移除Windows不允许的字符"""
    illegal_chars = r'[<>:"/\\|?*]'
    clean_name = re.sub(illegal_chars, '', filename)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    # 限制文件名长度
    if len(clean_name) > 80:
        clean_name = clean_name[:80]
    return clean_name


# 下载文件函数
def download_paper(paper_link, filepath):
    """下载单个论文文件"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
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


# 主程序
def main():
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 读取论文文本文件
    file_path = "formatted_papers.txt"

    # 提取论文信息
    print("正在从文件中提取论文信息...")
    all_papers = extract_papers_from_file(file_path)

    if not all_papers:
        print("没有找到论文信息，请检查文件路径和内容格式")
        return

    # 只取前10篇
    papers_to_download = all_papers[:10]

    print(f"找到 {len(all_papers)} 篇论文，将下载前 {len(papers_to_download)} 篇")
    print("=" * 80)

    # 显示前10篇论文信息
    print("前10篇论文列表:")
    for i, paper in enumerate(papers_to_download, 1):
        print(f"{i:2d}. {paper['title']}")
    print("=" * 80)

    # 顺序下载论文
    success_count = 0

    for i, paper in enumerate(papers_to_download, 1):
        print(f"\n[{i}/{len(papers_to_download)}] 正在下载: {paper['title']}")

        # 清理文件名（只使用论文标题，不加前缀）
        clean_title = clean_filename(paper['title'])
        filename = f"{clean_title}.pdf"
        filepath = os.path.join(output_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"文件已存在，跳过: {filename}")
            success_count += 1
            continue

        # 下载论文
        start_time = time.time()
        if download_paper(paper['paper_link'], filepath):
            end_time = time.time()
            download_time = end_time - start_time
            print(f"✓ 下载成功: {filename} (耗时: {download_time:.2f}秒)")
            success_count += 1
        else:
            print(f"✗ 下载失败: {filename}")

        print("-" * 60)

    # 下载结果汇总
    print(f"\n下载完成！")
    print(f"成功下载: {success_count}/{len(papers_to_download)} 篇论文")
    print(f"文件保存位置: {output_dir}")


# 运行主程序
if __name__ == "__main__":
    main()