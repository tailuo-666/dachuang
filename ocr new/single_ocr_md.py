import argparse
import os
import re
import shutil
from pathlib import Path

def collect_files(p):
    p = Path(p)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if p.is_dir():
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
    if p.is_file() and p.suffix.lower() in exts:
        return [p]
    return []

_ocr_cache = {}

def get_ocr(lang):
    key = lang or "ch"
    if key in _ocr_cache:
        return _ocr_cache[key]
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=key)
    except Exception:
        ocr = None
    _ocr_cache[key] = ocr
    return ocr

def run_ocr(ocr, img_path):
    if ocr is None:
        return []
    res = ocr.ocr(str(img_path), cls=True)
    lines = []
    for page in res:
        for item in page:
            txt = item[1][0]
            lines.append(txt)
    return lines

def highlight_chinese(text, tag):
    def repl(m):
        s = m.group(0)
        return f"{tag}{s}{tag}"
    return re.sub(r"[\u4e00-\u9fff]+", repl, text)

def build_markdown(src_name, lines, tag):
    header = f"# {src_name} OCR\n\n"
    body = []
    for t in lines:
        body.append(highlight_chinese(t, tag))
    return header + "\n".join(body) + "\n"

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def process(input_path, output_dir, lang, tag, copy_input):
    files = collect_files(input_path)
    ensure_dir(output_dir)
    if copy_input:
        for f in files:
            shutil.copy2(f, Path(output_dir) / f.name)
    ocr = get_ocr(lang)
    results = []
    for f in files:
        lines = run_ocr(ocr, f)
        md = build_markdown(f.name, lines, tag)
        md_path = Path(output_dir) / (f.stem + "_ocr.md")
        with open(md_path, "w", encoding="utf-8") as w:
            w.write(md)
        results.append(str(md_path))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("--output", default=None)
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--tag", default="【上传】")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    inp = Path(args.input)
    if args.output:
        out_dir = Path(args.output)
    else:
        base = inp if inp.is_dir() else inp.parent
        out_dir = base / "ocr_md_output"
    res = process(inp, out_dir, args.lang, args.tag, not args.no_copy)
    print("生成Markdown:")
    for r in res:
        print(r)

if __name__ == "__main__":
    main()

