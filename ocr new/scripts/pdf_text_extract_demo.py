import argparse
import os
import io
import sys
import json
import time
import tempfile
import datetime as dt
from typing import List, Optional

import requests


def _download_url_to_temp(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path


def _bytes_to_temp_pdf(b: bytes) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return tmp_path


def _extract_text_by_pymupdf(pdf_path: str) -> Optional[List[str]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for i in range(doc.page_count):
                page = doc[i]
                text = page.get_text("text") or ""
                # Normalize whitespace
                text = "\n".join([line.rstrip() for line in text.splitlines()])
                pages.append(text)
    except Exception:
        return None
    # If content is trivially empty on all pages, treat as None
    if all(len(p.strip()) == 0 for p in pages):
        return None
    return pages


def _render_pages_to_images(pdf_path: str):
    # Returns list of numpy BGR images suitable for PaddleOCR
    imgs = []
    try:
        import fitz
        from PIL import Image
        import numpy as np
        import cv2
        with fitz.open(pdf_path) as pdf:
            for i in range(pdf.page_count):
                page = pdf[i]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
        return imgs
    except Exception:
        pass

    # Fallback to pypdfium2 if PyMuPDF is unavailable
    try:
        import pypdfium2 as pdfium
        from PIL import Image
        import numpy as np
        import cv2
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        for i in range(n_pages):
            page = pdf.get_page(i)
            pil_image = page.render(scale=2).to_pil()
            img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            imgs.append(img)
        return imgs
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF pages to images: {e}")


def _ocr_pages_to_text(imgs: List, lang: str = "ch") -> List[str]:
    from paddleocr import PaddleOCR

    # use_angle_cls 为旧参数名，会自动映射到 use_textline_orientation
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    pages_text = []
    for img in imgs:
        # 使用新接口 predict，不再传 cls 参数
        results = ocr.predict(img, return_word_box=False)
        page_lines = []
        for res in results:
            try:
                # 根据 PaddleX OCRResult 的 JSON 结构抽取识别文本
                rec_texts = res.json["res"].get("rec_texts", [])
                for t in rec_texts:
                    if isinstance(t, str) and t.strip():
                        page_lines.append(t.strip())
            except Exception:
                # 容错：如果结构不匹配则跳过
                continue
        pages_text.append("\n".join(page_lines))
    return pages_text


def run(input_arg: Optional[str], bytes_path: Optional[str], output_dir: str, lang: str):
    os.makedirs(output_dir, exist_ok=True)

    # Resolve input into a temp PDF file
    tmp_pdf = None
    try:
        if bytes_path:
            with open(bytes_path, "rb") as f:
                b = f.read()
            tmp_pdf = _bytes_to_temp_pdf(b)
        elif input_arg:
            if input_arg.lower().startswith("http://") or input_arg.lower().startswith("https://"):
                tmp_pdf = _download_url_to_temp(input_arg)
            else:
                # local path
                if not os.path.exists(input_arg):
                    raise FileNotFoundError(f"Input file not found: {input_arg}")
                tmp_pdf = input_arg
        else:
            raise ValueError("No input provided. Use --input or --bytes_path.")

        # 1) Try direct text extraction via PyMuPDF
        pages_text = _extract_text_by_pymupdf(tmp_pdf)

        # 2) Fallback: render pages to images, OCR with PaddleOCR
        if pages_text is None:
            imgs = _render_pages_to_images(tmp_pdf)
            pages_text = _ocr_pages_to_text(imgs, lang=lang)

        # Build output
        extraction_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "pages": pages_text,
            "status": "success",
            "metadata": {
                "page_count": len(pages_text),
                "extraction_time": extraction_time,
            },
        }

        # Save JSON
        out_json = os.path.join(output_dir, "result.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Also save Markdown for convenience
        out_md = os.path.join(output_dir, "result.md")
        with open(out_md, "w", encoding="utf-8") as f:
            for idx, page in enumerate(pages_text, start=1):
                f.write(f"# 第{idx}页\n\n")
                f.write(page.strip() + "\n\n")

        # Save plain text in page order
        out_txt = os.path.join(output_dir, "result.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for idx, page in enumerate(pages_text, start=1):
                f.write(page.strip())
                if idx < len(pages_text):
                    f.write("\n\n")

        print(f"Saved: {out_json}\nSaved: {out_md}\nSaved: {out_txt}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # Clean temp when we created it
        try:
            if tmp_pdf and os.path.basename(tmp_pdf).startswith("tmp") and os.path.exists(tmp_pdf):
                os.remove(tmp_pdf)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Extract plain text from PDF (local path, URL, or bytes)."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input PDF path or URL.",
    )
    parser.add_argument(
        "--bytes_path",
        type=str,
        help="Path to a binary file containing raw PDF bytes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_pdf_text",
        help="Output directory to save result.json and result.md.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ch",
        help="Language for OCR fallback (e.g., 'ch','en').",
    )

    args = parser.parse_args()
    run(args.input, args.bytes_path, args.output, args.lang)


if __name__ == "__main__":
    main()
