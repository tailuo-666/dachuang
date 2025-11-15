from typing import List, Tuple, Dict, Optional, Union, Any
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def segment_text(
    text: str,
    strategy: str = "sentence",
    max_len: int = 300,
    overlap: int = 50,
) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    if strategy == "sentence":
        pattern = r"(?<=[。！？!?；;\.])\s+"
        segs = re.split(pattern, text)
        segs = [s for s in segs if s]
        return segs

    if strategy == "by_line":
        segs = [s.strip() for s in re.split(r"\r?\n+", text)]
        return [s for s in segs if s]

    if strategy == "fixed":
        if max_len <= 0:
            return [text]
        step = max(1, max_len - max(0, overlap))
        segs: List[str] = []
        for i in range(0, len(text), step):
            segs.append(text[i : i + max_len])
        return segs

    return [text]


def segment_text_with_spans(
    text: str,
    strategy: str = "sentence",
    max_len: int = 300,
    overlap: int = 50,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    text = _normalize_text(text)
    if not text:
        return [], []

    if strategy == "sentence":
        spans: List[Tuple[int, int]] = []
        segments: List[str] = []
        start = 0
        for m in re.finditer(r"[。！？!?；;\.]\s+", text):
            end = m.end()
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)
                spans.append((start, end))
            start = end
        if start < len(text):
            seg = text[start:].strip()
            if seg:
                segments.append(seg)
                spans.append((start, len(text)))
        return segments, spans

    if strategy == "by_line":
        segments: List[str] = []
        spans: List[Tuple[int, int]] = []
        pos = 0
        for line in re.split(r"\r?\n+", text):
            seg = line.strip()
            if not seg:
                pos += len(line) + 1
                continue
            start = text.find(seg, pos)
            end = start + len(seg)
            segments.append(seg)
            spans.append((start, end))
            pos = end
        return segments, spans

    if strategy == "fixed":
        if max_len <= 0:
            return [text], [(0, len(text))]
        step = max(1, max_len - max(0, overlap))
        segments: List[str] = []
        spans: List[Tuple[int, int]] = []
        for i in range(0, len(text), step):
            j = min(i + max_len, len(text))
            segments.append(text[i:j])
            spans.append((i, j))
        return segments, spans

    return [text], [(0, len(text))]


def vectorize_texts(
    texts: List[str],
    method: str = "tfidf",
    *,
    tokenizer: Optional[Any] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    custom_embedder: Optional[Any] = None,
) -> Tuple[Union[np.ndarray, Any], Any]:
    if not texts:
        return np.zeros((0, 0)), None

    if method == "tfidf":
        vec = TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            max_features=max_features,
        )
        m = vec.fit_transform(texts)
        return m, vec

    if method == "custom":
        if custom_embedder is None:
            raise ValueError("custom_embedder must be provided for method 'custom'")
        embeddings = custom_embedder(texts)
        if hasattr(embeddings, "toarray"):
            arr = embeddings.toarray()
        else:
            arr = np.asarray(embeddings)
        return arr, custom_embedder

    raise ValueError(f"Unsupported vectorization method: {method}")


def split_and_vectorize(
    input_data: Union[str, List[str]],
    *,
    segmentation: str = "sentence",
    max_len: int = 300,
    overlap: int = 50,
    vectorization: str = "tfidf",
    tokenizer: Optional[Any] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    custom_embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    if isinstance(input_data, str):
        segments, spans = segment_text_with_spans(input_data, segmentation, max_len, overlap)
    else:
        segments = []
        spans = []
        for t in input_data:
            sgs, sps = segment_text_with_spans(t, segmentation, max_len, overlap)
            segments.extend(sgs)
            spans.extend(sps)

    vectors, model = vectorize_texts(
        segments,
        method=vectorization,
        tokenizer=tokenizer,
        ngram_range=ngram_range,
        max_features=max_features,
        custom_embedder=custom_embedder,
    )

    return {
        "segments": segments,
        "spans": spans,
        "vectors": vectors,
        "vectorizer": model,
    }


__all__ = [
    "segment_text",
    "segment_text_with_spans",
    "vectorize_texts",
    "split_and_vectorize",
]