import argparse
import os
from pathlib import Path
import importlib


def _apply_safetensors_paddle_fallback():
    """
    Hotfix: paddlex loads VLM weights via safetensors with framework="paddle".
    Some safetensors builds don't recognize 'paddle' yet, causing
    SafetensorError: framework paddle is invalid.

    This patch forces paddlex's internal model_utils to use a wrapper that
    falls back to framework="numpy" when 'paddle' is rejected.
    """
    try:
        # Patch global safetensors.safe_open first
        import safetensors as _st
        _safe_open_original = _st.safe_open

        def _safe_open_wrapper(filename, framework="paddle", *args, **kwargs):
            try:
                return _safe_open_original(
                    filename, framework=framework, *args, **kwargs
                )
            except Exception as e:
                msg = str(e).lower()
                if "framework paddle is invalid" in msg:
                    # Fallback to numpy tensors
                    return _safe_open_original(
                        filename, framework="numpy", *args, **kwargs
                    )
                raise

        _st.safe_open = _safe_open_wrapper  # global monkey patch

        # Do NOT import paddlex here to avoid early PDX initialization.
        # Rely on global patch; if paddlex imports safetensors.safe_open later,
        # it will receive the wrapper.
        print("Applied global safetensors safe_open fallback (paddle→numpy).")
    except Exception as e:
        print(f"Safetensors fallback patch not applied: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL with ModelScope as model source and save results"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png",
        help="Input file path or URL (image/PDF)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save JSON/Markdown outputs",
    )
    parser.add_argument(
        "--use_doc_orientation_classify",
        action="store_true",
        help="Enable document orientation classification",
    )
    parser.add_argument(
        "--use_doc_unwarping",
        action="store_true",
        help="Enable document image unwarping",
    )
    parser.add_argument(
        "--use_layout_detection",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=True,
        help="Enable layout detection (default: True)",
    )
    parser.add_argument(
        "--format_block_content",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=False,
        help="Format content blocks into Markdown (default: False)",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        default="MODELSCOPE",
        help="Model download source, e.g., MODELSCOPE, HUGGINGFACE, BOS",
    )
    args = parser.parse_args()

    # Prefer ModelScope for model downloads unless the user overrides
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", args.model_source)

    # Lazy import to make CLI help fast even if deps are missing
    _apply_safetensors_paddle_fallback()
    from paddleocr import PaddleOCRVL

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_layout_detection=args.use_layout_detection,
        format_block_content=args.format_block_content,
    )

    print(f"Using model source: {os.environ.get('PADDLE_PDX_MODEL_SOURCE')}")
    print(f"Running PaddleOCR-VL on input: {args.input}")

    results = pipeline.predict(args.input)
    for idx, res in enumerate(results):
        print(f"\n--- Page {idx} ---")
        try:
            res.print()
        except Exception:
            pass
        # Save structured outputs
        res.save_to_json(save_path=str(output_dir))
        res.save_to_markdown(save_path=str(output_dir))

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()