"""
Image processing: extraction, caption resolution, and structured summarization.

Responsibilities:
- process_image: extract and summarize images/pictures found in a converted document
- summarize_and_classify_single_image: query VLM/LLM using structured JSON output
- summarize_and_classify_images: batch parallel image summarization
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, Field

from common.lang_utils import LanguageCodes
from common.llm_utils import apply_token_buffer, get_vllm_headers
from common import misc_utils
from common.misc_utils import get_logger
from common.retry_utils import retry_on_transient_error
from common.settings import settings as common_settings
from digitize.exceptions import JobCancelledError

logger = get_logger("processing.images")


class ImageSummaryResponse(BaseModel):
    """Structured response schema for image summarization and classification."""

    summary: str = Field(
        ...,
        description="Concise 1-3 sentence summary of the key information, charts, or visual contents.",
    )
    is_informative: bool = Field(
        ...,
        description="True if the image contains substantive domain information; False if decorative or uninformative.",
    )


IMAGE_SUMMARY_JSON_SCHEMA = ImageSummaryResponse.model_json_schema()

DEFAULT_IMAGE_SYSTEM_PROMPT = """You are an AI vision analysis assistant. Analyze the provided image and caption.

Write the summary as a single continuous paragraph of plain prose — no markdown, no bullet points, no headings.
The summary must contain ALL of the following in one paragraph:
- The chart or diagram type (bar chart, line graph, pie chart, flowchart, etc.)
- What the axes, legends, and labels represent
- Every data value, data point, and measurement visible — include all numbers, percentages, and units
- Trends, comparisons, or relationships shown
- All titles, subtitles, annotations, and footnotes
For non-chart images: describe all components, labels, and relationships in full detail.
Be exhaustive — include every number and label visible.

Set is_informative=true for charts, graphs, diagrams, or images with substantive data.
Set is_informative=false for decorative borders, logos, blank pages, or stock photos with no domain relevance.

Return your response conforming strictly to the requested JSON schema."""


def _extract_json_object(text: str) -> str:
    """Extract JSON object substring and remove accidental markdown fences from model outputs."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


def _parse_image_response_fallback(text: str) -> dict:
    """Regex-based fallback parser for when the model returns slightly malformed JSON.

    Handles truncated output or minor syntax errors (missing commas, unescaped
    characters) that json.loads rejects but whose field values are still readable.
    Extracts 'summary' and 'is_informative' directly via pattern matching.

    Raises ValueError if neither field can be extracted, so the caller's outer
    exception handler retries the request as normal.
    """
    summary = ""
    is_informative = False

    # Match "summary": "..." — handles escaped quotes inside the value
    summary_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).replace('\\"', '"').replace("\\n", "\n").strip()

    # Match "is_informative": true/false (case-insensitive for robustness)
    informative_match = re.search(r'"is_informative"\s*:\s*(true|false)', text, re.IGNORECASE)
    if informative_match:
        is_informative = informative_match.group(1).lower() == "true"

    if not summary and not informative_match:
        raise ValueError(f"Could not extract any fields from malformed JSON response: {text[:200]!r}")

    return {"summary": summary, "is_informative": is_informative}


@retry_on_transient_error(max_retries=3, initial_delay=1.0, backoff_multiplier=2.0)
def summarize_and_classify_single_image(
    image_uri: str,
    caption: str,
    gen_model: str,
    llm_endpoint: str,
    max_tokens: int = 1024,
) -> tuple[str, bool]:
    """Summarize and classify an image using structured output.

    Args:
        image_uri: Data URI or URL of the image.
        caption: Associated caption text (if any).
        gen_model: Model name for the VLM/LLM.
        llm_endpoint: VLM/LLM endpoint URL.
        max_tokens: Maximum tokens for the generated response.

    Returns:
        tuple[str, bool]: (summary, decision)
    """
    image_uri = (image_uri or "").strip()
    caption = (caption or "").strip()

    # Short-circuit if there is neither visual image content nor descriptive caption
    if not image_uri and not caption:
        return "No summary available.", False

    if misc_utils.SESSION is None:
        raise RuntimeError("LLM session not initialized. Call create_llm_session() first.")

    user_text = f"Analyze this image.\nCaption: {caption if caption else 'No caption provided.'}"

    # Build multimodal content payload
    if image_uri:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": image_uri},
            },
        ]
    else:
        user_content = [{"type": "text", "text": user_text}]

    effective_max_tokens = apply_token_buffer(max_tokens, context="image_summary")

    payload: dict[str, Any] = {
        "model": gen_model,
        "messages": [
            {"role": "system", "content": DEFAULT_IMAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": effective_max_tokens,
        "stream": False,
        # Standard OpenAI Structured Outputs — honoured by vLLM and LiteLLM
        # when the backing model supports guided decoding.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ImageSummaryResponse",
                "schema": IMAGE_SUMMARY_JSON_SCHEMA,
                "strict": True,
            },
        },
    }

    try:
        api_key = common_settings.llm.api_key.get_secret_value() if common_settings.llm.api_key else None
        response = misc_utils.SESSION.post(
            f"{llm_endpoint}/v1/chat/completions",
            json=payload,
            headers=get_vllm_headers(api_key),
        )
        response.raise_for_status()
        data = response.json() or {}
        choices = data.get("choices", [])
        text = ""
        if choices:
            text = (choices[0].get("message", {}).get("content") or "").strip()

        # Parse structured JSON response, with a regex fallback for
        # models that produce slightly malformed JSON (e.g. truncated output
        # from a proxy that doesn't enforce guided decoding).
        cleaned_json = _extract_json_object(text)
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError:
            parsed = _parse_image_response_fallback(cleaned_json)
        validated = ImageSummaryResponse.model_validate(parsed)

        return validated.summary, validated.is_informative

    except Exception as e:
        logger.error(f"Error summarizing/classifying image: {e}")
        raise


def summarize_and_classify_images(
    images_data: list[tuple[str, str]],
    gen_model: str,
    llm_endpoint: str,
    doc_path: Path | str,
    max_tokens: int = 1024,
    max_workers: int | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[list[str], list[bool], dict[int, str]]:
    """Batch summarize and classify images in parallel with structured outputs.

    Args:
        images_data: List of (image_uri, caption) tuples.
        gen_model: VLM/LLM model name.
        llm_endpoint: VLM/LLM endpoint URL.
        doc_path: Document path for logging.
        max_tokens: Maximum tokens for each response.
        max_workers: Maximum thread pool workers (defaults to settings.llm.max_batch_size).
        stop_event: Cancellation event.

    Returns:
        tuple[list[str], list[bool], dict[int, str]]: (summaries, decisions, failures)
    """
    if not images_data:
        return [], [], {}

    pool_size = max_workers or getattr(common_settings.llm, "max_batch_size", 16)
    results: list[tuple[str, bool]] = [("No summary.", False)] * len(images_data)
    failures: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=min(pool_size, len(images_data))) as executor:
        futures = {
            executor.submit(
                summarize_and_classify_single_image,
                uri,
                caption,
                gen_model,
                llm_endpoint,
                max_tokens,
            ): idx
            for idx, (uri, caption) in enumerate(images_data)
        }
        pending = dict(futures)
        while pending:
            for fut in list(pending):
                if not fut.done():
                    if stop_event is not None and stop_event.is_set() and not fut.running():
                        fut.cancel()
                        del pending[fut]
                    continue
                idx = pending.pop(fut)
                try:
                    results[idx] = fut.result()
                    logger.debug(f"Summarized image {idx + 1}/{len(images_data)} from '{doc_path}'")
                except Exception as e:
                    error_msg = f"Failed to process image {idx}: {str(e)}"
                    logger.error(error_msg)
                    failures[idx] = error_msg
                    results[idx] = ("No summary available.", False)

            if stop_event is not None and stop_event.is_set():
                raise JobCancelledError(
                    f"Job cancelled during image summarization for '{doc_path}'"
                )

            if pending:
                time.sleep(0.1)

    summaries = [r[0] for r in results]
    decisions = [r[1] for r in results]
    return summaries, decisions, failures


def process_image(
    converted_doc: DoclingDocument,
    doc_path: Path | str,
    out_path: Path | str,
    gen_model: str,
    gen_endpoint: str,
    document_language: str = LanguageCodes.ENGLISH,
    stop_event: threading.Event | None = None,
) -> tuple[int, float, dict[int, str]]:
    """Extract, process, and summarize images found in a converted document using Structured Outputs.

    Saves the extracted image summaries and metadata to a JSON file.

    Args:
        converted_doc: Converted DoclingDocument instance.
        doc_path: Source document path.
        out_path: Destination JSON output path.
        gen_model: VLM/LLM model name.
        gen_endpoint: VLM/LLM endpoint URL.
        document_language: Language code for localization.
        stop_event: Cancellation event.

    Returns:
        tuple[int, float, dict[int, str]]: (images_count, process_time, failures)
    """
    images_count = 0
    failures: dict[int, str] = {}
    out_path = Path(out_path)
    t0 = time.time()

    if not getattr(converted_doc, "pictures", None):
        logger.debug(f"No images found in '{doc_path}'")
        out_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
        return images_count, time.time() - t0, failures

    doc_path = Path(doc_path)
    is_docx = doc_path.suffix.lower() == ".docx"

    # --- 1. Extract Images, URIs, and Captions directly from DoclingDocument ---
    image_uris: list[str] = []
    image_captions: list[str] = []
    image_page_numbers: list[int | None] = []

    for image_idx, picture in enumerate(converted_doc.pictures):
        # Resolve caption directly via Docling API
        caption = picture.caption_text(doc=converted_doc) if hasattr(picture, "caption_text") else ""

        # Extract image URI
        uri = ""
        if getattr(picture, "image", None) and getattr(picture.image, "uri", None):
            uri = str(picture.image.uri)

        # Page provenance
        page_no = None
        if picture.prov and picture.prov[0].page_no is not None:
            page_no = picture.prov[0].page_no
        elif is_docx:
            page_no = image_idx + 1

        image_uris.append(uri)
        image_captions.append(caption.strip())
        image_page_numbers.append(page_no)

    if stop_event and stop_event.is_set():
        raise JobCancelledError(f"Job cancelled before summarizing images for document: {doc_path}")

    selected_max_tokens = 2048

    # --- 2. Batch Summarize & Classify with VLM/LLM ---
    images_data = list(zip(image_uris, image_captions))
    summaries, decisions, failures = summarize_and_classify_images(
        images_data=images_data,
        gen_model=gen_model,
        llm_endpoint=gen_endpoint,
        doc_path=doc_path,
        max_tokens=selected_max_tokens,
        stop_event=stop_event,
    )

    # --- 3. Filter and Save Output ---
    # The image URI (base64) is not stored — it was only needed for the VLM call.
    # Persisting multi-hundred-KB base64 strings into intermediate JSON files
    # is wasteful; only the summary, caption, and page metadata are needed downstream.
    filtered_image_dicts: dict[int, dict[str, Any]] = {
        idx: {
            "caption": caption,
            "summary": summary,
            "page_number": page_num,
        }
        for idx, (keep, uri, summary, caption, page_num) in enumerate(
            zip(decisions, image_uris, summaries, image_captions, image_page_numbers)
        )
        if keep
    }

    images_count = len(filtered_image_dicts)
    out_path.write_text(json.dumps(filtered_image_dicts, indent=2), encoding="utf-8")
    process_time = time.time() - t0

    return images_count, process_time, failures
