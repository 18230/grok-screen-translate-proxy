"""V2 routes backed by project-root browser Cookie config."""

import mimetypes

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.control.model.registry import resolve as resolve_model
from app.platform.auth.middleware import verify_api_key
from app.platform.errors import UpstreamError, ValidationError
from app.products.openai._format import make_chat_response
from app.products.openai.chat import _extract_message
from app.products.openai.schemas import ChatCompletionRequest
from app.products.openai.v2_browser import (
    config_fingerprint,
    download_image,
    extract_first_image_url,
    fetch_image_bytes,
    send_chat,
    send_chat_until_image,
    upload_image_bytes,
    upload_image_data_uri,
)

router = APIRouter(prefix="/v2")

MAX_SCREEN_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_SCREEN_TRANSLATE_MODEL = "grok-4.20-fast"
DEFAULT_SCREEN_QUESTION = "翻译下这个图片"
MAX_SCREEN_QUESTION_CHARS = 1000


SUPPORTED_TARGET_LANGUAGES = {
    "en": "English",
    "zh-cn": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Portuguese",
    "th": "Thai",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
}


class V2UploadRequest(BaseModel):
    """JSON 图片上传请求。"""

    image: str


class V2DownloadRequest(BaseModel):
    """图片下载请求。"""

    url: str
    filename: str | None = None


async def read_image_upload(
    image: UploadFile,
    *,
    param: str = "image",
    max_bytes: int | None = None,
) -> tuple[str, str, bytes]:
    """读取并校验上传图片，按调用场景选择是否限制文件大小。"""
    if max_bytes is not None:
        raw = await image.read(max_bytes + 1)
    else:
        raw = await image.read()
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValidationError("Uploaded image is too large", param=param)
    if not raw:
        raise ValidationError("Uploaded image cannot be empty", param=param)

    filename = image.filename or "image.jpg"
    mime = (
        (image.content_type or "").strip()
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
    if not mime.startswith("image/"):
        raise ValidationError("Uploaded file must be an image", param=param)
    return filename, mime, raw


def normalize_target_language(tag: str | None, name: str | None) -> str | None:
    """校验并解析目标语言，避免任意输入污染图片翻译指令。"""
    if not tag:
        return None
    normalized = tag.strip().lower().replace("_", "-")
    language = SUPPORTED_TARGET_LANGUAGES.get(normalized)
    if language is None:
        raise ValidationError("Unsupported target_language", param="target_language")
    if name and name.strip() and name.strip().lower() != language.lower():
        return f"{language} ({name.strip()})"
    return language


def build_image_translation_prompt(language: str) -> str:
    """生成稳定的图片翻译回填指令，统一由服务端控制目标语言和返回格式。"""
    return (
        f"Translate every visible non-{language} text in the image into {language}. "
        "Preserve the original screenshot pixel style as much as possible. Keep the "
        "same layout, icons, spacing, text position, and approximate font size. Do "
        "not recolor, tint, darken, blur, add shadows, add overlays, or regenerate "
        "unchanged UI areas. Every original background must remain visually identical "
        "to the source image, including any color, texture, gradient, shadow, image, "
        "or pattern. Only replace the original text in-place with the translation. "
        "Return only the translated image, with no explanation, caption, markdown, "
        "or extra text."
    )


def normalize_screen_question(question: str | None) -> str:
    """清理屏幕问答输入，空问题回退到默认翻译图片，过长问题直接拒绝。"""
    value = (question or "").strip() or DEFAULT_SCREEN_QUESTION
    if len(value) > MAX_SCREEN_QUESTION_CHARS:
        raise ValidationError("question is too long", param="question")
    return value


def build_screen_question_prompt(question: str, language: str | None = None) -> str:
    """生成屏幕问答提示词，要求 Grok 只依据当前截图回答。"""
    language_instruction = ""
    if language:
        language_instruction = (
            f" Always answer in {language}. If the user asks to translate the image, "
            f"translate visible text into {language}."
        )
    return (
        "You are a mobile screen assistant. Answer the user's question based only "
        "on the screenshot. Be concise and practical. If the question asks what "
        "to tap, identify the visible label or area. If the answer is not visible "
        f"in the screenshot, say that it is not visible.{language_instruction} "
        f"User question: {question}"
    )


@router.post("/images/upload", dependencies=[Depends(verify_api_key)])
async def v2_upload_image_json(req: V2UploadRequest):
    """上传 Data URI 图片，返回 Grok 文件 ID。"""
    result = await upload_image_data_uri(req.image)
    return JSONResponse({"file_id": result.file_id, "file_uri": result.file_uri})


@router.get("/debug/config", dependencies=[Depends(verify_api_key)])
async def v2_debug_config():
    """返回 V2 浏览器态配置指纹，用于排查云端 Secret 注入。"""
    return JSONResponse(config_fingerprint())


@router.post("/images/upload-file", dependencies=[Depends(verify_api_key)])
async def v2_upload_image_file(image: UploadFile = File(...)):
    """上传 multipart 图片文件，返回 Grok 文件 ID。"""
    filename, mime, raw = await read_image_upload(image)
    result = await upload_image_bytes(filename, mime, raw)
    return JSONResponse({"file_id": result.file_id, "file_uri": result.file_uri})


@router.post("/screen/translate", dependencies=[Depends(verify_api_key)])
async def v2_screen_translate_image(
    image: UploadFile = File(...),
    target_language: str = Form(...),
    target_language_name: str | None = Form(None),
    model: str = Form(DEFAULT_SCREEN_TRANSLATE_MODEL),
):
    """整屏图片翻译闭环：上传截图、等待 Grok 最终图片、代理下载并返回图片字节。"""
    spec = resolve_model(model)
    if not spec.is_chat():
        raise ValidationError(f"Model {model!r} is not a chat model", param="model")

    language = normalize_target_language(target_language, target_language_name)
    if not language:
        raise ValidationError("target_language is required", param="target_language")

    filename, mime, raw = await read_image_upload(image, max_bytes=MAX_SCREEN_IMAGE_BYTES)
    uploaded = await upload_image_bytes(filename, mime, raw)
    image_chat = await send_chat_until_image(
        model=model,
        mode_id=spec.mode_id,
        message=build_image_translation_prompt(language),
        file_attachments=[uploaded.file_id],
    )
    result = await fetch_image_bytes(image_chat.image_url)
    return Response(
        content=result.content,
        media_type=result.content_type.split(";", 1)[0] or "image/jpeg",
        headers={
            "Content-Length": str(len(result.content)),
        },
    )


@router.post("/screen/ask", dependencies=[Depends(verify_api_key)])
async def v2_screen_ask(
    image: UploadFile = File(...),
    question: str = Form(DEFAULT_SCREEN_QUESTION),
    target_language: str | None = Form(None),
    target_language_name: str | None = Form(None),
    model: str = Form(DEFAULT_SCREEN_TRANSLATE_MODEL),
):
    """屏幕问答闭环：Android 只上传一次截图，后端内部完成 Grok 上传和问答。"""
    spec = resolve_model(model)
    if not spec.is_chat():
        raise ValidationError(f"Model {model!r} is not a chat model", param="model")

    normalized_question = normalize_screen_question(question)
    language = normalize_target_language(target_language, target_language_name)
    filename, mime, raw = await read_image_upload(image, max_bytes=MAX_SCREEN_IMAGE_BYTES)
    uploaded = await upload_image_bytes(filename, mime, raw)
    result = await send_chat(
        model=model,
        mode_id=spec.mode_id,
        message=build_screen_question_prompt(normalized_question, language),
        file_attachments=[uploaded.file_id],
    )
    answer = result.content.strip()
    if not answer:
        raise UpstreamError("Grok returned empty answer")
    return JSONResponse(
        {
            "answer": answer,
            "reasoning_content": result.reasoning_content,
        }
    )


@router.post("/images/download", dependencies=[Depends(verify_api_key)])
async def v2_download_image(req: V2DownloadRequest):
    """使用 config.json 中的浏览器 Cookie 下载 Grok 图片。"""
    result = await download_image(req.url, filename=req.filename)
    return JSONResponse(
        {
            "path": str(result.path),
            "content_type": result.content_type,
            "size": result.size,
        }
    )


@router.post("/images/download-content", dependencies=[Depends(verify_api_key)])
async def v2_download_image_content(req: V2DownloadRequest):
    """使用浏览器 Cookie 下载 Grok 图片，并直接返回图片字节。"""
    result = await fetch_image_bytes(req.url)
    return Response(
        content=result.content,
        media_type=result.content_type.split(";", 1)[0] or "image/jpeg",
        headers={
            "Content-Length": str(len(result.content)),
        },
    )


@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def v2_chat_completions(req: ChatCompletionRequest):
    """使用 config.json 浏览器态发送 Chat Completions 请求。"""
    spec = resolve_model(req.model)
    if not spec.is_chat():
        raise ValidationError(f"Model {req.model!r} is not a chat model", param="model")
    if not req.messages:
        raise ValidationError("messages cannot be empty", param="messages")

    messages = [item.model_dump(exclude_none=True) for item in req.messages]
    message, files = _extract_message(messages)
    if not message.strip():
        raise ValidationError("messages cannot be empty after extraction", param="messages")
    target_language = normalize_target_language(req.target_language, req.target_language_name)
    if target_language and files:
        message = build_image_translation_prompt(target_language)

    attachments = []
    for file_input in files:
        if file_input.startswith("grok-file:"):
            attachments.append(file_input.split(":", 1)[1].strip())
            continue
        uploaded = await upload_image_data_uri(file_input)
        attachments.append(uploaded.file_id)

    result = await send_chat(
        model=req.model,
        mode_id=spec.mode_id,
        message=message,
        file_attachments=attachments,
    )
    response = make_chat_response(
        req.model,
        result.content,
        prompt_content=message,
        reasoning_content=result.reasoning_content,
    )
    image_url = extract_first_image_url(result.content)
    if image_url:
        response["image_url"] = image_url
    return JSONResponse(response)


__all__ = ["router"]
