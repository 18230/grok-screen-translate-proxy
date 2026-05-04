"""V2 routes backed by project-root browser Cookie config."""

import mimetypes

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.control.model.registry import resolve as resolve_model
from app.platform.auth.middleware import verify_api_key
from app.platform.errors import ValidationError
from app.products.openai._format import make_chat_response
from app.products.openai.chat import _extract_message
from app.products.openai.schemas import ChatCompletionRequest
from app.products.openai.v2_browser import (
    download_image,
    extract_first_image_url,
    fetch_image_bytes,
    send_chat,
    upload_image_bytes,
    upload_image_data_uri,
)

router = APIRouter(prefix="/v2")


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
        "Keep the original screenshot layout, visual style, background, icons, colors, "
        "spacing, text position, and approximate font size. Replace the original text "
        "in-place with the translation. Return only the translated image, with no "
        "explanation, caption, markdown, or extra text."
    )


@router.post("/images/upload", dependencies=[Depends(verify_api_key)])
async def v2_upload_image_json(req: V2UploadRequest):
    """上传 Data URI 图片，返回 Grok 文件 ID。"""
    result = await upload_image_data_uri(req.image)
    return JSONResponse({"file_id": result.file_id, "file_uri": result.file_uri})


@router.post("/images/upload-file", dependencies=[Depends(verify_api_key)])
async def v2_upload_image_file(image: UploadFile = File(...)):
    """上传 multipart 图片文件，返回 Grok 文件 ID。"""
    raw = await image.read()
    mime = (
        (image.content_type or "").strip()
        or mimetypes.guess_type(image.filename or "")[0]
        or "application/octet-stream"
    )
    if not mime.startswith("image/"):
        raise ValidationError("Uploaded file must be an image", param="image")
    result = await upload_image_bytes(image.filename or "image", mime, raw)
    return JSONResponse({"file_id": result.file_id, "file_uri": result.file_uri})


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
