"""V2 browser-cookie transport for Grok Web requests."""

import base64
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import orjson

from app.dataplane.proxy.adapters.headers import build_http_headers
from app.dataplane.proxy.adapters.profile import browser_from_user_agent
from app.dataplane.proxy.adapters.session import ResettableSession
from app.dataplane.reverse.protocol.xai_assets import infer_content_type
from app.dataplane.reverse.protocol.xai_chat import (
    StreamAdapter,
    build_chat_payload,
    classify_line,
    raise_for_stream_error,
)
from app.dataplane.reverse.runtime.endpoint_table import ASSETS_UPLOAD, CHAT
from app.dataplane.reverse.transport.asset_upload import parse_data_uri
from app.platform.config.snapshot import get_config
from app.platform.errors import UpstreamError, ValidationError

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _ROOT / "config.json"
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)
_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(slots=True, frozen=True)
class V2BrowserConfig:
    """V2 浏览器态配置。"""

    cookie: str
    user_agent: str
    browser: str
    output_dir: Path


@dataclass(slots=True, frozen=True)
class V2UploadResult:
    """Grok 上传文件结果。"""

    file_id: str
    file_uri: str


@dataclass(slots=True, frozen=True)
class V2ChatResult:
    """Grok 对话结果。"""

    content: str
    reasoning_content: str | None


@dataclass(slots=True, frozen=True)
class V2ImageChatResult:
    """Grok 图片生成流的首张最终图片结果。"""

    image_url: str
    reasoning_content: str | None


@dataclass(slots=True, frozen=True)
class V2DownloadResult:
    """Grok 图片下载结果。"""

    path: Path
    content_type: str
    size: int


@dataclass(slots=True, frozen=True)
class V2ImageBytesResult:
    """Grok 图片字节下载结果。"""

    content: bytes
    content_type: str


def load_config() -> V2BrowserConfig:
    """从环境变量或项目根目录 config.json 读取浏览器 Cookie 与指纹配置。"""
    data = _load_file_config()

    cookie = str(os.getenv("GROK_V2_COOKIE") or data.get("cookie") or "").strip()
    if not cookie:
        raise ValidationError("GROK_V2_COOKIE or config.json cookie cannot be empty", param="cookie")

    user_agent = str(
        os.getenv("GROK_V2_USER_AGENT") or data.get("user_agent") or _DEFAULT_UA
    ).strip() or _DEFAULT_UA
    browser = (
        str(os.getenv("GROK_V2_BROWSER") or data.get("browser") or "").strip()
        or browser_from_user_agent(user_agent)
        or "chrome136"
    )
    output_raw = str(
        os.getenv("GROK_V2_OUTPUT_DIR") or data.get("output_dir") or "images"
    ).strip() or "images"
    output_dir = Path(output_raw)
    if not output_dir.is_absolute():
        output_dir = _ROOT / output_dir
    return V2BrowserConfig(
        cookie=_merge_cookie(cookie),
        user_agent=user_agent,
        browser=browser,
        output_dir=output_dir,
    )


def _load_file_config() -> dict:
    """读取本地 config.json；云端使用环境变量时允许文件不存在。"""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        data = orjson.loads(_CONFIG_PATH.read_bytes())
    except Exception as exc:
        raise ValidationError("Invalid config.json", param="config") from exc
    if not isinstance(data, dict):
        raise ValidationError("config.json must be an object", param="config")
    return data


def _merge_cookie(cookie: str) -> str:
    """规范化 Cookie，兼容只填写 sso token 的情况。"""
    normalized = cookie.strip()
    if "=" not in normalized:
        token = normalized[4:] if normalized.startswith("sso=") else normalized
        normalized = f"sso={token}; sso-rw={token}"
    return normalized


def _cookie_value(cookie: str, name: str) -> str:
    """从 Cookie 字符串读取指定字段。"""
    prefix = f"{name}="
    for part in cookie.split(";"):
        item = part.strip()
        if item.startswith(prefix):
            return item[len(prefix):]
    return ""


def _browser_headers(
    cfg: V2BrowserConfig,
    *,
    content_type: str | None = "application/json",
    origin: str = "https://grok.com",
    referer: str = "https://grok.com/",
) -> dict[str, str]:
    """复用 V1 头部构造，并覆盖为 config.json 里的原始 Cookie。"""
    sso = _cookie_value(cfg.cookie, "sso") or _cookie_value(cfg.cookie, "sso-rw") or cfg.cookie
    headers = build_http_headers(
        sso,
        content_type=content_type,
        origin=origin,
        referer=referer,
    )
    headers["Cookie"] = cfg.cookie
    headers["User-Agent"] = cfg.user_agent
    headers["x-xai-request-id"] = str(uuid.uuid4())
    return headers


async def post_json(url: str, payload: dict, *, referer: str = "https://grok.com/") -> dict:
    """使用 V2 浏览器配置发送 JSON 请求。"""
    cfg = load_config()
    timeout_s = get_config().get_float("chat.timeout", 120.0)
    async with ResettableSession(browser_override=cfg.browser) as session:
        response = await session.post(
            url,
            headers=_browser_headers(cfg, referer=referer),
            data=orjson.dumps(payload),
            timeout=timeout_s,
        )
    if response.status_code != 200:
        body = response.content.decode("utf-8", "replace")[:400]
        raise UpstreamError(
            f"V2 upstream returned {response.status_code}",
            status=response.status_code,
            body=body,
        )
    return orjson.loads(response.content) if response.content.strip() else {}


async def upload_image_data_uri(data_uri: str) -> V2UploadResult:
    """上传 Data URI 图片到 Grok，并返回附件 ID。"""
    filename, b64, mime = parse_data_uri(data_uri)
    payload = {"fileName": filename, "fileMimeType": mime, "content": b64}
    result = await post_json(ASSETS_UPLOAD, payload, referer="https://grok.com/")
    file_id = str(result.get("fileMetadataId") or result.get("fileId") or "").strip()
    file_uri = str(result.get("fileUri") or "").strip()
    if not file_id:
        raise UpstreamError("V2 image upload returned no file id")
    return V2UploadResult(file_id=file_id, file_uri=file_uri)


async def upload_image_bytes(filename: str, mime: str, raw: bytes) -> V2UploadResult:
    """上传原始图片字节到 Grok。"""
    if not raw:
        raise ValidationError("Uploaded image cannot be empty", param="image")
    b64 = base64.b64encode(raw).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    return await upload_image_data_uri(data_uri)


async def stream_chat(
    *,
    model: str,
    mode_id,
    message: str,
    file_attachments: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """发送 Grok Chat 请求并返回原始 SSE 行。"""
    cfg = load_config()
    payload = build_chat_payload(
        message=message,
        mode_id=mode_id,
        file_attachments=file_attachments or [],
    )
    timeout_s = get_config().get_float("chat.timeout", 120.0)
    async with ResettableSession(browser_override=cfg.browser) as session:
        response = await session.post(
            CHAT,
            headers=_browser_headers(cfg),
            data=orjson.dumps(payload),
            timeout=timeout_s,
            stream=True,
        )
        if response.status_code != 200:
            body = response.content.decode("utf-8", "replace")[:400]
            raise UpstreamError(
                f"V2 chat upstream returned {response.status_code}",
                status=response.status_code,
                body=body,
            )
        async for line in response.aiter_lines():
            yield line


async def send_chat(
    *,
    model: str,
    mode_id,
    message: str,
    file_attachments: list[str] | None = None,
) -> V2ChatResult:
    """发送 Grok Chat 请求并聚合文本、推理和图片 URL。"""
    adapter = StreamAdapter()
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    async for line in stream_chat(
        model=model,
        mode_id=mode_id,
        message=message,
        file_attachments=file_attachments,
    ):
        event_type, data = classify_line(line)
        if event_type == "done":
            break
        if event_type != "data" or not data:
            continue
        raise_for_stream_error(data)
        for event in adapter.feed(data):
            if event.kind == "text":
                text_parts.append(event.content)
            elif event.kind == "thinking":
                thinking_parts.append(event.content)
            elif event.kind == "image" and event.content:
                text_parts.append(event.content)
    return V2ChatResult(
        content="".join(text_parts).strip(),
        reasoning_content="".join(thinking_parts).strip() or None,
    )


async def send_chat_until_image(
    *,
    model: str,
    mode_id,
    message: str,
    file_attachments: list[str] | None = None,
) -> V2ImageChatResult:
    """发送图片类 Chat 请求，拿到首张最终图片 URL 后立即返回。"""
    adapter = StreamAdapter()
    thinking_parts: list[str] = []
    async for line in stream_chat(
        model=model,
        mode_id=mode_id,
        message=message,
        file_attachments=file_attachments,
    ):
        event_type, data = classify_line(line)
        if event_type == "done":
            break
        if event_type != "data" or not data:
            continue
        raise_for_stream_error(data)
        for event in adapter.feed(data):
            if event.kind == "thinking":
                thinking_parts.append(event.content)
            elif event.kind == "image" and event.content:
                return V2ImageChatResult(
                    image_url=event.content,
                    reasoning_content="".join(thinking_parts).strip() or None,
                )
    raise UpstreamError("V2 image chat returned no image URL")


def extract_first_image_url(text: str) -> str:
    """从响应文本里提取第一张图片 URL。"""
    match = re.search(r"https://assets\.grok\.com/[^\s)]+", text or "")
    return match.group(0) if match else ""


def _output_path(url: str, content_type: str, filename: str | None = None) -> Path:
    """根据 URL 与 MIME 类型生成本地输出路径。"""
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        name = Path(filename).name
    else:
        ext = _IMAGE_EXT_BY_MIME.get(content_type.split(";", 1)[0].lower(), "")
        if not ext:
            inferred = infer_content_type(url) or ""
            ext = _IMAGE_EXT_BY_MIME.get(inferred, ".jpg")
        name = f"grok_v2_download_{uuid.uuid4().hex[:8]}{ext}"
    return cfg.output_dir / name


async def download_image(url: str, *, filename: str | None = None) -> V2DownloadResult:
    """使用 V2 浏览器配置下载 Grok 图片到本地。"""
    result = await fetch_image_bytes(url)
    out = _output_path(url, result.content_type, filename)
    out.write_bytes(result.content)
    return V2DownloadResult(
        path=out,
        content_type=result.content_type,
        size=out.stat().st_size,
    )


async def fetch_image_bytes(url: str) -> V2ImageBytesResult:
    """使用 V2 浏览器配置下载 Grok 图片字节，不写入本地文件。"""
    cfg = load_config()
    headers = _browser_headers(
        cfg,
        content_type=None,
        origin="https://assets.grok.com",
        referer="https://grok.com/",
    )
    headers.update({
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-site",
    })
    headers.pop("Content-Type", None)
    async with ResettableSession(browser_override=cfg.browser) as session:
        response = await session.get(url, headers=headers, timeout=120.0, allow_redirects=True)
    if response.status_code != 200:
        body = response.content.decode("utf-8", "replace")[:400]
        raise UpstreamError(
            f"V2 image download returned {response.status_code}",
            status=response.status_code,
            body=body,
        )
    content_type = response.headers.get("content-type", "image/jpeg")
    return V2ImageBytesResult(
        content=response.content,
        content_type=content_type,
    )


__all__ = [
    "V2ImageBytesResult",
    "V2UploadResult",
    "V2ChatResult",
    "V2ImageChatResult",
    "V2DownloadResult",
    "download_image",
    "extract_first_image_url",
    "fetch_image_bytes",
    "load_config",
    "send_chat",
    "send_chat_until_image",
    "stream_chat",
    "upload_image_bytes",
    "upload_image_data_uri",
]
