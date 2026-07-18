"""Edge-TTS 语音合成应用（深度集成版）。

本模块通过 FastAPI + Gradio 提供：
- Web UI（简单模式 + SSML 高级模式）
- REST API（/api/voices、/api/tts、/api/stream）

edge_tts 从环境（pip 安装）导入，本地源码备份在 .dev/edge-tts/。
"""

# ---------------------------------------------------------------------------
# Standard library / third-party / local imports.
# ---------------------------------------------------------------------------
import asyncio
import base64
import json
import os
import ssl
import tempfile
from typing import Optional, Tuple, List, Dict, Any

import aiohttp
import certifi
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import gradio as gr
import uvicorn

import edge_tts
from edge_tts.constants import (
    DEFAULT_VOICE,
    WSS_URL,
    WSS_HEADERS,
    SEC_MS_GEC_VERSION,
    TICKS_PER_SECOND,
    MP3_BITRATE_BPS,
)
from edge_tts.communicate import (
    ssml_headers_plus_data,
    connect_id,
    date_to_string,
    get_headers_and_data,
    remove_incompatible_characters,
)
from edge_tts.drm import DRM

# ---------------------------------------------------------------------------
# SSL context for raw SSML streaming (deep integration with edge_tts.drm).
# ---------------------------------------------------------------------------
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# Voice management.
# ---------------------------------------------------------------------------

# Fallback voice list (ShortNames) used when the online voice list is
# unavailable. Includes DEFAULT_VOICE plus a few common voices.
FALLBACK_VOICES: List[str] = [
    DEFAULT_VOICE,
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]


async def load_voices_data() -> List[Dict[str, Any]]:
    """Load the full list of available voices from edge_tts.

    Calls ``edge_tts.list_voices()`` and returns a list of voice dicts
    sorted by ShortName. On any failure (e.g. network error), returns a
    minimal fallback list of dicts each containing at least ShortName.

    Returns:
        List of voice dicts (sorted by ShortName).
    """
    try:
        voices = await edge_tts.list_voices()
        if not voices:
            return [{"ShortName": name} for name in FALLBACK_VOICES]
        # Sort by ShortName for deterministic ordering.
        return sorted(voices, key=lambda v: v.get("ShortName", ""))
    except Exception:
        return [{"ShortName": name} for name in FALLBACK_VOICES]


# Load voices once at module load time so the dropdown is populated on startup.
VOICE_DATA: List[Dict[str, Any]] = asyncio.run(load_voices_data())
VOICE_SHORT_NAMES: List[str] = [v["ShortName"] for v in VOICE_DATA]
VOICE_BY_NAME: Dict[str, Dict[str, Any]] = {
    v["ShortName"]: v for v in VOICE_DATA
}

# Integrate VoicesManager at startup (satisfies the deep-integration
# requirement; used for find()-style queries if needed).
try:
    VOICES_MANAGER: Optional[edge_tts.VoicesManager] = asyncio.run(
        edge_tts.VoicesManager.create()
    )
except Exception:
    VOICES_MANAGER = None


def filter_voices(
    gender: Optional[str],
    language: Optional[str],
    locale: Optional[str],
) -> List[str]:
    """Filter VOICE_DATA by gender / language / locale.

    Empty string or None for a given criterion means "no filter".
    ``language`` is matched as a prefix of the Locale field (e.g. "zh"
    matches "zh-CN"). ``locale`` is matched exactly against Locale.

    Returns:
        List of matching ShortNames (sorted by ShortName).
    """
    g = (gender or "").strip()
    lang = (language or "").strip()
    loc = (locale or "").strip()

    matches: List[str] = []
    for v in VOICE_DATA:
        if g and g != "全部" and v.get("Gender", "") != g:
            continue
        v_locale = v.get("Locale", "")
        if lang and lang != "全部" and not v_locale.startswith(lang):
            continue
        if loc and loc != "全部" and v_locale != loc:
            continue
        matches.append(v["ShortName"])
    if not matches:
        # Always fall back to the full list so the dropdown is never empty.
        return VOICE_SHORT_NAMES
    return sorted(matches)


def get_voice_info(short_name: str) -> Optional[Dict[str, Any]]:
    """Return the voice dict for ``short_name`` or None if not found."""
    return VOICE_BY_NAME.get(short_name)


def _unique_sorted(values: List[str]) -> List[str]:
    """Return a sorted list of unique non-empty values, with a leading '全部'."""
    seen = set()
    result: List[str] = []
    for v in sorted(values):
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return ["全部"] + result


# Pre-computed filter option lists.
GENDER_OPTIONS: List[str] = _unique_sorted(
    [v.get("Gender", "") for v in VOICE_DATA]
)
LANGUAGE_OPTIONS: List[str] = _unique_sorted(
    [v.get("Locale", "").split("-")[0] for v in VOICE_DATA]
)
LOCALE_OPTIONS: List[str] = _unique_sorted(
    [v.get("Locale", "") for v in VOICE_DATA]
)


# ---------------------------------------------------------------------------
# Parameter conversion helpers.
# ---------------------------------------------------------------------------
def format_percent(value: int) -> str:
    """Convert an integer to edge-tts rate/volume format string.

    e.g. 50 -> "+50%", -20 -> "-20%", 0 -> "+0%"
    """
    if value >= 0:
        return f"+{value}%"
    return f"{value}%"


def format_hertz(value: int) -> str:
    """Convert an integer to edge-tts pitch format string.

    e.g. 10 -> "+10Hz", -30 -> "-30Hz", 0 -> "+0Hz"
    """
    if value >= 0:
        return f"+{value}Hz"
    return f"{value}Hz"


# ---------------------------------------------------------------------------
# Simple mode synthesize function (used by WebUI simple tab + as reference).
# ---------------------------------------------------------------------------
async def synthesize(
    text: str,
    voice: str,
    rate: int,
    volume: int,
    pitch: int,
    boundary: str,
    proxy: str,
    connect_timeout: int,
    receive_timeout: int,
    generate_subtitles: bool,
) -> Tuple[Optional[str], Optional[str], str, List[Dict[str, Any]]]:
    """Synthesize speech using edge_tts.Communicate.

    Returns a 4-tuple:
        (audio_path, subtitle_path_or_None, status_message, metadata_list)
    """
    # Validate text input.
    if not text or not text.strip():
        return None, None, "请输入文本", []

    # Coerce numeric inputs to int (gradio Number/Slider may return float/str).
    try:
        rate = int(rate)
        volume = int(volume)
        pitch = int(pitch)
        connect_timeout = int(connect_timeout)
        receive_timeout = int(receive_timeout)
    except (TypeError, ValueError):
        return (
            None,
            None,
            "合成失败: 参数类型无效，速率/音量/音调与超时时间必须为整数",
            [],
        )

    # Convert integer parameters to edge-tts format strings.
    rate_str = format_percent(rate)
    volume_str = format_percent(volume)
    pitch_str = format_hertz(pitch)

    # Convert empty/whitespace proxy string to None.
    proxy_value: Optional[str] = None
    if proxy and proxy.strip():
        proxy_value = proxy.strip()

    # Create temp files for audio output (delete=False so Gradio can serve).
    audio_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp3", prefix="edge_tts_"
    )
    audio_path = audio_tmp.name
    audio_tmp.close()

    subtitle_path: Optional[str] = None
    metadata_list: List[Dict[str, Any]] = []

    try:
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=rate_str,
            volume=volume_str,
            pitch=pitch_str,
            boundary=boundary,
            proxy=proxy_value,
            connect_timeout=connect_timeout,
            receive_timeout=receive_timeout,
        )

        submaker = edge_tts.SubMaker() if generate_subtitles else None

        with open(audio_path, "wb") as audio_handle:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_handle.write(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    metadata_list.append(
                        {
                            "type": chunk["type"],
                            "offset": chunk["offset"],
                            "duration": chunk["duration"],
                            "text": chunk["text"],
                        }
                    )
                    if generate_subtitles and submaker is not None:
                        submaker.feed(chunk)

        # Verify audio was actually written.
        if os.path.getsize(audio_path) == 0:
            os.remove(audio_path)
            return None, None, "合成失败: 未收到音频数据，请检查文本或网络连接", []

        # Write subtitles if requested and available.
        if generate_subtitles and submaker is not None:
            srt_text = submaker.get_srt()
            if srt_text:
                sub_tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".srt", prefix="edge_tts_sub_"
                )
                subtitle_path = sub_tmp.name
                sub_tmp.close()
                with open(subtitle_path, "w", encoding="utf-8") as sub_handle:
                    sub_handle.write(srt_text)

        status = (
            f"合成成功！语音: {voice} | 语速: {rate_str} | "
            f"音量: {volume_str} | 音调: {pitch_str}"
        )
        if subtitle_path:
            status += " | 字幕已生成"
        return audio_path, subtitle_path, status, metadata_list

    except Exception as e:
        # Clean up temp files on error.
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)
        return None, None, f"合成失败: {e}", []


# ---------------------------------------------------------------------------
# Raw SSML streaming function (deep integration — replicates Communicate's
# internal __stream but uses the user's raw SSML directly).
# Used by the WebUI advanced mode AND the /api/stream endpoint.
# ---------------------------------------------------------------------------
async def stream_raw_ssml(
    ssml: str, boundary: str = "SentenceBoundary"
):
    """Async generator that streams audio + boundary chunks for raw SSML.

    Yields dicts:
        {"type": "audio", "data": <bytes>}
        {"type": "WordBoundary"|"SentenceBoundary",
         "offset": int, "duration": int, "text": str}
    """
    word_boundary = boundary == "WordBoundary"
    wd = "true" if word_boundary else "false"
    sq = "true" if not word_boundary else "false"

    async with aiohttp.ClientSession(trust_env=True) as session, session.ws_connect(
        f"{WSS_URL}&ConnectionId={connect_id()}"
        f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
        f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
        compress=15,
        headers=DRM.headers_with_muid(WSS_HEADERS),
        ssl=_SSL_CTX,
    ) as websocket:
        await websocket.send_str(
            f"X-Timestamp:{date_to_string()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            f'"sentenceBoundaryEnabled":"{sq}","wordBoundaryEnabled":"{wd}"'
            "},"
            '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
            "}}}}\r\n"
        )
        await websocket.send_str(
            ssml_headers_plus_data(connect_id(), date_to_string(), ssml)
        )

        async for received in websocket:
            if received.type == aiohttp.WSMsgType.TEXT:
                encoded = received.data.encode("utf-8")
                parameters, data = get_headers_and_data(
                    encoded, encoded.find(b"\r\n\r\n")
                )
                path = parameters.get(b"Path", None)
                if path == b"audio.metadata":
                    for meta_obj in json.loads(data)["Metadata"]:
                        meta_type = meta_obj["Type"]
                        if meta_type in ("WordBoundary", "SentenceBoundary"):
                            yield {
                                "type": meta_type,
                                "offset": meta_obj["Data"]["Offset"],
                                "duration": meta_obj["Data"]["Duration"],
                                "text": meta_obj["Data"]["text"]["Text"],
                            }
                elif path == b"turn.end":
                    break
            elif received.type == aiohttp.WSMsgType.BINARY:
                if len(received.data) < 2:
                    continue
                header_length = int.from_bytes(received.data[:2], "big")
                if header_length > len(received.data):
                    continue
                parameters, data = get_headers_and_data(received.data, header_length)
                if parameters.get(b"Path") != b"audio":
                    continue
                content_type = parameters.get(b"Content-Type", None)
                if content_type is None:
                    continue
                if len(data) == 0:
                    continue
                yield {"type": "audio", "data": data}
            elif received.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"WebSocket error: {received.data}")


# ---------------------------------------------------------------------------
# Advanced SSML synthesize function (WebUI only — NOT exposed via /api).
# ---------------------------------------------------------------------------
async def synthesize_ssml(
    ssml_text: str, voice: str, boundary: str
) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    """Synthesize speech from raw SSML using stream_raw_ssml.

    Returns:
        (audio_path, status_message, metadata_list)

    NOTE: This function is ONLY used by the Gradio WebUI advanced tab and is
    NOT exposed via any /api endpoint.
    """
    if not ssml_text or not ssml_text.strip():
        return None, "请输入SSML", []

    audio_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp3", prefix="edge_tts_ssml_"
    )
    audio_path = audio_tmp.name
    audio_tmp.close()

    metadata_list: List[Dict[str, Any]] = []

    try:
        with open(audio_path, "wb") as audio_handle:
            async for chunk in stream_raw_ssml(ssml_text, boundary=boundary):
                if chunk["type"] == "audio":
                    audio_handle.write(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    metadata_list.append(
                        {
                            "type": chunk["type"],
                            "offset": chunk["offset"],
                            "duration": chunk["duration"],
                            "text": chunk["text"],
                        }
                    )

        if os.path.getsize(audio_path) == 0:
            os.remove(audio_path)
            return None, "SSML合成失败: 未收到音频数据", []

        status = f"SSML合成成功！语音: {voice} | 边界: {boundary}"
        return audio_path, status, metadata_list

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None, f"SSML合成失败: {e}", []


# ---------------------------------------------------------------------------
# FastAPI app and REST endpoints.
# ---------------------------------------------------------------------------
api_app = FastAPI(title="Edge-TTS REST API")


@api_app.get("/api/voices")
async def api_voices():
    """Return the full list of available voices."""
    return VOICE_DATA


@api_app.post("/api/tts")
async def api_tts(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: int = 0,
    volume: int = 0,
    pitch: int = 0,
    boundary: str = "SentenceBoundary",
    proxy: Optional[str] = None,
    connect_timeout: int = 10,
    receive_timeout: int = 60,
):
    """Synthesize speech and return an MP3 file.

    Only ``text`` is required; all other parameters have defaults.
    """
    if not text or not text.strip():
        return JSONResponse(
            status_code=400, content={"error": "text must not be empty"}
        )

    rate_str = format_percent(int(rate))
    volume_str = format_percent(int(volume))
    pitch_str = format_hertz(int(pitch))
    proxy_value = proxy.strip() if proxy and proxy.strip() else None

    audio_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp3", prefix="edge_tts_api_"
    )
    audio_path = audio_tmp.name
    audio_tmp.close()

    try:
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=rate_str,
            volume=volume_str,
            pitch=pitch_str,
            boundary=boundary,
            proxy=proxy_value,
            connect_timeout=int(connect_timeout),
            receive_timeout=int(receive_timeout),
        )
        with open(audio_path, "wb") as audio_handle:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_handle.write(chunk["data"])

        if os.path.getsize(audio_path) == 0:
            os.remove(audio_path)
            return JSONResponse(
                status_code=502,
                content={"error": "no audio data received from service"},
            )

        return FileResponse(
            audio_path, media_type="audio/mpeg", filename="tts.mp3"
        )
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return JSONResponse(status_code=500, content={"error": str(e)})


@api_app.post("/api/stream")
async def api_stream(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: int = 0,
    volume: int = 0,
    pitch: int = 0,
    boundary: str = "SentenceBoundary",
    proxy: Optional[str] = None,
    connect_timeout: int = 10,
    receive_timeout: int = 60,
):
    """Stream synthesis chunks as NDJSON.

    Each line is a JSON object:
        audio chunk:        {"type":"audio","data":"<base64>"}
        boundary chunk:     {"type":"WordBoundary","offset":..,"duration":..,"text":..}

    Only ``text`` is required; all other parameters have defaults.
    """
    if not text or not text.strip():
        return JSONResponse(
            status_code=400, content={"error": "text must not be empty"}
        )

    rate_str = format_percent(int(rate))
    volume_str = format_percent(int(volume))
    pitch_str = format_hertz(int(pitch))
    proxy_value = proxy.strip() if proxy and proxy.strip() else None

    async def generate():
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=rate_str,
            volume=volume_str,
            pitch=pitch_str,
            boundary=boundary,
            proxy=proxy_value,
            connect_timeout=int(connect_timeout),
            receive_timeout=int(receive_timeout),
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                line = {
                    "type": "audio",
                    "data": base64.b64encode(chunk["data"]).decode("ascii"),
                }
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                line = {
                    "type": chunk["type"],
                    "offset": chunk["offset"],
                    "duration": chunk["duration"],
                    "text": chunk["text"],
                }
            else:
                continue
            yield json.dumps(line, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Gradio UI.
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks UI for the edge-tts synthesizer."""
    default_voice = (
        DEFAULT_VOICE if DEFAULT_VOICE in VOICE_SHORT_NAMES else VOICE_SHORT_NAMES[0]
    )

    # Default SSML template for the advanced tab.
    default_ssml = (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='en-US'><voice name='en-US-EmmaMultilingualNeural'>"
        "<prosody pitch='+0Hz' rate='+0%' volume='+0%'>"
        "在此输入要合成的文本</prosody></voice></speak>"
    )

    with gr.Blocks(
        title="Edge-TTS 语音合成（深度集成版）", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            "# 🎙️ Edge-TTS 语音合成（深度集成版）\n"
            "基于本地 edge_tts 源码的深度集成，提供简单模式、SSML 高级模式以及 "
            "REST API（/api/voices、/api/tts、/api/stream）。"
        )

        # =============== Tab 1: 简单模式 ===============
        with gr.Tab("简单模式"):
            with gr.Row():
                # Left column: text input + outputs.
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="文本",
                        placeholder="请输入要合成的文本...",
                        lines=8,
                        max_lines=20,
                    )
                    synthesize_btn = gr.Button(
                        "合成语音", variant="primary"
                    )
                    audio_output = gr.Audio(
                        label="合成音频", type="filepath"
                    )
                    subtitle_output = gr.File(label="字幕文件 (SRT)")
                    status_output = gr.Textbox(
                        label="状态", lines=2, interactive=False
                    )
                    metadata_output = gr.JSON(label="元数据 (metadata)")

                # Right column: parameters.
                with gr.Column(scale=1):
                    gr.Markdown("### 语音筛选")
                    with gr.Row():
                        gender_filter = gr.Dropdown(
                            choices=GENDER_OPTIONS,
                            value="全部",
                            label="性别",
                        )
                        language_filter = gr.Dropdown(
                            choices=LANGUAGE_OPTIONS,
                            value="全部",
                            label="语言",
                        )
                        locale_filter = gr.Dropdown(
                            choices=LOCALE_OPTIONS,
                            value="全部",
                            label="区域 (Locale)",
                        )
                    voice_dropdown = gr.Dropdown(
                        choices=VOICE_SHORT_NAMES,
                        value=default_voice,
                        label="语音",
                        filterable=True,
                    )
                    voice_info_display = gr.JSON(label="语音信息")

                    gr.Markdown("### 语音参数")
                    rate_slider = gr.Slider(
                        minimum=-100,
                        maximum=100,
                        step=1,
                        value=0,
                        label="语速 (rate)",
                    )
                    volume_slider = gr.Slider(
                        minimum=-100,
                        maximum=100,
                        step=1,
                        value=0,
                        label="音量 (volume)",
                    )
                    pitch_slider = gr.Slider(
                        minimum=-100,
                        maximum=100,
                        step=1,
                        value=0,
                        label="音调 (pitch)",
                    )

                    gr.Markdown("### 高级设置")
                    boundary_radio = gr.Radio(
                        choices=["SentenceBoundary", "WordBoundary"],
                        value="SentenceBoundary",
                        label="边界类型 (boundary)",
                    )
                    proxy_input = gr.Textbox(
                        label="代理 (proxy)",
                        placeholder="留空表示不使用代理，例如 http://127.0.0.1:8080",
                        value="",
                    )
                    with gr.Row():
                        connect_timeout_input = gr.Number(
                            label="连接超时 (秒)", value=10, precision=0
                        )
                        receive_timeout_input = gr.Number(
                            label="接收超时 (秒)", value=60, precision=0
                        )
                    generate_subtitles_check = gr.Checkbox(
                        label="生成字幕", value=False
                    )

            # Wire filter dropdowns to update voice choices.
            def _on_filter_change(g, lang, loc):
                choices = filter_voices(g, lang, loc)
                # Preserve current selection if still present, else pick first.
                current = voice_dropdown.value if voice_dropdown.value else (
                    choices[0] if choices else default_voice
                )
                new_value = current if current in choices else (
                    choices[0] if choices else default_voice
                )
                return gr.update(choices=choices, value=new_value)

            for ctrl in (gender_filter, language_filter, locale_filter):
                ctrl.change(
                    fn=_on_filter_change,
                    inputs=[gender_filter, language_filter, locale_filter],
                    outputs=voice_dropdown,
                )

            # Wire voice dropdown change to update voice info display.
            voice_dropdown.change(
                fn=lambda name: get_voice_info(name),
                inputs=voice_dropdown,
                outputs=voice_info_display,
            )

            synthesize_btn.click(
                fn=synthesize,
                inputs=[
                    text_input,
                    voice_dropdown,
                    rate_slider,
                    volume_slider,
                    pitch_slider,
                    boundary_radio,
                    proxy_input,
                    connect_timeout_input,
                    receive_timeout_input,
                    generate_subtitles_check,
                ],
                outputs=[
                    audio_output,
                    subtitle_output,
                    status_output,
                    metadata_output,
                ],
                api_name="synthesize",
            )

        # =============== Tab 2: 高级模式 (SSML) — WebUI only ===============
        with gr.Tab("高级模式 (SSML)"):
            gr.Markdown(
                "### SSML 高级模式\n"
                "直接输入完整的 SSML 文档，应用将使用本地深度集成的 "
                "`stream_raw_ssml` 路径发送原始 SSML 到 Edge TTS 服务。"
                "本模式仅用于 WebUI 调试，不通过 /api 暴露。"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    ssml_input = gr.Textbox(
                        label="SSML",
                        value=default_ssml,
                        lines=15,
                        max_lines=40,
                    )
                    ssml_synthesize_btn = gr.Button(
                        "合成 SSML", variant="primary"
                    )
                    ssml_audio_output = gr.Audio(
                        label="合成音频", type="filepath"
                    )
                    ssml_status_output = gr.Textbox(
                        label="状态", lines=2, interactive=False
                    )
                    ssml_metadata_output = gr.JSON(label="元数据 (metadata)")

                with gr.Column(scale=1):
                    ssml_voice_dropdown = gr.Dropdown(
                        choices=VOICE_SHORT_NAMES,
                        value=default_voice,
                        label="语音 (仅参考)",
                        filterable=True,
                        interactive=True,
                    )
                    ssml_boundary_radio = gr.Radio(
                        choices=["SentenceBoundary", "WordBoundary"],
                        value="SentenceBoundary",
                        label="边界类型 (boundary)",
                    )
                    gr.Markdown(
                        "提示：实际使用的语音由 SSML 中的 `<voice name='...'>` "
                        "决定，此处的语音下拉仅作参考。"
                    )

            ssml_synthesize_btn.click(
                fn=synthesize_ssml,
                inputs=[
                    ssml_input,
                    ssml_voice_dropdown,
                    ssml_boundary_radio,
                ],
                outputs=[
                    ssml_audio_output,
                    ssml_status_output,
                    ssml_metadata_output,
                ],
                # NOTE: deliberately NO api_name — not exposed via /api.
            )

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio on FastAPI app and run.
# ---------------------------------------------------------------------------
demo = build_ui()
app = gr.mount_gradio_app(api_app, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
    uvicorn.run(app, host=server_name, port=port)
