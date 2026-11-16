import os
import re
import time
import json
import textwrap
import tempfile
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="멀티 AI 야담·민담 스튜디오", page_icon="📜", layout="wide")
st.title("📜 멀티 AI 야담·민담 올인원 제작 스튜디오")
st.caption("영상 분석 ➔ 대본 집필 ➔ 삽화·성우 생성 ➔ 줌인 효과 & 자막 MP4 영상 완성까지 한 번에 처리합니다.")

# Gemini 사용 가능 모델 자동 조회
@st.cache_data(ttl=600)
def get_available_gemini_models(key: str):
    try:
        client = genai.Client(api_key=key)
        models = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name.lower() and not any(x in name.lower() for x in ["embedding", "imagen", "veo", "aqa", "tts", "audio"]):
                models.append(name)
        models.sort(key=lambda x: (0 if "3.8-flash" in x else 1, x))
        return models if models else ["gemini-3.8-flash"]
    except Exception:
        return ["gemini-3.8-flash"]

# 한글 자막용 폰트 자동 준비
@st.cache_resource
def get_korean_font_path():
    font_path = os.path.join(tempfile.gettempdir(), "NanumGothicBold.ttf")
    if not os.path.exists(font_path):
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
        res = requests.get(url, timeout=15)
        with open(font_path, "wb") as f:
            f.write(res.content)
    return font_path

# ==========================================
# 사이드바: 여러 AI API 키 연결 (발급 링크 포함) 및 설정
# ==========================================
with st.sidebar:
    st.header("🔑 1. AI 엔진 연결")
    
    gemini_key = st.text_input("Gemini API 키 (Google)", type="password", placeholder="AIzaSy...")
    st.caption("👉 [Google AI Studio에서 Gemini 키 무료 발급받기](https://aistudio.google.com/app/apikey)")
    
    openai_key = st.text_input("OpenAI API 키 (GPT / 삽화 / 성우)", type="password", placeholder="sk-...")
    st.caption("👉 [OpenAI 플랫폼에서 API 키 발급받기](https://platform.openai.com/api-keys)")
    
    claude_key = st.text_input("Claude API 키 (Anthropic)", type="password", placeholder="sk-ant-...")
    st.caption("👉 [Anthropic 콘솔에서 Claude 키 발급받기](https://console.anthropic.com/settings/keys)")
    
    connected_ais = []
    if gemini_key:
        for gm in get_available_gemini_models(gemini_key)[:4]:
            connected_ais.append(f"Gemini ({gm})")
    if claude_key:
        connected_ais.extend([
            "Claude (claude-sonnet-4-6)",
            "Claude (claude-opus-4-6)"
        ])
    if openai_key:
        connected_ais.extend([
            "OpenAI (gpt-4o)",
            "OpenAI (gpt-4.1)"
        ])
    if not connected_ais:
        connected_ais = ["위에서 API 키를 1개 이상 입력하세요"]

    st.divider()
    st.header("🤝 2. AI 역할 분담")
    analyst_ai = st.selectbox("🔍 1단계 [영상 분석가] AI", options=connected_ais, index=0)
    writer_ai = st.selectbox(
        "✍️ 2단계 [야담 작가] AI",
        options=connected_ais,
        index=min(1, len(connected_ais) - 1)
    )

    st.divider()
    st.subheader("🎙️ 성우 목소리 & 화풍 설정")
    tts_voice = st.selectbox(
        "성우 목소리 톤 (OpenAI TTS)",
        options=["onyx", "echo", "fable", "alloy", "nova", "shimmer"],
        index=0,
        help="onyx: 중후하고 차분한 남성 내레이션 추천 / fable: 이야기꾼 톤"
    )
    style_anchor = st.text_area(
        "고정 화풍 태그 (Style Anchor)",
        value="Traditional Korean folk painting style, Joseon Dynasty historical illustration, moody cinematic lighting, ink wash and soft watercolor texture --",
        height=90
    )

# ==========================================
# 통합 AI 호출 엔진
# ==========================================
class Scene(BaseModel):
    scene_number: int = Field(description="장면 번호 (1부터 순차 증가)")
    narration: str = Field(description="전기수(옛날 이야기꾼) 말투의 한국어 내레이션 대사")
    image_prompt: str = Field(description="해당 장면을 묘사하는 구체적인 영문 이미지 생성 프롬프트 (화풍 태그 제외, 인물/배경/행동 중심)")

class YadamScript(BaseModel):
    title: str = Field(description="클릭을 유도하는 매력적인 야담 유튜브 제목")
    hook_summary: str = Field(description="초반 15초 시선 끌기(후킹) 전략 요약")
    scenes: list[Scene] = Field(description="장면별로 분할된 대본 및 삽화 프롬프트 리스트")

def call_selected_ai(ai_label: str, prompt: str, is_json: bool = False) -> str:
    if ai_label.startswith("Gemini"):
        if not gemini_key:
            raise ValueError("Gemini API 키를 입력해 주세요.")
        model_name = ai_label.replace("Gemini (", "").rstrip(")")
        client = genai.Client(api_key=gemini_key)
        for attempt in range(3):
            try:
                if is_json:
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=YadamScript,
                        temperature=0.8
                    )
                    res = client.models.generate_content(model=model_name, contents=prompt, config=config)
                else:
                    res = client.models.generate_content(model=model_name, contents=prompt)
                return res.text
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(3)
                    continue
                raise e

    elif ai_label.startswith("OpenAI"):
        if not openai_key:
            raise ValueError("OpenAI API 키를 입력해 주세요.")
        model_name = ai_label.replace("OpenAI (", "").rstrip(")")
        client = OpenAI(api_key=openai_key)
        if is_json:
            json_instruction = prompt + '\n\n반드시 아래 JSON 형식으로만 응답하세요:\n{"title": "제목", "hook_summary": "후킹요약", "scenes": [{"scene_number": 1, "narration": "대사", "image_prompt": "영문묘사"}]}'
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": json_instruction}],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            return res.choices[0].message.content
        else:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content

    elif ai_label.startswith("Claude"):
        if not claude_key:
            raise ValueError("Claude API 키를 입력해 주세요.")
        model_name = ai_label.replace("Claude (", "").rstrip(")")
        client = Anthropic(api_key=claude_key)
        if is_json:
            json_instruction = prompt + '\n\n다른 설명 없이 반드시 아래 순수 JSON 형식으로만 응답하세요:\n{"title": "제목", "hook_summary": "후킹요약", "scenes": [{"scene_number": 1, "narration": "대사", "image_prompt": "영문묘사"}]}'
            res = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": json_instruction}],
                temperature=0.8
            )
            text = res.content[0].text.strip()
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
            return text
        else:
            res = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.content[0].text
    else:
        raise ValueError("왼쪽 사이드바에서 AI API 키를 먼저 입력해 주세요.")

# 유튜브 자막 추출 함수
def extract_video_id(url: str):
    pattern = r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_youtube_transcript(video_id: str) -> str:
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            return " ".join([item['text'] for item in transcript_list])
        else:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id, languages=['ko', 'en'])
            return " ".join([snippet.text for snippet in transcript])
    except Exception as e:
        err_msg = str(e)
        if "blocking requests" in err_msg or "IpBlocked" in err_msg or "RequestBlocked" in err_msg:
            raise RuntimeError("유튜브가 무료 웹서버 IP의 자동 추출을 차단했습니다. 오른쪽 [📝 대본/자막 직접 붙여넣기] 탭을 이용해 주세요!")
        raise RuntimeError(f"자막 추출 실패: {err_msg}")

# ==========================================
# MP4 동영상 렌더링 엔진 (줌인 효과 + 한글 자막)
# ==========================================
def create_subtitle_overlay(width: int, height: int, text: str, font_path: str) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(26, int(height * 0.042))
    font = ImageFont.truetype(font_path, font_size)

    wrapped_lines = textwrap.wrap(text.strip(), width=34)
    if not wrapped_lines:
        return overlay

    line_heights = []
    line_widths
