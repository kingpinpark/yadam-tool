import os, re, io, wave, time, json, asyncio, zipfile, tempfile, requests
import numpy as np
import streamlit as st
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip, VideoClip, AudioFileClip, CompositeAudioClip

st.set_page_config(page_title="야담 & 카테고리 숏폼 스튜디오", layout="wide")
st.title("🎬 멀티 AI 초장편 야담 & 카테고리 숏폼 스튜디오")

for k, v in [("analysis", ""), ("scenes_data", None), ("media", {}), ("ch", 1), ("hist", []), ("shorts_mp4", None)]:
    if k not in st.session_state: st.session_state[k] = v

def clean_k(k):
    k = (k or "").strip()
    return k if k.isascii() else ""

@st.cache_resource
def get_font():
    p = os.path.join(tempfile.gettempdir(), "NanumGothicBold.ttf")
    if not os.path.exists(p):
        u = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
        open(p, "wb").write(requests.get(u, timeout=15).content)
    return p

with st.sidebar:
    st.header("1. AI 엔진 연결")
    g_key = clean_k(st.text_input("Gemini API 키", type="password"))
    st.caption("[Gemini 키 발급](https://aistudio.google.com/app/apikey)")
    o_key = clean_k(st.text_input("OpenAI API 키 (선택)", type="password"))
    c_key = clean_k(st.text_input("Claude API 키 (선택)", type="password"))

    ais = ["Gemini (gemini-3.8-flash)", "Gemini (gemini-3.8-flash-lite)", "Gemini (gemini-3.8-pro)"] if g_key else []
    if c_key: ais += ["Claude (claude-sonnet-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)"]
    if not ais: ais = ["Gemini (gemini-3.8-flash)"]

    st.divider()
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [작가 AI]", ais, index=0)
    img_eng = st.selectbox("3단계 [삽화 엔진]", ["Gemini (Imagen 3)", "OpenAI (DALL-E 3)"] if o_key else ["Gemini (Imagen 3)"])
    tts_eng = st.selectbox("3단계 [성우 엔진]", ["초고속 한국어 성우 (1초 생성)", "Gemini 성우", "OpenAI 성우"])
    v_opts = ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural", "ko-KR-SunHiNeural"] if "초고속" in tts_eng else (["Charon", "Fenrir", "Aoede"] if "Gemini" in tts_eng else ["onyx", "echo", "fable"])
    tts_voice = st.selectbox("야담 목소리", v_opts)
    style_anchor = st.text_area("고정 화풍", "Traditional Korean folk painting, Joseon Dynasty illustration, ink wash and watercolor --", height=65)

class Scene(BaseModel):
    scene_number: int
    narration: str
    image_prompt: str

class YadamScript(BaseModel):
    title: str
    chapter_summary: str
    scenes: list[Scene]

def get_gclient(t_sec=35):
    if not g_key: raise ValueError("사이드바에 Gemini API 키를 입력해 주세요.")
    return genai.Client(api_key=g_key, http_options=types.HttpOptions(timeout=t_sec * 1000))

def call_ai(label, prompt, is_json=False, fast=False):
    jh = '\nJSON만 출력: {"title":"","chapter_summary":"","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}'
    if label.startswith("Gemini"):
        client = get_gclient(45)
        pri = label.split("(")[1][:-1]
        cands = [pri] + [m for m in ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.8-pro"] if m != pri]
        for m_name in cands:
            for att in range(2):
                try:
                    kw = {"response_mime_type": "application/json", "response_schema": YadamScript} if is_json else {}
                    if fast and "flash" in m_name: kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                    cfg = types.GenerateContentConfig(**kw) if kw else None
                    return client.models.generate_content(model=m_name, contents=prompt, config=cfg).text
                except Exception as e:
                    last_e = e
                    if any(k in str(e) for k in ["503", "UNAVAILABLE", "429"]): time.sleep(2); continue
                    break
        raise RuntimeError(f"구글 서버 혼잡(503): {last_e}")
    elif label.startswith("OpenAI"):
        p = prompt + jh if is_json else prompt
        kw = {"response_format": {"type": "json_object"}} if is_json else {}
        return OpenAI(api_key=o_key, timeout=45.0).chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": p}], **kw).choices[0].message.content
    else:
        p = prompt + jh if is_json else prompt
        t = Anthropic(api_key=c_key, timeout=45.0).messages.create(model="claude-sonnet-4-6", max_tokens=8192, messages=[{"role": "user", "content": p}]).content[0].text
        return re.sub(r"^```json\s*|\s*```$", "", t.strip(), flags=re.M).strip() if is_json else t

tab_yadam, tab_shorts = st.tabs(["📜 초장편 야담 스튜디오", "📱 카테고리형 숏폼 제작 (608×1080)"])

# --- 탭 1: 초장편 야담 ---
with tab_yadam:
    st.subheader("1단계: 벤치마킹 유튜브 영상 분석")
    fast_opt = st.checkbox("⚡ 초고속 핵심 구간 분석 (3~5초 완료)", value=True)
    cu, cb = st.columns([4, 1])
    yt_url = cu.text_input("🔗 유튜브 영상 링크 입력")
    cb.write(""); cb.write("")
    if cb.button("🔍 링크 분석", type="primary", use_container_width=True):
        try:
            m = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", yt_url or "")
            if not m: raise ValueError("올바른 링크를 입력하세요.")
            vid = m.group(1)
            raw = " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"])) if hasattr(YouTubeTranscriptApi, "get_transcript") else " ".join(x.text for x in YouTubeTranscriptApi().fetch(vid, languages=["ko", "en"]))
            st.session_state.analysis = call_ai(analyst_ai, f"다음 야담의 1)후킹 2)화법 3)전개 4)공식 3줄 요약
