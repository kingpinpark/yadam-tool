import os, re, io, wave, time, json, zipfile, textwrap, tempfile, requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips

st.set_page_config(page_title="야담 스튜디오", layout="wide")
st.title("멀티 AI 초장편 야담 스튜디오 (초고속 엔진)")

def clean_key(k):
    k = (k or "").strip()
    if k and not k.isascii():
        st.sidebar.error("API 키에 한글이 섞여 있습니다.")
        return ""
    return k

@st.cache_data(ttl=600)
def get_gemini_models(key):
    try:
        client = genai.Client(api_key=key)
        all_m = [m.name.replace("models/", "") for m in client.models.list()]
        bad = ["embedding", "imagen", "veo", "aqa", "tts", "audio", "2.5", "2.0", "1.5"]
        txt_m = [
            m for m in all_m
            if "gemini" in m.lower()
            and not any(x in m.lower() for x in bad)
        ]
        img_m = [m for m in all_m if "imagen" in m.lower()]
        txt_m.sort(key=lambda x: (0 if "3.8-flash" in x and "lite" not in x else 1, x))
        img_m.sort(reverse=True)
        t_res = txt_m[:6] if txt_m else ["gemini-3.8-flash"]
        i_res = img_m[0] if img_m else "imagen-3.0-generate-002"
        return t_res, i_res
    except Exception:
        return ["gemini-3.8-flash"], "imagen-3.0-generate-002"

@st.cache_resource
def get_font():
    p = os.path.join(tempfile.gettempdir(), "NanumGothicBold.ttf")
    if not os.path.exists(p):
        u1 = "https://raw.githubusercontent.com/google/fonts/"
        u2 = "main/ofl/nanumgothic/NanumGothic-Bold.ttf"
        open(p, "wb").write(requests.get(u1 + u2, timeout=15).content)
    return p

with st.sidebar:
    st.header("1. AI 엔진 연결")
    g_key = clean_key(st.text_input("Gemini API 키 (올인원)", type="password"))
    st.caption("[Gemini 키 발급](https://aistudio.google.com/app/apikey)")
    o_key = clean_key(st.text_input("OpenAI API 키 (선택)", type="password"))
    c_key = clean_key(st.text_input("Claude API 키 (선택)", type="password"))

    ais, imagen_model = [], "imagen-3.0-generate-002"
    if g_key:
        g_txts, imagen_model = get_gemini_models(g_key)
        ais += [f"Gemini ({m})" for m in g_txts]
    if c_key:
        ais += ["Claude (claude-sonnet-4-6)", "Claude (claude-opus-4-6)"]
    if o_key:
        ais += ["OpenAI (gpt-4o)", "OpenAI (gpt-4.1)"]
    if not ais:
        ais = ["위에서 Gemini API 키를 입력하세요"]

    st.divider()
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [장편 작가 AI]", ais, index=0)

    m_engs = []
    if g_key:
        m_engs.append("Gemini 올인원 (Imagen + Gemini 성우)")
    if o_key:
        m_engs.append("OpenAI (DALL-E 3 + OpenAI 성우)")
    media_engine = st.selectbox("3단계 [삽화 & 성우]", m_engs or ["키 필요"], index=0)

    st.divider()
    g_voices = [
        "Charon (중후한 남성)", "Fenrir (묵직한 남성)",
        "Puck (이야기꾼)", "Aoede (차분한 여성)", "Kore (단아한 여성)"
    ]
    o_voices = ["onyx", "echo", "fable", "alloy", "nova", "shimmer"]
    v_list = g_voices if "Gemini" in media_engine else o_voices
    tts_voice = st.selectbox("성우 목소리", v_list)
    def_style = (
        "Traditional Korean folk painting style, "
        "Joseon Dynasty historical illustration, "
        "ink wash and watercolor --"
    )
    style_anchor = st.text_area("고정 화풍 태그", def_style, height=75)

class Scene(BaseModel):
    scene_number: int
    narration: str
    image_prompt: str

class YadamScript(BaseModel):
    title: str
    hook_summary: str
    chapter_summary: str
    scenes: list[Scene]

def call_ai(label, prompt, is_json=False, fast_mode=False):
    j_hint = (
        '\nJSON만 출력: {"title":"","hook_summary":"",'
        '"chapter_summary":"","scenes":[{"scene_number":1,'
        '"narration":"","image_prompt":""}]}'
    )
    if label.startswith("Gemini"):
        client = genai.Client(api_key=g_key)
