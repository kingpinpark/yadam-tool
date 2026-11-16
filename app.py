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
from moviepy.editor import (
    VideoFileClip, VideoClip, AudioFileClip, CompositeAudioClip
)

st.set_page_config(page_title="야담 & 숏폼 스튜디오", layout="wide")
st.title("🎬 멀티 AI 초장편 야담 & 카테고리 숏폼 스튜디오")

init_keys = [
    ("analysis", ""), ("scenes_data", None), ("media", {}),
    ("ch", 1), ("hist", []), ("shorts_mp4", None)
]
for k, v in init_keys:
    if k not in st.session_state:
        st.session_state[k] = v

def clean_k(k):
    k = (k or "").strip()
    return k if k.isascii() else ""

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
    g_key = clean_k(st.text_input("Gemini API 키", type="password"))
    st.caption("[Gemini 키 발급](https://aistudio.google.com/app/apikey)")
    o_key = clean_k(st.text_input("OpenAI API 키 (선택)", type="password"))
    c_key = clean_k(st.text_input("Claude API 키 (선택)", type="password"))

    ais = []
    if g_key:
        ais += [
            "Gemini (gemini-3.8-flash)",
            "Gemini (gemini-3.8-flash-lite)",
            "Gemini (gemini-3.8-pro)"
        ]
    if c_key:
        ais += ["Claude (claude-sonnet-4-6)"]
    if o_key:
        ais += ["OpenAI (gpt-4o)"]
    if not ais:
        ais = ["Gemini (gemini-3.8-flash)"]

    st.divider()
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [작가 AI]", ais, index=0)
    i_opts = ["Gemini (Imagen 3)", "OpenAI"] if o_key else ["Gemini (Imagen 3)"]
    img_eng = st.selectbox("3단계 [삽화 엔진]", i_opts)
    t_opts = ["초고속 한국어 성우 (1초)", "Gemini 성우", "OpenAI 성우"]
    tts_eng = st.selectbox("3단계 [성우 엔진]", t_opts)

    if "초고속" in tts_eng:
        v_opts = ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural", "ko-KR-SunHiNeural"]
    elif "Gemini
