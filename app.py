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

st.set_page_config(page_title="Gemini 올인원 장편 야담 스튜디오", layout="wide")
st.title("멀티 AI 초장편 야담·민담 제작 스튜디오 (초고속 엔진)")

def clean_key(k):
    k = (k or "").strip()
    if k and not k.isascii():
        st.sidebar.error("API 키에 한글이나 잘못된 문자가 섞여 있습니다. 영문/숫자 키만 다시 붙여넣어 주세요.")
        return ""
    return k

@st.cache_data(ttl=600)
def get_gemini_models(key):
    try:
        client = genai.Client(api_key=key)
        all_m = [m.name.replace("models/", "") for m in client.models.list()]
        # 구버전(1.5, 2.0, 2.5) 및 특수 목적 모델을 제외하고 최신 모델만 필터링
        txt_m = [
            m for m in all_m
            if "gemini" in m.lower()
            and not any(x in m.lower() for x in ["embedding", "imagen", "veo", "aqa", "tts", "audio", "2.5", "2.0", "1.5"])
        ]
        img_m = [m for m in all_m if "imagen" in m.lower()]
        txt_m.sort(key=lambda x: (0 if "3.8-flash" in x and "lite" not in x else 1, x))
        img_m.sort(reverse=True)
        return (txt_m[:6] if txt_m else ["gemini-3.8-flash"]), (img_m[0] if img_m else "imagen-3.0-generate-002")
    except Exception:
        return ["gemini-3.8-flash"], "imagen-3.0-generate-002"

@st.cache_resource
def get_font():
    p = os.path.join(tempfile.gettempdir(), "NanumGothicBold.ttf")
    if not os.path.exists(p):
        r = requests.get("https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Bold.ttf", timeout=15)
        open(p, "wb").write(r.content)
    return p

# --- 사이드바 ---
with st.sidebar:
    st.header("1. AI 엔진 연결")
    g_key = clean_key(st.text_input("Gemini API 키 (분석·대본·삽화·성우 올인원)", type="password"))
    st.caption("[Google AI Studio에서 키 발급/결제 관리](https://aistudio.google.com/app/apikey)")
    o_key = clean_key(st.text_input("OpenAI API 키 (선택 사항)", type="password"))
    c_key = clean_key(st.text_input("Claude API 키 (선택 사항)", type="password"))

    ais = []
    imagen_model = "imagen-3.0-generate-002"
    if g_key:
        g_txts, imagen_model = get_gemini_models(g_key)
        ais += [f"Gemini ({m})" for m in g_txts]
    if c_key: ais += ["Claude (claude-sonnet-4-6)", "Claude (claude-opus-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)", "OpenAI (gpt-4.1)"]
    if not ais: ais = ["위에서 Gemini API 키를 입력하세요"]

    st.divider()
    st.header("2. AI 역할 분담")
    analyst_ai = st.selectbox("1단계 [분석가 AI] (flash 모델 권장)", ais, index=0)
    writer_ai = st.selectbox("2단계 [장편 작가 AI]", ais, index=0)

    media_engines = []
    if g_key: media_engines.append("Gemini 올인원 (Imagen 삽화 + Gemini 성우)")
    if o_key: media_engines.append("OpenAI (DALL-E 3 삽화 + OpenAI 성우)")
    if not media_engines: media_engines = ["API 키를 먼저 입력하세요"]
    media_engine = st.selectbox("3단계 [삽화 & 성우 엔진]", media_engines, index=0)

    st.divider()
    if "Gemini" in media_engine:
        tts_voice = st.selectbox("Gemini 성우 목소리", ["Charon (중후한 남성)", "Fenrir (묵직한 남성)", "Puck (경쾌한 이야기꾼)", "Aoede (차분한 여성)", "Kore (단아한 여성)"])
    else:
        tts_voice = st.selectbox("OpenAI 성우 목소리", ["onyx", "echo", "fable", "alloy", "nova", "shimmer"])
    style_anchor = st.text_area("고정 화풍 태그", "Traditional Korean folk painting style, Joseon Dynasty historical illustration, moody cinematic lighting, ink wash and watercolor --", height=80)

# --- AI 호출 및 미디어 생성 함수 ---
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
    if label.startswith("Gemini"):
        client = genai.Client(api_key=g_key)
        model = label[8:-1]
        for i in range(3):
            try:
                cfg_args = {}
                if is_json:
                    cfg_args["response_mime_type"] = "application/json"
                    cfg_args["response_schema"] = YadamScript
                    cfg_args["temperature"] = 0.8
                if fast_mode and "flash" in model.lower():
                    cfg_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                cfg = types.GenerateContentConfig(**cfg_args) if cfg_args else None
                return client.models.generate_content(model=model, contents=prompt, config=cfg).text
            except Exception as e:
                if "503" in str(e) and i < 2: time.sleep(3); continue
                raise e
    elif label.startswith("OpenAI"):
        client = OpenAI(api_key=o_key)
        model = label[8:-1]
        p = prompt + '\nJSON만 출력: {"title":"","hook_summary":"","chapter_summary":"이번챕터줄거리요약","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}' if is_json else prompt
        kw = {"response_format": {"type": "json_object"}} if is_json else {}
        return client.chat.completions.create(model=model, messages=[{"role": "user", "content": p}], **kw).choices[0].message.content
    elif label.startswith("Claude"):
        client = Anthropic(api_key=c_key)
        model = label[8:-1]
        p = prompt + '\n순수 JSON만 출력: {"title":"","hook_summary":"","chapter_summary":"이번챕터줄거리요약","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}' if is_json else prompt
        txt = client.messages.create(model=model, max_tokens=8192, messages=[{"role": "user", "content": p}]).content[0].text.strip()
        return re.sub(r"^```json\s*|\s*```$", "", txt, flags=re.MULTILINE).strip() if is_json else txt
    raise ValueError("API 키를 먼저 입력해 주세요.")

def generate_image_bytes(prompt_text):
    if "Gemini" in media_engine:
        client = genai.Client(api_key=g_key)
        res = client.models.generate_images(
            model=imagen_model,
            prompt=prompt_text,
            config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9", output_mime_type="image/png")
        )
        return res.generated_images[0].image.image_bytes
    else:
        url = OpenAI(api_key=o_key).images.generate(model="dall-e-3", prompt=prompt_text, size="1792x1024", n=1).data[0].url
        return requests.get(url, timeout=30).content

def generate_audio_bytes(narration_text):
    if "Gemini" in media_engine:
        client = genai.Client(api_key=g_key)
        v_name = tts_voice.split(" ")[0]
        last_err = None
        for m_name in ["gemini-3.8-flash", "gemini-3.8-flash-preview-tts"]:
            try:
                res = client.models.generate_content(
                    model=m_name,
                    contents=f"옛날이야기를 들려주는 조선시대 전기수처럼 차분하고 몰입감 있는 한국어 억양으로 다음 대사를 읽어주세요:\n{narration_text}",
                    config=types.
