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
st.title("📜 멀티 AI 초장편 야담 스튜디오 (초고속 엔진)")

for k, v in [("analysis", ""), ("scenes_data", None), ("media", {}), ("mp4", None), ("ch", 1), ("hist", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

def clean_k(k):
    k = (k or "").strip()
    return k if k.isascii() else ""

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("1. AI 엔진 연결")
    g_key = clean_k(st.text_input("Gemini API 키 (올인원)", type="password"))
    st.caption("[Gemini 키 발급](https://aistudio.google.com/app/apikey)")
    o_key = clean_k(st.text_input("OpenAI API 키 (선택)", type="password"))
    c_key = clean_k(st.text_input("Claude API 키 (선택)", type="password"))

    ais = []
    if g_key: ais += ["Gemini (gemini-3.8-flash)", "Gemini (gemini-3.8-pro)"]
    if c_key: ais += ["Claude (claude-sonnet-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)"]
    if not ais: ais = ["Gemini (gemini-3.8-flash)"]

    st.divider()
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [작가 AI]", ais, index=0)
    m_eng = st.selectbox("3단계 [삽화/성우]", ["Gemini 올인원", "OpenAI"] if o_key else ["Gemini 올인원"])
    v_list = ["Charon", "Fenrir", "Puck", "Aoede", "Kore"] if "Gemini" in m_eng else ["onyx", "echo", "fable"]
    tts_voice = st.selectbox("성우 목소리", v_list)
    def_st = "Traditional Korean folk painting, Joseon Dynasty illustration, ink wash and watercolor --"
    style_anchor = st.text_area("고정 화풍", def_st, height=68)

# =========================================================
# [1단계 UI] 유튜브 링크 입력창 (최상단 고정 배치!)
# =========================================================
st.subheader("1단계: 벤치마킹 유튜브 영상 분석")
fast_opt = st.checkbox("⚡ 초고속 핵심 구간 분석 (도입+중반+결말 추출로 3~5초 완료)", value=True)

col_u, col_b = st.columns([4, 1])
yt_url = col_u.text_input("🔗 유튜브 영상 링크를 여기에 붙여넣으세요", placeholder="https://www.youtube.com/watch?v=...")
col_b.write("")
col_b.write("")
btn_url_analyze = col_b.button("🔍 링크 분석", type="primary", use_container_width=True)

with st.expander("📝 대본/자막 직접 붙여넣기 (유튜브 IP 차단 시 클릭)"):
    m_txt = st.text_area("유튜브 [스크립트 표시] 전체 내용을 붙여넣으세요", height=120)
    btn_txt_analyze = st.button("붙여넣은 대본 분석")

# --- 핵심 동작 함수 ---
class Scene(BaseModel):
    scene_number: int
    narration: str
    image_prompt: str

class YadamScript(BaseModel):
    title: str
    chapter_summary: str
    scenes: list[Scene]

def call_ai(label, prompt, is_json=False, fast=False):
    jh = '\nJSON만 출력: {"title":"","chapter_summary":"","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}'
    if label.startswith("Gemini"):
        if not g_key: raise ValueError("왼쪽 사이드바에 Gemini API 키를 입력해 주세요.")
        client = genai.Client(api_key=g_key)
        model = label.split("(")[1][:-1]
        kw = {"response_mime_type": "application/json", "response_schema": YadamScript} if is_json else {}
        if fast and "flash" in model:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        cfg = types.GenerateContentConfig(**kw) if kw else None
        return client.models.generate_content(model=model, contents=prompt, config=cfg).text
    elif label.startswith("OpenAI"):
        client = OpenAI(api_key=o_key)
        p = prompt + jh if is_json else prompt
        kw = {"response_format": {"type": "json_object"}} if is_json else {}
        return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": p}], **kw).choices[0].message.content
    elif label.startswith("Claude"):
        client = Anthropic(api_key=c_key)
        p = prompt + jh if is_json else prompt
        t = client.messages.create(model="claude-sonnet-4-6", max_tokens=8192, messages=[{"role": "user", "content": p}]).content[0].text
        return re.sub(r"^```json\s*|\s*```$", "", t.strip(), flags=re.M).strip() if is_json else t

def get_yt(url):
    m = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url or "")
    if not m: raise ValueError("올바른 유튜브 링크를 입력해 주세요.")
    vid = m.group(1)
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            return " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"]))
        return " ".join(x.text for x in YouTubeTranscriptApi().fetch(vid, languages=["ko", "en"]))
    except Exception:
        raise RuntimeError("유튜브 IP 차단됨: 바로 아래 [대본/자막 직접 붙여넣기]를 열어 스크립트를 붙여넣어 주세요.")

def run_analyze(txt):
    c = re.sub(r"\s+", " ", re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", txt)).strip()
    if fast_opt and len(c) > 30000:
        mid = len(c) // 2
        c = f"[도입]\n{c[:10000]}\n[중반]\n{c[mid-4000:mid+4000]}\n[결말]\n{c[-10000:]}"
    p = f"다음 야담의 1)초반 후킹 기법 2)구어체 화법 3)전개 및 결말 특징 4)벤치마킹 공식 3줄을 요약하세요:\n{c[:120000]}"
    return call_ai(analyst_ai, p, is_json=False, fast=True)

if btn_url_analyze:
    try:
        t0 = time.time()
        with st.spinner("유튜브 자막 추출 및 초고속 분석 중..."):
            st.session_state.analysis = run_analyze(get_yt(yt_url))
            st.success(f"분석 완료! ({time.time()-t0:.1f}초)")
    except Exception as e: st.error(str(e))

if btn_txt_analyze and m_txt.strip():
    try:
        t0 = time.time()
        with st.spinner("대본 초고속 분석 중..."):
            st.session_state.analysis = run_analyze(m_txt)
            st.success(f"분석 완료! ({time.time()-t0:.1f}초)")
    except Exception as e: st.error(str(e))

if st.session_state.analysis:
    with st.expander("📊 분석된 스토리텔링 공식 보기", expanded=True):
        st.markdown(st.session_state.analysis)

# =========================================================
# [2단계 UI] 대장편 야담 대본 집필
# =========================================================
st.divider()
st.subheader("2단계: 대장편 야담 챕터별 대본 집필")
c_t, c_n, c_l = st.columns([3, 1, 2])
topic = c_t.text_input("야담 대주제 입력", placeholder="예: 산속 폐가에서 만난 여인의 비밀")
sc_cnt = c_n.number_input("장면 수", 4, 25, 8)
l_mode = c_l.selectbox("분량", ["중편 (4~5문장)", "초장편 (7~10문장)", "단편 (2~3문장)"])

b1, b2 = st.columns(2)
s_new = b1.button("✍️ 1부 처음부터 집필", type="primary", use_container_width=True)
c_nxt = b2.button(f"이어서 [{st.session_state.ch + 1}부] 집필", disabled=(not st.session_state.scenes_data), use_container_width=True)

if (s_new or c_nxt) and topic:
    if s_new:
        st.session_state.ch, st.session_state.hist = 1, []
    else:
        st.session_state.hist.append(f"[{st.session_state.ch}부]: {st.session_state.scenes_data.get('chapter_summary','')}")
        st.session_state.ch += 1
    ch = st.session_state.ch
    with st.spinner(f"[{ch}부] 대본 집필 중..."):
        try:
            gp = f"조선시대 전기수 말투('~했더랍니다')로 '{topic}' 제 {ch}부를 {sc_cnt}개 장면({l_mode})으로 작성하세요. chapter_summary는 3줄 요약, image_prompt는 영문 묘사.\n이전줄거리:{st.session_state.hist}\n공식:{st.session_state.analysis}"
            st.session_state.scenes_data = json.loads(call_ai(writer_ai, gp, is_json=True))
            st.session_state.media, st.session_state.mp4 = {}, None
            st.rerun()
        except Exception as e: st.error(f"대본 오류: {e}")

# =========================================================
# [3~4단계 UI] 삽화·성우 생성 및 ZIP/MP4 다운로드
# =========================================================
def gen_img(p):
    if "Gemini" in m_eng:
        client = genai.Client(api_key=g_key)
        cfg = types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9", output_mime_type="image/png")
        return client.models.generate_images(model="imagen-3.0-generate-002", prompt=p, config=cfg).generated_images[0].image.image_bytes
    u = OpenAI(api_key=o_key).images.generate(model="dall-e-3", prompt=p, size="1792x1024", n=1).data[0].url
    return requests.get(u, timeout=30).content

def gen_aud(txt):
    if "Gemini" in m_eng:
        client = genai.Client(api_key=g_key)
        vc = types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=tts_voice))
        cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=vc))
        res = client.models.generate_content(model="gemini-3.8-flash", contents=f"조선시대 이야기꾼처럼 차분히 읽어주세요:\n{txt}", config=cfg)
        pcm = res.candidates[0].content.parts[0].inline_data.data
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(pcm)
        return buf.getvalue(), "wav"
    return OpenAI(api_key=o_key).audio.speech.create(model="tts-1", voice=tts_voice, input=txt).content, "mp3"

if st.session_state.scenes_data:
    d, ch = st.session_state.scenes_data, st.session_state.ch
    st.divider()
    st.subheader(f"3단계: [{ch}부] {d['title']} - 삽화 및 성우 생성")
    edited = []
    for sc in d["scenes"]:
        n = sc["scene_number"]
        with st.container(border=True):
            t1, t2 = st.columns(2)
            nar = t1.text_area(f"장면 {n} 대본", sc["narration"], key=f"n_{ch}_{n}", height=100)
            imp = t2.text_area(f"장면 {n} 프롬프트", f"{style_anchor} {sc['image_prompt']}", key=f"i_{ch}_{n}", height=100)
            edited.append({"scene_number": n, "narration": nar, "prompt": imp})
            m = st.session_state.media.get(n, {})
            if "img" in m: t1.image(m["img"], use_container_width=True)
            if "aud" in m: t2.audio(m["aud"], format=f"audio/{m.get('ext','wav')}")

    if st.button(f"⚡ [{ch}부] 모든 장면 삽화 + 성우 한 번에 생성", type="primary", use_container_width=True):
        pb = st.progress(0)
        for idx, sc in enumerate(edited):
            n = sc["scene_number"]
            pb.progress(idx / len(edited), text=f"장면 {n} 생성 중...")
            if "img" not in st.session_state.media.get(n, {}):
                st.session_state.media.setdefault(n, {})["img"] = gen_img(sc["prompt"])
            if "aud" not in st.session_state.media.get(n, {}):
                ab, ext = gen_aud(sc["narration"])
                st.session_state.media.setdefault(n, {}).update({"aud": ab, "ext": ext})
        pb.progress(1.0, text="생성 완료!"); st.rerun()

    z_buf = io.BytesIO()
    with zipfile.ZipFile(z_buf, "w") as zf:
        for sc in edited:
            n, m = sc["scene_number"], st.session_state.media.get(sc["scene_number"], {})
            if "img" in m: zf.writestr(f"part{ch}_{n:02d}.png", m["img"])
            if "aud" in m: zf.writestr(f"part{ch}_{n:02d}.{m.get('ext','wav')}", m["aud"])
    st.download_button(f"📥 [{ch}부] 캡컷/브루용 전체 소스 ZIP 다운로드", z_buf.getvalue(), f"yadam_part{ch}.zip", "application/zip", use_container_width=True)
