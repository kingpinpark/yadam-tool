import os, re, time, json, textwrap, tempfile, requests
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

st.set_page_config(page_title="멀티 AI 야담 스튜디오", layout="wide")
st.title("멀티 AI 야담·민담 올인원 제작 스튜디오")

@st.cache_data(ttl=600)
def get_gemini_models(key):
    try:
        client = genai.Client(api_key=key)
        ms = [m.name.replace("models/", "") for m in client.models.list()
              if "gemini" in m.name.lower() and not any(x in m.name.lower() for x in ["embedding", "imagen", "veo", "aqa", "tts", "audio"])]
        ms.sort(key=lambda x: (0 if "3.8-flash" in x else 1, x))
        return ms[:5] if ms else ["gemini-3.8-flash"]
    except Exception:
        return ["gemini-3.8-flash"]

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
    g_key = st.text_input("Gemini API 키", type="password")
    st.caption("[Gemini 키 무료 발급](https://aistudio.google.com/app/apikey)")
    o_key = st.text_input("OpenAI API 키 (삽화/성우 필수)", type="password")
    st.caption("[OpenAI 키 발급](https://platform.openai.com/api-keys)")
    c_key = st.text_input("Claude API 키", type="password")
    st.caption("[Claude 키 발급](https://console.anthropic.com/settings/keys)")

    ais = []
    if g_key: ais += [f"Gemini ({m})" for m in get_gemini_models(g_key)]
    if c_key: ais += ["Claude (claude-sonnet-4-6)", "Claude (claude-opus-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)", "OpenAI (gpt-4.1)"]
    if not ais: ais = ["위에서 API 키를 입력하세요"]

    st.divider()
    st.header("2. AI 역할 분담")
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [작가 AI]", ais, index=min(1, len(ais)-1))

    st.divider()
    tts_voice = st.selectbox("성우 목소리", ["onyx", "echo", "fable", "alloy", "nova", "shimmer"])
    style_anchor = st.text_area("고정 화풍 태그", "Traditional Korean folk painting style, Joseon Dynasty historical illustration, moody cinematic lighting, ink wash and watercolor --", height=80)

# --- AI 호출 및 자막 함수 ---
class Scene(BaseModel):
    scene_number: int
    narration: str
    image_prompt: str

class YadamScript(BaseModel):
    title: str
    hook_summary: str
    scenes: list[Scene]

def call_ai(label, prompt, is_json=False):
    if label.startswith("Gemini"):
        client = genai.Client(api_key=g_key)
        model = label[8:-1]
        for i in range(3):
            try:
                cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=YadamScript, temperature=0.8) if is_json else None
                return client.models.generate_content(model=model, contents=prompt, config=cfg).text
            except Exception as e:
                if "503" in str(e) and i < 2: time.sleep(3); continue
                raise e
    elif label.startswith("OpenAI"):
        client = OpenAI(api_key=o_key)
        model = label[8:-1]
        p = prompt + '\nJSON만 출력: {"title":"","hook_summary":"","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}' if is_json else prompt
        kw = {"response_format": {"type": "json_object"}} if is_json else {}
        return client.chat.completions.create(model=model, messages=[{"role": "user", "content": p}], **kw).choices[0].message.content
    elif label.startswith("Claude"):
        client = Anthropic(api_key=c_key)
        model = label[8:-1]
        p = prompt + '\n순수 JSON만 출력: {"title":"","hook_summary":"","scenes":[{"scene_number":1,"narration":"","image_prompt":""}]}' if is_json else prompt
        txt = client.messages.create(model=model, max_tokens=4096, messages=[{"role": "user", "content": p}]).content[0].text.strip()
        return re.sub(r"^```json\s*|\s*```$", "", txt, flags=re.MULTILINE).strip() if is_json else txt
    raise ValueError("API 키를 먼저 입력해 주세요.")

def get_yt_text(url):
    m = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", url)
    if not m: raise ValueError("올바른 유튜브 링크가 아닙니다.")
    vid = m.group(1)
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            return " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"]))
        return " ".join(x.text for x in YouTubeTranscriptApi().fetch(vid, languages=["ko", "en"]))
    except Exception as e:
        if any(k in str(e) for k in ["blocking", "IpBlocked", "RequestBlocked"]):
            raise RuntimeError("유튜브가 클라우드 서버 IP를 차단했습니다. 아래 [대본/자막 직접 붙여넣기]를 이용해 주세요.")
        raise RuntimeError(f"자막 추출 실패: {e}")

def analyze_text(txt):
    return call_ai(analyst_ai, f"다음 야담 자막의 1)초반 15초 후킹 기법 2)구어체 문체 특징 3)기승전결 템포 4)벤치마킹 공식 3줄 요약을 분석해 주세요:\n{txt[:15000]}")

# --- MP4 렌더링 함수 ---
def make_sub_img(w, h, text, font_path):
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    font = ImageFont.truetype(font_path, max(26, int(h * 0.042)))
    lines = textwrap.wrap(text.strip(), width=34)
    if not lines: return ov
    sizes = [draw.textbbox((0, 0), l, font=font) for l in lines]
    ws, hs = [b[2]-b[0] for b in sizes], [b[3]-b[1] for b in sizes]
    th = sum(hs) + 10 * (len(lines) - 1)
    bw, bh = min(w - 80, max(ws) + 64), th + 36
    x0, y0 = (w - bw) // 2, h - bh - int(h * 0.06)
    draw.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=14, fill=(10, 10, 15, 185))
    cy = y0 + 18
    for i, l in enumerate(lines):
        draw.text(((w - ws[i]) // 2, cy), l, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        cy += hs[i] + 10
    return ov

def render_mp4(scenes, media_dict, font_path):
    W, H, clips, tmps = 1280, 720, [], []
    try:
        for sc in scenes:
            m = media_dict.get(sc["scene_number"], {})
            if "img" not in m or "aud" not in m: raise ValueError(f"장면 {sc['scene_number']}의 삽화 또는 음성이 없습니다.")
            it, at = tempfile.NamedTemporaryFile(delete=False, suffix=".png"), tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            it.write(m["img"]); it.close(); tmps.append(it.name)
            at.write(m["aud"]); at.close(); tmps.append(at.name)
            ac = AudioFileClip(at.name)
            dur = ac.duration + 0.4
            bimg = Image.open(it.name).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            sub = make_sub_img(W, H, sc["narration"], font_path)
            def frame_fn(t, img=bimg, s=sub, d=dur):
                z = 1.0 + 0.12 * (t / max(d, 0.1))
                nw, nh = int(W * z), int(H * z)
                zm = img.resize((nw, nh), Image.Resampling.BILINEAR)
                cr = zm.crop(((nw - W)//2, (nh - H)//2, (nw + W)//2, (nh + H)//2)).convert("RGBA")
                return np.array(Image.alpha_composite(cr, s).convert("RGB"))
            clips.append(VideoClip(frame_fn, duration=dur).set_audio(ac))
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); out.close()
        fv = concatenate_videoclips(clips, method="compose")
        fv.write_videofile(out.name, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", threads=2, logger=None)
        fv.close()
        for c in clips: c.close()
        return out.name
    finally:
        for f in tmps:
            if os.path.exists(f): os.remove(f)

for k, v in [("analysis", ""), ("scenes_data", None), ("media", {}), ("mp4", None)]:
    if k not in st.session_state: st.session_state[k] = v

# --- 1단계 ---
st.header(f"1단계: 벤치마킹 영상 분석 ({analyst_ai})")
c1, c2 = st.columns([4, 1])
yt_url = c1.text_input("유튜브 영상 링크 입력", placeholder="https://www.youtube.com/watch?v=...")
c2.write(""); c2.write("")
if c2.button("링크로 분석", use_container_width=True):
    try:
        with st.spinner("자막 추출 및 분석 중..."):
            st.session_state.analysis = analyze_text(get_yt_text(yt_url))
    except Exception as e: st.error(str(e))

with st.expander("대본/자막 직접 붙여넣기 (유튜브 IP 차단 시 클릭)"):
    m_txt = st.text_area("유튜브 [스크립트 표시] 내용을 붙여넣으세요", height=130)
    if st.button("붙여넣은 대본 분석"):
        try:
            with st.spinner("대본 분석 중..."):
                st.session_state.analysis = analyze_text(m_txt)
        except Exception as e: st.error(str(e))

if st.session_state.analysis:
    with st.expander("분석된 스토리텔링 공식 보기", expanded=True):
        st.markdown(st.session_state.analysis)

st.divider()

# --- 2단계 ---
st.header(f"2단계: 새로운 야담 대본 집필 ({writer_ai})")
col_t, col_n = st.columns([3, 1])
topic = col_t.text_input("야담 소재 입력", placeholder="예: 산속 폐가에서 만난 여인의 비밀")
sc_cnt = col_n.number_input("장면 수", 3, 15, 6)
if st.button("작가 AI에게 대본 의뢰", type="primary"):
    if topic:
        with st.spinner("대본 집필 및 장면 분할 중..."):
            try:
                guide = f"적용할 분석 공식:\n{st.session_state.analysis}" if st.session_state.analysis else "초반 15초 강렬한 미스터리로 시작하세요."
                gp = f"조선시대 전기수 말투('~했더랍니다','~였지요')로 '{topic}' 주제의 야담을 총 {sc_cnt}개 장면으로 작성하세요. 각 장면 narration은 2~3문장, image_prompt는 영문 묘사(화풍 제외).\n{guide}"
                st.session_state.scenes_data = json.loads(call_ai(writer_ai, gp, is_json=True))
                st.session_state.media = {}; st.session_state.mp4 = None
            except Exception as e: st.error(f"대본 오류: {e}")

# --- 3단계 & 4단계 ---
if st.session_state.scenes_data:
    d = st.session_state.scenes_data
    st.divider()
    st.header("3단계: 대본 수정 & 삽화·성우 생성")
    st.subheader(f"제목: {d['title']}")
    st.info(f"후킹 포인트: {d['hook_summary']}")

    edited = []
    for sc in d["scenes"]:
        n = sc["scene_number"]
        with st.container(border=True):
            st.markdown(f"**장면 {n}**")
            t1, t2 = st.columns(2)
            nar = t1.text_area("내레이션 대본", sc["narration"], key=f"n_{n}", height=100)
            imp = t2.text_area("삽화 프롬프트", f"{style_anchor} {sc['image_prompt']}", key=f"i_{n}", height=100)
            edited.append({"scene_number": n, "narration": nar, "prompt": imp})

            if o_key:
                b1, b2 = st.columns(2)
                if b1.button(f"장면 {n} 삽화 생성", key=f"bi_{n}"):
                    with st.spinner("삽화 생성 중..."):
                        url = OpenAI(api_key=o_key).images.generate(model="dall-e-3", prompt=imp, size="1792x1024", n=1).data[0].url
                        st.session_state.media.setdefault(n, {})["img"] = requests.get(url, timeout=30).content
                if b2.button(f"장면 {n} 성우 녹음", key=f"ba_{n}"):
                    with st.spinner("음성 녹음 중..."):
                        st.session_state.media.setdefault(n, {})["aud"] = OpenAI(api_key=o_key).audio.speech.create(model="tts-1", voice=tts_voice, input=nar).content
                m = st.session_state.media.get(n, {})
                m1, m2 = st.columns(2)
                if "img" in m: m1.image(m["img"], use_container_width=True)
                if "aud" in m: m2.audio(m["aud"], format="audio/mp3")

    if o_key:
        if st.button("모든 장면 삽화 + 성우 음성 일괄 생성", use_container_width=True):
            pb = st.progress(0)
            oai = OpenAI(api_key=o_key)
            for idx, sc in enumerate(edited):
                n = sc["scene_number"]
                pb.progress(idx / len(edited), text=f"장면 {n}/{len(edited)} 작업 중...")
                if "img" not in st.session_state.media.get(n, {}):
                    url = oai.images.generate(model="dall-e-3", prompt=sc["prompt"], size="1792x1024", n=1).data[0].url
                    st.session_state.media.setdefault(n, {})["img"] = requests.get(url, timeout=30).content
                if "aud" not in st.session_state.media.get(n, {}):
                    st.session_state.media.setdefault(n, {})["aud"] = oai.audio.speech.create(model="tts-1", voice=tts_voice, input=sc["narration"]).content
            pb.progress(1.0, text="완료!")
            st.rerun()

    st.divider()
    st.header("4단계: 줌인 효과 & 자막 MP4 영상 완성")
    ready = sum(1 for sc in edited if "img" in st.session_state.media.get(sc["scene_number"], {}) and "aud" in st.session_state.media.get(sc["scene_number"], {}))
    st.write(f"준비된 장면: **{ready} / {len(edited)}개**")
    if st.button("최종 MP4 동영상 합치기", type="primary", disabled=(ready < len(edited))):
        with st.spinner("줌인 효과와 한글 자막을 입혀 MP4 렌더링 중... (약 30초 소요)"):
            try:
                p = render_mp4(edited, st.session_state.media, get_font())
                st.session_state.mp4 = open(p, "rb").read()
                os.remove(p)
            except Exception as e: st.error(f"렌더링 오류: {e}")

    if st.session_state.mp4:
        st.video(st.session_state.mp4)
        st.download_button("완성된 MP4 동영상 다운로드", st.session_state.mp4, "yadam_final.mp4", "video/mp4", type="primary", use_container_width=True)
