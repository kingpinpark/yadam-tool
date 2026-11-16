import os, re, io, time, json, zipfile, textwrap, tempfile, requests
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

st.set_page_config(page_title="멀티 AI 초장편 야담 스튜디오", layout="wide")
st.title("멀티 AI 초장편 야담·민담 제작 스튜디오 (2시간 대작 지원)")

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
    g_key = clean_key(st.text_input("Gemini API 키 (장편 분석 추천)", type="password"))
    st.caption("[Gemini 키 무료 발급](https://aistudio.google.com/app/apikey)")
    o_key = clean_key(st.text_input("OpenAI API 키 (삽화/성우 필수)", type="password"))
    st.caption("[OpenAI 키 발급](https://platform.openai.com/api-keys)")
    c_key = clean_key(st.text_input("Claude API 키 (장편 작가 추천)", type="password"))
    st.caption("[Claude 키 발급](https://console.anthropic.com/settings/keys)")

    ais = []
    if g_key: ais += [f"Gemini ({m})" for m in get_gemini_models(g_key)]
    if c_key: ais += ["Claude (claude-sonnet-4-6)", "Claude (claude-opus-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)", "OpenAI (gpt-4.1)"]
    if not ais: ais = ["위에서 API 키를 입력하세요"]

    st.divider()
    st.header("2. AI 역할 분담")
    analyst_ai = st.selectbox("1단계 [2시간 분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [장편 작가 AI]", ais, index=min(1, len(ais)-1))

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
    chapter_summary: str
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
    full_txt = txt[:150000]  # 최대 15만 자 (약 2시간 분량) 지원
    prompt = f"""다음은 장편 유튜브 야담/민담 영상의 전체 대본(총 {len(full_txt):,}자)입니다. 2시간에 달하는 긴 호흡의 이야기를 처음부터 결말까지 빠짐없이 읽고 아래 5가지를 정밀하게 분석해 주세요.
1) 초반 후킹 기법: 시청자를 붙잡은 첫 1분 도입부 전략
2) 구어체 문체 및 화법: 어미 처리('~했더랍니다' 등), 감정 고조 방식, 인물 대사 처리법
3) 초장편 호흡 유지 비결: 긴 시간 내내 지루하지 않게 중간마다 새로운 갈등과 사건을 배치한 구조
4) 결말 및 여운 처리: 반전, 인과응보, 교훈을 마무리하는 방식
5) 벤치마킹 핵심 공식: 새로운 장편 야담에 그대로 적용할 스토리텔링 공식 요약

[전체 자막]
{full_txt}"""
    return call_ai(analyst_ai, prompt)

# --- ZIP 패키징 및 MP4 렌더링 함수 ---
def build_zip_package(title, chapter, edited_scenes, media_dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        script_lines = [f"제목: {title} ({chapter}부)\n" + "="*40 + "\n"]
        for sc in edited_scenes:
            n = sc["scene_number"]
            prefix = f"{chapter}부_{n:02d}장면"
            script_lines.append(f"[{prefix}]\n{sc['narration']}\n(삽화 프롬프트: {sc['prompt']})\n\n")
            m = media_dict.get(n, {})
            if "img" in m:
                zf.writestr(f"{prefix}.png", m["img"])
            if "aud" in m:
                zf.writestr(f"{prefix}.mp3", m["aud"])
        zf.writestr(f"{chapter}부_전체대본.txt", "".join(script_lines))
    buf.seek(0)
    return buf.getvalue()

def make_sub_img(w, h, text, font_path):
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    font = ImageFont.truetype(font_path, max(24, int(h * 0.038)))
    lines = textwrap.wrap(text.strip(), width=36)[-4:]
    if not lines: return ov
    sizes = [draw.textbbox((0, 0), l, font=font) for l in lines]
    ws, hs = [b[2]-b[0] for b in sizes], [b[3]-b[1] for b in sizes]
    th = sum(hs) + 8 * (len(lines) - 1)
    bw, bh = min(w - 80, max(ws) + 64), th + 32
    x0, y0 = (w - bw) // 2, h - bh - int(h * 0.05)
    draw.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=14, fill=(10, 10, 15, 185))
    cy = y0 + 16
    for i, l in enumerate(lines):
        draw.text(((w - ws[i]) // 2, cy), l, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        cy += hs[i] + 8
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

for k, v in [("analysis", ""), ("scenes_data", None), ("media", {}), ("mp4", None), ("chapter_num", 1), ("story_history", [])]:
    if k not in st.session_state: st.session_state[k] = v

# --- 1단계 ---
st.header(f"1단계: 2시간 초장편 영상 통째로 분석 ({analyst_ai})")
c1, c2 = st.columns([4, 1])
yt_url = c1.text_input("유튜브 영상 링크 입력 (최대 2시간 / 15만 자 지원)", placeholder="https://www.youtube.com/watch?v=...")
c2.write(""); c2.write("")
if c2.button("링크로 전체 분석", use_container_width=True):
    try:
        with st.spinner("장편 자막을 끝까지 추출하고 결말까지 분석 중..."):
            raw_txt = get_yt_text(yt_url)
            st.session_state.analysis = analyze_text(raw_txt)
            st.success(f"총 {len(raw_txt):,}자(약 {max(1, len(raw_txt)//750)}분 분량)의 자막을 결말까지 완벽히 분석했습니다!")
    except Exception as e: st.error(str(e))

with st.expander("대본/자막 직접 붙여넣기 (유튜브 IP 차단 시 클릭 - 최대 150,000자 지원)"):
    m_txt = st.text_area("유튜브 [스크립트 표시] 전체 내용을 붙여넣으세요", height=150)
    if st.button("붙여넣은 장편 대본 분석"):
        try:
            with st.spinner("장편 대본 전체 구조 분석 중..."):
                st.session_state.analysis = analyze_text(m_txt)
                st.success(f"총 {len(m_txt):,}자의 대본을 분석했습니다!")
        except Exception as e: st.error(str(e))

if st.session_state.analysis:
    with st.expander("분석된 장편 스토리텔링 공식 보기", expanded=True):
        st.markdown(st.session_state.analysis)

st.divider()

# --- 2단계 (챕터별 이어쓰기 지원) ---
st.header(f"2단계: 대장편 야담 챕터별 집필 ({writer_ai})")
col_t, col_n, col_len = st.columns([3, 1, 2])
topic = col_t.text_input("야담 대주제 입력", placeholder="예: 몰락한 양반가 막내아들이 전국을 떠돌며 기이한 사건들을 해결하는 대서사시")
sc_cnt = col_n.number_input("이번 챕터 장면 수", 4, 30, 12)
length_mode = col_len.selectbox("장면당 대사 분량", ["초장편 라디오용 (장면당 7~10문장, 극대화 묘사)", "중편 야담용 (장면당 4~5문장)", "단편/숏폼용 (장면당 2~3문장)"])

btn_c1, btn_c2 = st.columns(2)
start_new = btn_c1.button("1부(새로운 이야기) 처음부터 집필하기", type="primary", use_container_width=True)
continue_next = btn_c2.button(
    f"이전 내용에 이어서 [{st.session_state.chapter_num + 1}부] 계속 집필하기",
    disabled=(st.session_state.scenes_data is None),
    use_container_width=True
)

if start_new or continue_next:
    if not topic:
        st.warning("야담 대주제를 먼저 입력해 주세요.")
    else:
        if start_new:
            st.session_state.chapter_num = 1
            st.session_state.story_history = []
        else:
            # 직전 챕터 요약 및 마지막 장면 저장
            prev = st.session_state.scenes_data
            last_nar = prev["scenes"][-1]["narration"] if prev.get("scenes") else ""
            st.session_state.story_history.append(f"[{st.session_state.chapter_num}부 요약]: {prev.get('chapter_summary', '')} / [마지막 장면 대사]: {last_nar}")
            st.session_state.chapter_num += 1

        ch = st.session_state.chapter_num
        with st.spinner(f"대장편 야담 [{ch}부] 대본을 집필 중입니다..."):
            try:
                guide = f"적용할 문체 및 구성 공식:\n{st.session_state.analysis}" if st.session_state.analysis else "구수하고 긴장감 넘치는 조선시대 전기수 화법을 유지하세요."
                if "초장편" in length_mode:
                    detail_rule = "각 장면 narration은 인물 간의 대화, 심리 묘사, 주변 풍경을 소설처럼 아주 상세하게 7~10문장 이상 길게 쓰세요."
                elif "중편" in length_mode:
                    detail_rule = "각 장면 narration은 4~5문장 분량으로 풍부하게 쓰세요."
                else:
                    detail_rule = "각 장면 narration은 2~3문장으로 간결하게 쓰세요."

                history_ctx = ""
                if st.session_state.story_history:
                    history_ctx = "\n[지금까지 진행된 이전 챕터 줄거리 - 반드시 이 사건과 인물에 자연스럽게 이어서 다음 이야기를 전개하세요]\n" + "\n".join(st.session_state.story_history)

                gp = f"""당신은 조선시대 최고의 전기수(이야기꾼)입니다. '{topic}' 주제의 대장편 야담 중 제 [{ch}부]를 총 {sc_cnt}개 장면으로 집필하세요.
{history_ctx}
[집필 규칙]
1. {detail_rule}
2. chapter_summary에는 다음 챕터로 이어질 수 있도록 이번 {ch}부의 핵심 사건과 인물 상태를 3줄로 요약하세요.
3. image_prompt는 각 장면에 맞는 영문 삽화 묘사(화풍 태그 제외).
{guide}"""
                st.session_state.scenes_data = json.loads(call_ai(writer_ai, gp, is_json=True))
                st.session_state.media = {}; st.session_state.mp4 = None
                st.rerun()
            except Exception as e: st.error(f"대본 오류: {e}")

# --- 3단계 & 4단계 ---
if st.session_state.scenes_data:
    d = st.session_state.scenes_data
    ch = st.session_state.chapter_num
    st.divider()
    st.header(f"3단계: [{ch}부] 대본 검수 & 삽화·성우 음성 생성")
    st.subheader(f"제목: {d['title']} - 제 {ch}부")
    st.info(f"이번 챕터 핵심 전개: {d.get('chapter_summary', d.get('hook_summary', ''))}")

    edited = []
    for sc in d["scenes"]:
        n = sc["scene_number"]
        with st.container(border=True):
            st.markdown(f"**{ch}부 - 장면 {n}**")
            t1, t2 = st.columns(2)
            nar = t1.text_area("내레이션 대본", sc["narration"], key=f"n_{ch}_{n}", height=150)
            imp = t2.text_area("삽화 프롬프트", f"{style_anchor} {sc['image_prompt']}", key=f"i_{ch}_{n}", height=150)
            edited.append({"scene_number": n, "narration": nar, "prompt": imp})

            if o_key:
                b1, b2 = st.columns(2)
                if b1.button(f"장면 {n} 삽화 생성", key=f"bi_{ch}_{n}"):
                    with st.spinner("삽화 생성 중..."):
                        url = OpenAI(api_key=o_key).images.generate(model="dall-e-3", prompt=imp, size="1792x1024", n=1).data[0].url
                        st.session_state.media.setdefault(n, {})["img"] = requests.get(url, timeout=30).content
                if b2.button(f"장면 {n} 성우 녹음", key=f"ba_{ch}_{n}"):
                    with st.spinner("음성 녹음 중..."):
                        st.session_state.media.setdefault(n, {})["aud"] = OpenAI(api_key=o_key).audio.speech.create(model="tts-1", voice=tts_voice, input=nar).content
                m = st.session_state.media.get(n, {})
                m1, m2 = st.columns(2)
                if "img" in m: m1.image(m["img"], use_container_width=True)
                if "aud" in m: m2.audio(m["aud"], format="audio/mp3")

    if o_key:
        if st.button(f"[{ch}부] 모든 장면 삽화 + 성우 음성 일괄 생성", use_container_width=True):
            pb = st.progress(0)
            oai = OpenAI(api_key=o_key)
            for idx, sc in enumerate(edited):
                n = sc["scene_number"]
                pb.progress(idx / len(edited), text=f"{ch}부 장면 {n}/{len(edited)} 작업 중...")
                if "img" not in st.session_state.media.get(n, {}):
                    url = oai.images.generate(model="dall-e-3", prompt=sc["prompt"], size="1792x1024", n=1).data[0].url
                    st.session_state.media.setdefault(n, {})["img"] = requests.get(url, timeout=30).content
                if "aud" not in st.session_state.media.get(n, {}):
                    st.session_state.media.setdefault(n, {})["aud"] = oai.audio.speech.create(model="tts-1", voice=tts_voice, input=sc["narration"]).content
            pb.progress(1.0, text="완료!")
            st.rerun()

    st.divider()
    st.header(f"4단계: [{ch}부] 소스 ZIP 다운로드 & MP4 렌더링")
    ready = sum(1 for sc in edited if "img" in st.session_state.media.get(sc["scene_number"], {}) and "aud" in st.session_state.media.get(sc["scene_number"], {}))
    st.write(f"현재 준비된 장면: **{ready} / {len(edited)}개**")

    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        zip_bytes = build_zip_package(d["title"], ch, edited, st.session_state.media)
        st.download_button(
            label=f"캡컷/브루 편집용 [{ch}부] 전체 소스 ZIP 다운로드 (그림+음성+대본)",
            data=zip_bytes,
            file_name=f"yadam_part{ch}_assets.zip",
            mime="application/zip",
            use_container_width=True
        )
    with dl_c2:
        if st.button(f"[{ch}부] 웹에서 바로 MP4 동영상 합치기 (15분 이하 권장)", type="primary", disabled=(ready < len(edited)), use_container_width=True):
            with st.spinner("줌인 효과와 한글 자막을 입혀 MP4 렌더링 중..."):
                try:
                    p = render_mp4(edited, st.session_state.media, get_font())
                    st.session_state.mp4 = open(p, "rb").read()
                    os.remove(p)
                except Exception as e: st.error(f"렌더링 오류: {e}")

    if st.session_state.mp4:
        st.video(st.session_state.mp4)
        st.download_button(f"[{ch}부] 완성된 MP4 동영상 다운로드", st.session_state.mp4, f"yadam_part{ch}.mp4", "video/mp4", type="primary", use_container_width=True)
