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

st.set_page_config(page_title="야담 & 숏폼 스튜디오", layout="wide")
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
    o_key = clean_k(st.text_input("OpenAI API 키 (선택)", type="password"))
    c_key = clean_k(st.text_input("Claude API 키 (선택)", type="password"))
    ais = ["Gemini (gemini-3.8-flash)", "Gemini (gemini-3.8-flash-lite)", "Gemini (gemini-3.8-pro)"] if g_key else []
    if c_key: ais += ["Claude (claude-sonnet-4-6)"]
    if o_key: ais += ["OpenAI (gpt-4o)"]
    if not ais: ais = ["Gemini (gemini-3.8-flash)"]
    st.divider()
    analyst_ai = st.selectbox("1단계 [분석가 AI]", ais, index=0)
    writer_ai = st.selectbox("2단계 [작가 AI]", ais, index=0)
    img_eng = st.selectbox("3단계 [삽화 엔진]", ["Gemini (Imagen 3)", "OpenAI"] if o_key else ["Gemini (Imagen 3)"])
    tts_eng = st.selectbox("3단계 [성우 엔진]", ["초고속 한국어 성우 (1초)", "Gemini 성우", "OpenAI 성우"])
    v_opts = ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural", "ko-KR-SunHiNeural"] if "초고속" in tts_eng else (["Charon", "Fenrir", "Aoede"] if "Gemini" in tts_eng else ["onyx", "echo", "fable"])
    tts_voice = st.selectbox("야담 목소리", v_opts)
    style_anchor = st.text_area("고정 화풍", "Traditional Korean folk painting, Joseon Dynasty illustration, ink wash --", height=65)

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
            for _ in range(2):
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
        return re.sub(r"^```json\s*|\s*```$", "", t.strip(), flags=re.M).strip()

tab_yadam, tab_shorts = st.tabs(["📜 초장편 야담 스튜디오", "📱 카테고리형 숏폼 제작 (608×1080)"])

tab_yadam, tab_shorts = st.tabs([
    "📜 초장편 야담 스튜디오",
    "📱 카테고리형 숏폼 스튜디오 (직접 편집 지원)"
])

with tab_yadam:
    st.subheader("1단계: 벤치마킹 유튜브 영상 분석")
    fast_opt = st.checkbox("⚡ 초고속 핵심 구간 분석 (3~5초)", value=True)
    cu, cb = st.columns([4, 1])
    yt_url = cu.text_input("🔗 유튜브 영상 링크 입력")
    cb.write(""); cb.write("")
    if cb.button("🔍 링크 분석", type="primary", use_container_width=True):
        try:
            m = re.search(r"(?:v=|\/|shorts\/)([0-9A-Za-z_-]{11})", yt_url or "")
            if not m: raise ValueError("올바른 링크를 입력하세요.")
            vid = m.group(1)
            raw = (
                " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"]))
                if hasattr(YouTubeTranscriptApi, "get_transcript")
                else " ".join(x.text for x in YouTubeTranscriptApi().fetch(vid, languages=["ko", "en"]))
            )
            st.session_state.analysis = call_ai(analyst_ai, f"야담의 1)후킹 2)화법 3)전개 4)공식 3줄 요약:\n{raw[:25000]}", fast=True)
            st.success("분석 완료!")
        except Exception as e:
            st.error(f"자막 추출 실패(아래 직접 붙여넣기 이용): {e}")

    with st.expander("📝 대본/자막 직접 붙여넣기"):
        m_txt = st.text_area("스크립트 붙여넣기", height=100)
        if st.button("붙여넣은 대본 분석") and m_txt.strip():
            c = m_txt[:15000] + "\n" + m_txt[-10000:] if (fast_opt and len(m_txt) > 28000) else m_txt[:100000]
            st.session_state.analysis = call_ai(analyst_ai, f"야담의 1)후킹 2)화법 3)전개 4)공식 3줄 요약:\n{c}", fast=True)

    if st.session_state.analysis:
        with st.expander("📊 분석 공식 보기", expanded=True):
            st.markdown(st.session_state.analysis)

    st.divider()
    st.subheader("2단계: 대장편 야담 챕터별 대본 집필")
    ct, cn, cl = st.columns([3, 1, 2])
    topic = ct.text_input("야담 대주제 입력")
    sc_cnt = cn.number_input("장면 수", 4, 25, 6)
    l_mode = cl.selectbox("분량", ["중편 (4~5문장)", "초장편 (7~10문장)", "단편 (2~3문장)"])
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
        with st.spinner(f"[{ch}부] 집필 중..."):
            gp = f"조선시대 전기수 말투('~했더랍니다')로 '{topic}' 제 {ch}부를 {sc_cnt}개 장면({l_mode})으로 작성. chapter_summary는 3줄 요약, image_prompt는 영문 묘사.\n이전:{st.session_state.hist}\n공식:{st.session_state.analysis}"
            st.session_state.scenes_data = json.loads(call_ai(writer_ai, gp, is_json=True, fast=True))
            st.session_state.media = {}; st.rerun()

    def gen_img(p):
        if "Gemini" in img_eng:
            client = get_gclient(30)
            cfg = types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9", output_mime_type="image/png")
            for m_n in ["imagen-3.0-fast-generate-001", "imagen-3.0-generate-002"]:
                try: return client.models.generate_images(model=m_n, prompt=p, config=cfg).generated_images[0].image.image_bytes
                except Exception as e: last_e = e; time.sleep(2)
            raise RuntimeError(f"Imagen 오류: {last_e}")
        u = OpenAI(api_key=o_key, timeout=30.0).images.generate(model="dall-e-3", prompt=p, size="1792x1024", n=1).data[0].url
        return requests.get(u, timeout=20).content

    def gen_aud(txt):
        if "초고속" in tts_eng:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3"); tmp.close()
            asyncio.run(edge_tts.Communicate(txt, tts_voice, rate="-5%").save(tmp.name))
            d = open(tmp.name, "rb").read(); os.remove(tmp.name); return d, "mp3"
        elif "Gemini" in tts_eng:
            vc = types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=tts_voice))
            cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=vc))
            res = get_gclient(25).models.generate_content(model="gemini-3.8-flash", contents=f"차분히 읽기:\n{txt}", config=cfg)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(res.candidates[0].content.parts[0].inline_data.data)
            return buf.getvalue(), "wav"
        return OpenAI(api_key=o_key, timeout=25.0).audio.speech.create(model="tts-1", voice=tts_voice, input=txt).content, "mp3"

    if st.session_state.scenes_data:
        d, ch = st.session_state.scenes_data, st.session_state.ch
        st.divider(); st.subheader(f"3단계: [{ch}부] {d['title']}")
        edited = []
        for sc in d["scenes"]:
            n = sc["scene_number"]
            with st.container(border=True):
                t1, t2 = st.columns(2)
                nar = t1.text_area(f"장면 {n} 대본", sc["narration"], key=f"n_{ch}_{n}", height=85)
                imp = t2.text_area(f"장면 {n} 프롬프트", f"{style_anchor} {sc['image_prompt']}", key=f"i_{ch}_{n}", height=85)
                edited.append({"scene_number": n, "narration": nar, "prompt": imp})
                if t1.button(f"🎙️ 장면 {n} 성우", key=f"ba_{ch}_{n}"):
                    ab, ext = gen_aud(nar); st.session_state.media.setdefault(n, {}).update({"aud": ab, "ext": ext})
                if t2.button(f"🎨 장면 {n} 삽화", key=f"bi_{ch}_{n}"):
                    st.session_state.media.setdefault(n, {})["img"] = gen_img(imp)
                m = st.session_state.media.get(n, {})
                if "aud" in m: t1.audio(m["aud"], format=f"audio/{m.get('ext','mp3')}")
                if "img" in m: t2.image(m["img"], use_container_width=True)

        if st.button(f"⚡ [{ch}부] 모든 장면 일괄 생성", type="primary", use_container_width=True):
            pb = st.progress(0)
            for idx, sc in enumerate(edited):
                n = sc["scene_number"]
                if "aud" not in st.session_state.media.get(n, {}):
                    try: ab, ext = gen_aud(sc["narration"]); st.session_state.media.setdefault(n, {}).update({"aud": ab, "ext": ext})
                    except Exception as e: st.warning(f"장면 {n} 성우 오류: {e}")
                if "img" not in st.session_state.media.get(n, {}):
                    try: st.session_state.media.setdefault(n, {})["img"] = gen_img(sc["prompt"]); time.sleep(1.2)
                    except Exception as e: st.warning(f"장면 {n} 삽화 오류: {e}")
                pb.progress((idx + 1) / len(edited))
            st.rerun()

        z_buf = io.BytesIO()
        with zipfile.ZipFile(z_buf, "w") as zf:
            for sc in edited:
                n, m = sc["scene_number"], st.session_state.media.get(sc["scene_number"], {})
                if "img" in m: zf.writestr(f"part{ch}_{n:02d}.png", m["img"])
                if "aud" in m: zf.writestr(f"part{ch}_{n:02d}.{m.get('ext','mp3')}", m["aud"])
        st.download_button(f"📥 [{ch}부] 소스 ZIP 다운로드", z_buf.getvalue(), f"yadam_part{ch}.zip", "application/zip", use_container_width=True)

# =========================================================
# [탭 2] 카테고리형 숏폼 제작 + 결과물 실시간 직접 수정 에디터
# =========================================================
with tab_shorts:
    st.subheader("📱 카테고리형 숏폼 제작 & 실시간 편집기 (608×1080)")
    sc1, sc2 = st.columns([2, 1])
    up_vid = sc1.file_uploader("1. 원본 영상 업로드 (.mp4, .mov)", type=["mp4", "mov", "mkv"])
    s_lang = sc2.selectbox("언어 & 성우 (+50%)", ["한국어 (InJoon +50%)", "일본어 (Nanami +50%)"])

    if up_vid is not None:
        st.session_state["sh_raw_vid"] = up_vid.getvalue()

    if "sh_rows" not in st.session_state:
        st.session_state["sh_rows"] = [
            {"시작(초)": 0.0, "끝(초)": 2.2, "자막(*초록강조* / (속마음))": "처음엔 가만히 보고 있었는데"},
            {"시작(초)": 2.2, "끝(초)": 4.5, "자막(*초록강조* / (속마음))": "(설마 진짜 할까 싶더니)"},
            {"시작(초)": 4.5, "끝(초)": 7.0, "자막(*초록강조* / (속마음))": "갑자기 *엄청난 속도* 로 성공하더니"},
            {"시작(초)": 7.0, "끝(초)": 9.5, "자막(*초록강조* / (속마음))": "결국 레전드를 찍었다고 .."},
        ]

    s_desc = st.text_input("2. AI 초안 자동 생성용 상황 설명 (선택 사항 — 아래 표에서 직접 수정 가능)")
    if st.button("✨ AI로 제목·훅·자막 초안 채우기", use_container_width=True) and s_desc:
        try:
            l_rule = "일본어 작성, 끝은 '~って…'" if "일본어" in s_lang else "한국어, 연결형(~하고/~했는데), 속마음 (괄호), 강조 *별표*, 끝은 '~다고 ..'"
            sp = f'카테고리형 숏폼 대본 JSON. 주제:{s_desc}. 규칙:{l_rule}. 형식:{{"title1":"","title2":"","hook":"","subs":[{{"start":0.0,"end":2.2,"text":""}}]}}'
            raw_s = call_ai(writer_ai, sp, fast=True).strip()
            jd = json.loads(re.sub(r"^```json\s*|\s*```$", "", raw_s, flags=re.M).strip())
            st.session_state["sh_t1"] = jd.get("title1", "")
            st.session_state["sh_t2"] = jd.get("title2", "")
            st.session_state["sh_hk"] = jd.get("hook", "")
            if jd.get("subs"):
                st.session_state["sh_rows"] = [
                    {"시작(초)": float(x["start"]), "끝(초)": float(x["end"]), "자막(*초록강조* / (속마음))": x["text"]}
                    for x in jd["subs"]
                ]
            st.rerun()
        except Exception as e:
            st.warning(f"AI 서버 혼잡으로 자동 생성을 건너뜁니다. 아래 표에서 직접 입력해 주세요: {e}")

    st.divider()
    st.markdown("### 🎛️ 3. 제목 · 훅 음성 · 컷별 자막 직접 편집")
    ct1, ct2 = st.columns(2)
    t_l1 = ct1.text_input("상단 제목 1줄 (흰색)", value=st.session_state.get("sh_t1", "아니 이게 진짜 된다고?"))
    t_l2 = ct2.text_input("상단 제목 2줄 (노란 강조)", value=st.session_state.get("sh_t2", "결말까지 보면 무조건 놀람"))

    hk_c1, hk_c2 = st.columns([3, 1])
    hk_txt = hk_c1.text_input("첫 문장 훅 음성 (첫 자막 구간에만 재생)", value=st.session_state.get("sh_hk", "처음엔 평범해 보였는데"))
    hk_rate = hk_c2.selectbox("훅 음성 속도", ["+50%", "+35%", "+20%", "+0%"], index=0)

    edited_rows = st.data_editor(
        st.session_state["sh_rows"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "시작(초)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
            "끝(초)": st.column_config.NumberColumn(min_value=0.1, step=0.1, format="%.1f"),
            "자막(*초록강조* / (속마음))": st.column_config.TextColumn(width="large"),
        }
    )

    with st.expander("🎨 4. 영상 구도(배율·Y위치) & 글자 크기·볼륨 미세 조정", expanded=True):
        g1, g2, g3 = st.columns(3)
        vid_zoom = g1.slider("영상 확대 배율", 0.8, 1.8, 1.0, 0.05)
        vid_y_off = g2.slider("영상 상하(Y) 위치 이동 (px)", -200, 200, 0, 10)
        duck_vol = g3.slider("훅 구간 원본 소리 크기 (%)", 0, 100, 20, 5) / 100.0

        f1, f2, f3 = st.columns(3)
        title_sz = f1.slider("상단 제목 글자 크기", 26, 52, 38, 2)
        sub_sz = f2.slider("하단 자막 글자 크기", 20, 44, 28, 2)
        sub_y_pos = f3.slider("하단 자막 높이 (바닥 여백 px)", 80, 320, 175, 5)

    p_subs = []
    for r in edited_rows:
        txt_v = str(r.get("자막(*초록강조* / (속마음))") or "").strip()
        if txt_v:
            p_subs.append({"start": float(r["시작(초)"]), "end": float(r["끝(초)"]), "text": txt_v})

    v_code = "ja-JP-NanamiNeural" if "일본어" in s_lang else "ko-KR-InJoonNeural"
    p_json = {
        "resolution": [608, 1080], "fps": 30, "top_bar_px": 280,
        "voice": v_code, "voice_rate": hk_rate,
        "title": [t_l1, t_l2], "hook_text": hk_txt,
        "zoom": vid_zoom, "y_offset": vid_y_off, "ducking": duck_vol,
        "subtitles": p_subs
    }

    has_vid = "sh_raw_vid" in st.session_state
    btn_render = st.button(
        "🎬 현재 설정으로 608×1080 MP4 생성 / 수정 반영 재렌더링",
        type="primary", disabled=(not has_vid), use_container_width=True
    )

    if btn_render and has_vid:
        with st.spinner("수정된 자막·구도·배율을 반영해 MP4를 렌더링 중입니다..."):
            W, H, TH, tmps = 608, 1080, 280, []
            try:
                vt = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                vt.write(st.session_state["sh_raw_vid"]); vt.close(); tmps.append(vt.name)
                ht = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                ht.close(); tmps.append(ht.name)

                asyncio.run(edge_tts.Communicate(hk_txt, v_code, rate=hk_rate).save(ht.name))
                hac = AudioFileClip(ht.name); h_dur = hac.duration
                if p_subs and h_dur > p_subs[0]["end"]:
                    p_subs[0]["end"] = round(h_dur + 0.2, 2)

                sc = VideoFileClip(vt.name)
                tdur = min(sc.duration, max([s["end"] for s in p_subs], default=10.0))
                sc = sc.subclip(0, tdur)
                fp = get_font()
                ft, fs = ImageFont.truetype(fp, title_sz), ImageFont.truetype(fp, sub_sz)

                def m_frame(t):
                    im = Image.fromarray(sc.get_frame(t)).convert("RGB")
                    sw, sh = im.size; VH = H - TH
                    scl = max(W / sw, VH / sh) * vid_zoom
                    nw, nh = int(sw * scl), int(sh * scl)
                    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
                    x0 = max(0, (nw - W) // 2)
                    y0 = max(0, min(nh - VH, (nh - VH) // 2 - vid_y_off))
                    cr = resized.crop((x0, y0, min(nw, x0 + W), min(nh, y0 + VH))).resize((W, VH))

                    cv = Image.new("RGB", (W, H), (0, 0, 0))
                    cv.paste(cr, (0, TH))
                    dr = ImageDraw.Draw(cv)

                    for i, tl in enumerate(p_json["title"][:2]):
                        tw = dr.textbbox((0, 0), tl, font=ft)[2]
                        ty = 65 + i * (title_sz + 24)
                        dr.text(((W - tw) // 2, ty), tl, font=ft, fill=(255, 230, 0) if i else (255, 255, 255))

                    act = next((s["text"] for s in p_subs if s["start"] <= t < s["end"]), None)
                    if act:
                        tw = dr.textbbox((0, 0), act.replace("*", ""), font=fs)[2]
                        cx, sy = (W - tw) // 2, H - sub_y_pos
                        dr.rounded_rectangle([cx - 16, sy - 10, cx + tw + 16, sy + sub_sz + 14], radius=10, fill=(0, 0, 0, 190))
                        for seg in re.split(r"(\*[^*]+\*)", act):
                            if not seg: continue
                            gr = seg.startswith("*") and seg.endswith("*")
                            tx = seg[1:-1] if gr else seg
                            dr.text((cx, sy), tx, font=fs, fill=(80, 255, 120) if gr else (255, 255, 255))
                            cx += dr.textbbox((0, 0), tx, font=fs)[2]
                    return np.array(cv)

                oc = VideoClip(m_frame, duration=tdur)
                if sc.audio:
                    p1 = min(h_dur, tdur)
                    al = [sc.audio.subclip(0, p1).volumex(duck_vol), hac.set_start(0)]
                    if tdur > p1:
                        al.append(sc.audio.subclip(p1, tdur).volumex(1.0).set_start(p1))
                    fa = CompositeAudioClip(al)
                else:
                    fa = CompositeAudioClip([hac.set_start(0)])

                oc = oc.set_audio(fa.set_duration(tdur))
                ot = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); ot.close()
                oc.write_videofile(ot.name, fps=30, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
                oc.close(); sc.close(); hac.close()

                st.session_state.shorts_mp4 = open(ot.name, "rb").read()
                st.session_state.shorts_hook_aud = open(ht.name, "rb").read()
                os.remove(ot.name)
            finally:
                for f in tmps:
                    if os.path.exists(f): os.remove(f)

    if st.session_state.shorts_mp4:
        st.divider()
        st.markdown("### 📺 완성 영상 미리보기 (마음에 안 들면 위 표·슬라이더 수정 후 재렌더링 클릭)")
        pv1, pv2 = st.columns([1, 1])
        with pv1:
            st.video(st.session_state.shorts_mp4)
        with pv2:
            st.info("💡 **즉시 수정 가이드**\n- 피사체가 잘렸다면 **4번 구도 패널**에서 `영상 확대 배율`이나 `상하(Y) 위치`를 조절하세요.\n- 자막 타이밍이나 강조 색(`*단어*`)을 바꾸려면 **3번 자막 표**를 고친 뒤 위의 **[재렌더링]** 버튼을 다시 누르면 원본 재업로드 없이 바로 새 영상이 만들어집니다.")
            st.download_button(
                "📥 완성된 숏폼 MP4 다운로드",
                st.session_state.shorts_mp4, "category_shorts.mp4",
                "video/mp4", type="primary", use_container_width=True
            )
            # 2차 외부 편집기(캡컷/프리미어)용 SRT + 훅음성 + JSON ZIP 패키지
            srt_lines = []
            for idx, s in enumerate(p_subs, 1):
                def fmt_t(sec):
                    h, r = divmod(sec, 3600); m, s_v = divmod(r, 60)
                    return f"{int(h):02d}:{int(m):02d}:{int(s_v):02d},{int((s_v%1)*1000):03d}"
                srt_lines.append(f"{idx}\n{fmt_t(s['start'])} --> {fmt_t(s['end'])}\n{s['text'].replace('*','')}\n")
            sz_buf = io.BytesIO()
            with zipfile.ZipFile(sz_buf, "w") as zf:
                zf.writestr("category_shorts.mp4", st.session_state.shorts_mp4)
                zf.writestr("subtitles.srt", "\n".join(srt_lines))
                zf.writestr("project.json", json.dumps(p_json, ensure_ascii=False, indent=2))
                if "shorts_hook_aud" in st.session_state:
                    zf.writestr("hook_voice.mp3", st.session_state["shorts_hook_aud"])
            st.download_button(
                "📦 캡컷/프리미어 2차 편집용 소스 ZIP (MP4+SRT+훅음성+JSON)",
                sz_buf.getvalue(), "shorts_edit_package.zip",
                "application/zip", use_container_width=True
            )

