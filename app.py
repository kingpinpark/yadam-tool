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
            raw = " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"])) if hasattr(YouTubeTranscriptApi, "get_transcript") else " ".join(x.text for x in YouTubeTranscriptApi().fetch(vid, languages=["ko", "en"]))
            st.session_state.analysis = call_ai(analyst_ai, f"야담의 1)후킹 2)화법 3)전개 4)공식 3줄 요약:\n{raw[:25000]}", fast=True)
            st.success("분석 완료!")
        except Exception as e: st.error(f"자막 추출 실패(아래 직접 붙여넣기 이용): {e}")

    with st.expander("📝 대본/자막 직접 붙여넣기"):
        m_txt = st.text_area("스크립트 붙여넣기", height=100)
        if st.button("붙여넣은 대본 분석") and m_txt.strip():
            c = m_txt[:15000] + "\n" + m_txt[-10000:] if (fast_opt and len(m_txt) > 28000) else m_txt[:100000]
            st.session_state.analysis = call_ai(analyst_ai, f"야담의 1)후킹 2)화법 3)전개 4)공식 3줄 요약:\n{c}", fast=True)

    if st.session_state.analysis:
        with st.expander("📊 분석 공식 보기", expanded=True): st.markdown(st.session_state.analysis)

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
        if s_new: st.session_state.ch, st.session_state.hist = 1, []
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

with tab_shorts:
    st.subheader("📱 카테고리형 숏폼 제작기 (608×1080 / 상단 280px 제목 / 훅 +50%)")
    sc1, sc2 = st.columns([2, 1])
    up_vid = sc1.file_uploader("1. 원본 영상 업로드 (.mp4, .mov)", type=["mp4", "mov", "mkv"])
    s_lang = sc2.selectbox("언어 & 성우 (+50%)", ["한국어 (InJoon +50%)", "일본어 (Nanami +50%)"])
    s_desc = st.text_input("2. 영상 상황 한 줄 설명")

    if st.button("✨ AI로 제목 2줄 + 훅 + 컷별 자막 자동 작성", use_container_width=True) and s_desc:
        l_rule = "일본어 작성, 끝은 '~って…'" if "일본어" in s_lang else "한국어, 연결형(~하고/~했는데), 속마음 (괄호), 강조 *별표*, 끝은 '~다고 ..'"
        sp = f'카테고리형 숏폼 대본 JSON 작성. 주제:{s_desc}. 규칙:{l_rule}. 형식:{{"title1":"","title2":"","hook":"","subs":"0.0|2.0|첫 자막 *강조*\\n2.0|4.0|(속마음)\\n4.0|6.0|마지막 ~다고 .."}}'
        jd = json.loads(re.sub(r"^```json\s*|\s*```$", "", call_ai(writer_ai, sp, fast=True).strip(), flags=re.M).strip())
        st.session_state.update({"sh_t1": jd.get("title1",""), "sh_t2": jd.get("title2",""), "sh_hk": jd.get("hook",""), "sh_sb": jd.get("subs","")})

    ct1, ct2 = st.columns(2)
    t_l1 = ct1.text_input("상단 제목 1줄", value=st.session_state.get("sh_t1", "아니 이게 진짜 된다고?"))
    t_l2 = ct2.text_input("상단 제목 2줄", value=st.session_state.get("sh_t2", "결말까지 보면 무조건 놀람"))
    hk_txt = st.text_input("첫 문장 훅 음성 (+50% 배속)", value=st.session_state.get("sh_hk", "처음엔 평범해 보였는데"))
    def_sb = "0.0|2.2|처음엔 가만히 보고 있었는데\n2.2|4.5|(설마 진짜 할까 싶더니)\n4.5|7.0|갑자기 *엄청난 속도* 로 성공하더니\n7.0|9.5|결국 레전드를 찍었다고 .."
    sb_raw = st.text_area("컷별 자막 (시작초|끝초|자막)", value=st.session_state.get("sh_sb", def_sb), height=110)

    p_subs = []
    for ln in sb_raw.strip().splitlines():
        ps = [p.strip() for p in ln.split("|")]
        if len(ps) >= 3:
            try: p_subs.append({"start": float(ps[0]), "end": float(ps[1]), "text": ps[2]})
            except ValueError: pass

    v_code = "ja-JP-NanamiNeural" if "일본어" in s_lang else "ko-KR-InJoonNeural"
    p_json = {"resolution": [608, 1080], "fps": 30, "top_bar_px": 280, "voice": v_code, "title": [t_l1, t_l2], "hook_text": hk_txt, "subtitles": p_subs}
    bc1, bc2 = st.columns(2)
    fn_j = "project_ja.json" if "일본어" in s_lang else "project_ko.json"
    bc1.download_button(f"📄 {fn_j} 다운로드", json.dumps(p_json, ensure_ascii=False, indent=2), fn_j, "application/json", use_container_width=True)

    if bc2.button("🎬 608×1080 MP4 렌더링", type="primary", disabled=(up_vid is None), use_container_width=True):
        with st.spinner("렌더링 중..."):
            W, H, TH, tmps = 608, 1080, 280, []
            try:
                vt = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); vt.write(up_vid.read()); vt.close(); tmps.append(vt.name)
                ht = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3"); ht.close(); tmps.append(ht.name)
                asyncio.run(edge_tts.Communicate(p_json["hook_text"], v_code, rate="+50%").save(ht.name))
                hac = AudioFileClip(ht.name); h_dur = hac.duration
                if p_subs and h_dur > p_subs[0]["end"]: p_subs[0]["end"] = round(h_dur + 0.2, 2)
                sc = VideoFileClip(vt.name)
                tdur = min(sc.duration, max([s["end"] for s in p_subs], default=10.0))
                sc = sc.subclip(0, tdur)
                fp = get_font(); ft, fs = ImageFont.truetype(fp, 38), ImageFont.truetype(fp, 28)

                def m_frame(t):
                    im = Image.fromarray(sc.get_frame(t)).convert("RGB")
                    sw, sh = im.size; VH = H - TH
                    scl = max(W / sw, VH / sh); nw, nh = int(sw * scl), int(sh * scl)
                    cr = im.resize((nw, nh)).crop(((nw - W)//2, (nh - VH)//2, (nw + W)//2, (nh + VH)//2))
                    cv = Image.new("RGB", (W, H), (0, 0, 0)); cv.paste(cr, (0, TH))
                    dr = ImageDraw.Draw(cv)
                    for i, tl in enumerate(p_json["title"][:2]):
                        tw = dr.textbbox((0, 0), tl, font=ft)[2]
                        dr.text(((W - tw)//2, 65 + i * 65), tl, font=ft, fill=(255, 230, 0) if i else (255, 255, 255))
                    act = next((s["text"] for s in p_subs if s["start"] <= t < s["end"]), None)
                    if act:
                        tw = dr.textbbox((0, 0), act.replace("*", ""), font=fs)[2]
                        cx, sy = (W - tw)//2, H - 175
                        dr.rounded_rectangle([cx - 16, sy - 10, cx + tw + 16, sy + 42], radius=10, fill=(0, 0, 0, 190))
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
                    al = [sc.audio.subclip(0, p1).volumex(0.2), hac.set_start(0)]
                    if tdur > p1: al.append(sc.audio.subclip(p1, tdur).volumex(1.0).set_start(p1))
                    fa = CompositeAudioClip(al)
                else: fa = CompositeAudioClip([hac.set_start(0)])
                oc = oc.set_audio(fa.set_duration(tdur))
                ot = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); ot.close()
                oc.write_videofile(ot.name, fps=30, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
                oc.close(); sc.close(); hac.close()
                st.session_state.shorts_mp4 = open(ot.name, "rb").read(); os.remove(ot.name)
            finally:
                for f in tmps:
                    if os.path.exists(f): os.remove(f)

    if st.session_state.shorts_mp4:
        st.video(st.session_state.shorts_mp4)
        st.download_button("📥 완성된 숏폼 MP4 다운로드", st.session_state.shorts_mp4, "category_shorts.mp4", "video/mp4", use_container_width=True)
