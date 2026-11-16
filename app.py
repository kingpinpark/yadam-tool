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
# 사이드바: 여러 AI API 키 연결 및 설정
# ==========================================
with st.sidebar:
    st.header("🔑 1. AI 엔진 연결")
    gemini_key = st.text_input("Gemini API 키 (Google)", type="password", placeholder="AIzaSy...")
    openai_key = st.text_input("OpenAI API 키 (GPT / 삽화 / 성우)", type="password", placeholder="sk-...")
    claude_key = st.text_input("Claude API 키 (Anthropic)", type="password", placeholder="sk-ant-...")
    
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
        connected_ais = ["사이드바에 API 키를 1개 이상 입력하세요"]

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
    line_widths = []
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    padding_x = 32
    padding_y = 18
    line_spacing = 10
    total_text_h = sum(line_heights) + line_spacing * (len(wrapped_lines) - 1)
    max_text_w = max(line_widths)

    box_w = min(width - 80, max_text_w + padding_x * 2)
    box_h = total_text_h + padding_y * 2
    box_x0 = (width - box_w) // 2
    box_y0 = height - box_h - int(height * 0.06)
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=14, fill=(10, 10, 15, 185))

    curr_y = box_y0 + padding_y
    for idx, line in enumerate(wrapped_lines):
        tx = (width - line_widths[idx]) // 2
        draw.text((tx, curr_y), line, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        curr_y += line_heights[idx] + line_spacing

    return overlay

def build_final_mp4(edited_scenes: list, scene_media: dict, font_path: str) -> str:
    W, H = 1280, 720
    clips = []
    temp_files = []

    try:
        for sc in edited_scenes:
            s_num = sc["scene_number"]
            media = scene_media.get(s_num, {})
            if "image_bytes" not in media or "audio_bytes" not in media:
                raise ValueError(f"장면 {s_num}의 삽화 또는 성우 음성이 아직 생성되지 않았습니다.")

            img_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_tmp.write(media["image_bytes"])
            img_tmp.close()
            temp_files.append(img_tmp.name)

            aud_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            aud_tmp.write(media["audio_bytes"])
            aud_tmp.close()
            temp_files.append(aud_tmp.name)

            audio_clip = AudioFileClip(aud_tmp.name)
            duration = audio_clip.duration + 0.4

            base_img = Image.open(img_tmp.name).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            sub_overlay = create_subtitle_overlay(W, H, sc["narration"], font_path)

            def make_scene_frame(t, img=base_img, sub=sub_overlay, dur=duration):
                zoom = 1.0 + 0.12 * (t / max(dur, 0.1))
                new_w = int(W * zoom)
                new_h = int(H * zoom)
                zoomed = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                left = (new_w - W) // 2
                top = (new_h - H) // 2
                cropped = zoomed.crop((left, top, left + W, top + H)).convert("RGBA")
                combined = Image.alpha_composite(cropped, sub).convert("RGB")
                return np.array(combined)

            v_clip = VideoClip(make_scene_frame, duration=duration).set_audio(audio_clip)
            clips.append(v_clip)

        final_video = concatenate_videoclips(clips, method="compose")
        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_tmp.close()

        final_video.write_videofile(
            out_tmp.name,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            logger=None
        )
        final_video.close()
        for c in clips:
            c.close()

        return out_tmp.name
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass

# 세션 상태 초기화
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "generated_scenes" not in st.session_state:
    st.session_state.generated_scenes = None
if "scene_media" not in st.session_state:
    st.session_state.scene_media = {}
if "final_video_bytes" not in st.session_state:
    st.session_state.final_video_bytes = None

def run_structure_analysis(script_text: str):
    prompt = f"""
    다음은 조회수가 잘 나온 유튜브 야담/민담 영상의 전체 대본(자막)입니다.
    다른 작가 AI가 이 스타일을 그대로 재현할 수 있도록 핵심 비결을 분석해 주세요.
    
    1. 초반 15초 후킹(Hook) 기법: 첫 문장을 어떻게 시작해서 호기심을 자극했는가?
    2. 화법과 문체 특징: 어미 처리(~했더랍니다, ~였지요 등), 호흡의 길이, 분위기 조성 방식
    3. 기승전결 전개 비율: 발단-위기-절정-결말이 어떤 템포로 배치되었는가?
    4. 벤치마킹 핵심 공식: 새로운 야담을 쓸 때 그대로 적용할 스토리텔링 공식 3줄 요약

    [원본 자막/대본]
    {script_text[:15000]}
    """
    return call_selected_ai(analyst_ai, prompt, is_json=False)

# ==========================================
# [1단계] 레퍼런스 유튜브 영상 분석 (링크 추출 + 직접 붙여넣기)
# ==========================================
st.header(f"1단계: 벤치마킹 영상 분석 (담당: {analyst_ai})")

tab_url, tab_paste = st.tabs(["🔗 유튜브 링크로 자동 분석", "📝 대본/자막 직접 붙여넣기 (유튜브 IP 차단 시)"])

with tab_url:
    col1, col2 = st.columns([4, 1])
    with col1:
        youtube_url = st.text_input("유튜브 영상 링크를 붙여넣으세요", placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        st.write("")
        st.write("")
        analyze_url_btn = st.button("🔍 링크로 분석하기", use_container_width=True)

    if analyze_url_btn:
        if not youtube_url:
            st.warning("유튜브 링크를 입력해 주세요.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("올바른 유튜브 링크가 아닙니다.")
            else:
                with st.spinner(f"[{analyst_ai}]가 자막을 추출하고 분석하는 중입니다..."):
                    try:
                        transcript_text = fetch_youtube_transcript(video_id)
                        st.session_state.analysis_result = run_structure_analysis(transcript_text)
                    except Exception as e:
                        st.error(f"{e}")

with tab_paste:
    st.caption("💡 유튜브 영상 설명란의 **[스크립트 표시]**를 눌러 전체 자막을 복사한 뒤 아래에 붙여넣으면 100% 분석됩니다.")
    manual_transcript = st.text_area(
        "벤치마킹할 야담 영상의 자막(스크립트)이나 참고할 옛이야기 본문을 붙여넣으세요",
        height=160,
        placeholder="여기에 복사한 유튜브 스크립트나 야담 텍스트를 붙여넣으세요..."
    )
    if st.button("🔍 붙여넣은 대본 구조 분석하기", type="secondary"):
        if not manual_transcript.strip():
            st.warning("분석할 텍스트를 먼저 붙여넣어 주세요.")
        else:
            with st.spinner(f"[{analyst_ai}]가 붙여넣은 대본의 구조를 분석하는 중입니다..."):
                try:
                    st.session_state.analysis_result = run_structure_analysis(manual_transcript)
                except Exception as e:
                    st.error(f"분석 오류: {e}")

if st.session_state.analysis_result:
    with st.expander("📊 분석가 AI가 뽑아낸 스토리텔링 공식 (클릭하여 접기/펴기)", expanded=True):
        st.markdown(st.session_state.analysis_result)

st.divider()

# ==========================================
# [2단계] 새로운 야담 대본 & 장면 자동 분할
# ==========================================
st.header(f"2단계: 새로운 야담 대본 집필 (담당: {writer_ai})")

c1, c2 = st.columns([3, 1])
with c1:
    new_topic = st.text_input(
        "만들고 싶은 새로운 야담 소재나 키워드를 입력하세요",
        placeholder="예: 한양으로 과거 시험 보러 가던 가난한 선비가 산속 폐가에서 만난 여인의 비밀"
    )
with c2:
    scene_count = st.number_input("분할할 장면 수", min_value=3, max_value=15, value=6)

generate_btn = st.button("✍️ 작가 AI에게 대본 의뢰하기", type="primary")

if generate_btn:
    if not new_topic:
        st.warning("새로운 야담 주제를 입력해 주세요.")
    else:
        with st.spinner(f"[{writer_ai}]가 분석 결과를 바탕으로 야담 대본을 집필 중입니다..."):
            try:
                reference_guide = (
                    f"아래 [분석가 AI]가 추출한 인기 영상의 성공 공식을 반드시 대본에 적용하세요:\n{st.session_state.analysis_result}"
                    if st.session_state.analysis_result
                    else "초반 15초 안에 강렬한 미스터리나 기이한 사건을 던져 시청자의 시선을 사로잡으세요."
                )

                gen_prompt = f"""
                당신은 조선시대 저잣거리에서 사람들을 울고 웃기는 최고의 이야기꾼(전기수)입니다.
                아래 주제와 분석 가이드를 바탕으로 몰입감 넘치는 야담 대본을 작성하고, 총 {scene_count}개의 장면으로 나누어 주세요.

                [주제]
                {new_topic}

                [적용할 스타일 가이드]
                {reference_guide}

                [작성 규칙]
                1. narration(대사): 성우가 읽기 좋고 자막으로 띄우기 알맞게 구수하고 긴장감 있는 사극 구어체('~했더랍니다', '~였지요', '~한 것이었습니다')로 2~3문장 내외로 작성하세요.
                2. image_prompt(그림 묘사): 각 장면에 어울리는 조선시대 배경, 인물 표정, 복식, 조명을 영어로 구체적으로 묘사하세요. (화풍 스타일은 자동 추가되므로 장면 내용만 영어로 작성)
                """

                raw_json = call_selected_ai(writer_ai, gen_prompt, is_json=True)
                st.session_state.generated_scenes = json.loads(raw_json)
                st.session_state.scene_media = {}
                st.session_state.final_video_bytes = None
            except Exception as e:
                st.error(f"대본 생성 오류: {e}")

# ==========================================
# [3단계] 반자동 검수 + 삽화/성우 AI 제작
# ==========================================
if st.session_state.generated_scenes:
    data = st.session_state.generated_scenes
    st.divider()
    st.header("3단계: 대본 다듬기 & 삽화·성우 음성 생성")
    st.subheader(f"🎬 추천 제목: {data['title']}")
    st.info(f"💡 **초반 후킹 포인트:** {data['hook_summary']}")

    if not openai_key:
        st.warning("⚠️ 왼쪽 사이드바에 **OpenAI API 키**를 입력하시면 삽화 그리기, 성우 녹음, 최종 MP4 영상 제작 기능이 모두 활성화됩니다.")

    edited_scenes = []
    for scene in data["scenes"]:
        s_num = scene["scene_number"]
        with st.container(border=True):
            st.markdown(f"### 장면 {s_num}")
            col_text, col_img = st.columns(2)
            
            with col_text:
                new_narration = st.text_area(
                    "내레이션 대본 (수정 가능 - 영상 자막과 성우 음성에 반영됨)",
                    value=scene["narration"],
                    key=f"nar_{s_num}",
                    height=105
                )
            with col_img:
                full_prompt = f"{style_anchor} {scene['image_prompt']}"
                new_img_prompt = st.text_area(
                    "최종 삽화 프롬프트",
                    value=full_prompt,
                    key=f"img_{s_num}",
                    height=105
                )

            edited_scenes.append({
                "scene_number": s_num,
                "narration": new_narration,
                "final_image_prompt": new_img_prompt
            })

            if openai_key:
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(f"🎨 장면 {s_num} 삽화만 다시 그리기", key=f"btn_img_{s_num}"):
                        with st.spinner(f"장면 {s_num} 삽화를 그리는 중..."):
                            try:
                                oai = OpenAI(api_key=openai_key)
                                img_res = oai.images.generate(
                                    model="dall-e-3",
                                    prompt=new_img_prompt,
                                    size="1792x1024",
                                    quality="standard",
                                    n=1
                                )
                                img_bytes = requests.get(img_res.data[0].url, timeout=30).content
                                st.session_state.scene_media.setdefault(s_num, {})["image_bytes"] = img_bytes
                            except Exception as e:
                                st.error(f"삽화 생성 실패: {e}")
                with b_col2:
                    if st.button(f"🎙️ 장면 {s_num} 음성만 다시 녹음", key=f"btn_tts_{s_num}"):
                        with st.spinner(f"장면 {s_num} 성우 녹음 중..."):
                            try:
                                oai = OpenAI(api_key=openai_key)
                                tts_res = oai.audio.speech.create(
                                    model="tts-1",
                                    voice=tts_voice,
                                    input=new_narration
                                )
                                st.session_state.scene_media.setdefault(s_num, {})["audio_bytes"] = tts_res.content
                            except Exception as e:
                                st.error(f"음성 생성 실패: {e}")

                media = st.session_state.scene_media.get(s_num, {})
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if "image_bytes" in media:
                        st.image(media["image_bytes"], caption=f"장면 {s_num} 삽화", use_container_width=True)
                with m_col2:
                    if "audio_bytes" in media:
                        st.audio(media["audio_bytes"], format="audio/mp3")

    if openai_key:
        st.write("")
        if st.button("⚡ 모든 장면 삽화 + 성우 음성 한 번에 일괄 생성하기", type="secondary", use_container_width=True):
            progress_bar = st.progress(0, text="전체 장면 삽화와 성우 음성을 순서대로 생성하고 있습니다...")
            oai = OpenAI(api_key=openai_key)
            total = len(edited_scenes)
            try:
                for idx, sc in enumerate(edited_scenes):
                    s_num = sc["scene_number"]
                    progress_bar.progress(idx / total, text=f"장면 {s_num}/{total} 삽화 그리는 중...")
                    if "image_bytes" not in st.session_state.scene_media.get(s_num, {}):
                        img_res = oai.images.generate(
                            model="dall-e-3",
                            prompt=sc["final_image_prompt"],
                            size="1792x1024",
                            quality="standard",
                            n=1
                        )
                        img_bytes = requests.get(img_res.data[0].url, timeout=30).content
                        st.session_state.scene_media.setdefault(s_num, {})["image_bytes"] = img_bytes

                    progress_bar.progress((idx + 0.5) / total, text=f"장면 {s_num}/{total} 성우 음성 녹음 중...")
                    if "audio_bytes" not in st.session_state.scene_media.get(s_num, {}):
                        tts_res = oai.audio.speech.create(
                            model="tts-1",
                            voice=tts_voice,
                            input=sc["narration"]
                        )
                        st.session_state.scene_media.setdefault(s_num, {})["audio_bytes"] = tts_res.content

                progress_bar.progress(1.0, text="✅ 모든 장면의 삽화와 성우 음성 생성이 완료되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"일괄 생성 중 오류 발생: {e}")

    # ==========================================
    # [4단계] 최종 MP4 동영상 자동 조립 및 다운로드
    # ==========================================
    st.divider()
    st.header("4단계: 줌인 효과 & 한글 자막 MP4 영상 자동 완성")
    
    ready_count = sum(
        1 for sc in edited_scenes
        if "image_bytes" in st.session_state.scene_media.get(sc["scene_number"], {})
        and "audio_bytes" in st.session_state.scene_media.get(sc["scene_number"], {})
    )
    st.write(f"현재 준비된 장면: **{ready_count} / {len(edited_scenes)}개** (모든 장면의 그림과 음성이 준비되면 영상을 합칠 수 있습니다)")

    if st.button("🎬 최종 MP4 동영상 합치기 (줌인 효과 + 자막)", type="primary", disabled=(ready_count < len(edited_scenes))):
        with st.spinner("한글 폰트를 불러오고 모든 장면에 줌인 효과와 자막을 입혀 MP4 영상을 렌더링하는 중입니다... (약 20~40초 소요)"):
            try:
                font_path = get_korean_font_path()
                mp4_path = build_final_mp4(edited_scenes, st.session_state.scene_media, font_path)
                with open(mp4_path, "rb") as f:
                    st.session_state.final_video_bytes = f.read()
                os.remove(mp4_path)
                st.success("🎉 야담 영상 렌더링이 완료되었습니다! 아래에서 미리보고 다운로드하세요.")
            except Exception as e:
                st.error(f"영상 렌더링 중 오류가 발생했습니다: {e}")

    if st.session_state.final_video_bytes:
        st.video(st.session_state.final_video_bytes)
        st.download_button(
            label="📥 완성된 야담 MP4 동영상 다운로드",
            data=st.session_state.final_video_bytes,
            file_name="yadam_final_video.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )
