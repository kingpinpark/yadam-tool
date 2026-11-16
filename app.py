import re
import time
import json
import base64
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="멀티 AI 야담·민담 스튜디오", page_icon="📜", layout="wide")
st.title("📜 멀티 AI 협력형 야담·민담 제작 스튜디오")
st.caption("분석가(Gemini) · 작가(Claude / GPT) · 화가·성우(OpenAI)가 각자 잘하는 일을 분담해 영상을 기획합니다.")

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

# ==========================================
# 사이드바: 여러 AI API 키 연결 및 역할 분담
# ==========================================
with st.sidebar:
    st.header("🔑 1. AI 엔진 연결 (필요한 것만 입력)")
    gemini_key = st.text_input("Gemini API 키 (Google)", type="password", placeholder="AIzaSy...")
    openai_key = st.text_input("OpenAI API 키 (GPT / 삽화 / 성우)", type="password", placeholder="sk-...")
    claude_key = st.text_input("Claude API 키 (Anthropic)", type="password", placeholder="sk-ant-...")
    
    # 연결된 AI 목록 구성
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
    st.header("🤝 2. AI 역할 분담 설정")
    analyst_ai = st.selectbox("🔍 1단계 [영상 분석가] 담당 AI", options=connected_ais, index=0)
    writer_ai = st.selectbox(
        "✍️ 2단계 [야담 작가] 담당 AI",
        options=connected_ais,
        index=min(1, len(connected_ais) - 1)
    )
    st.caption("💡 팁: 분석은 빠르고 긴 글에 강한 Gemini에게, 대본 집필은 문장력이 좋은 Claude나 GPT에게 맡겨보세요!")

    st.divider()
    st.subheader("🎨 고정 화풍 설정 (Style Anchor)")
    style_anchor = st.text_area(
        "모든 장면 그림 프롬프트 앞에 자동 결합됩니다.",
        value="Traditional Korean folk painting style, Joseon Dynasty historical illustration, moody cinematic lighting, ink wash and soft watercolor texture --",
        height=90
    )

# ==========================================
# 통합 AI 호출 엔진 (Gemini / OpenAI / Claude 공용)
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
            # 마크다운 코드블록(```json ... ```) 제거 처리
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
        raise RuntimeError(f"자막을 가져오지 못했습니다: {e}")

# 세션 상태 초기화
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "generated_scenes" not in st.session_state:
    st.session_state.generated_scenes = None
if "scene_media" not in st.session_state:
    st.session_state.scene_media = {}

# ==========================================
# [1단계] 레퍼런스 유튜브 영상 분석
# ==========================================
st.header(f"1단계: 벤치마킹 영상 분석 (담당: {analyst_ai})")
col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input("유튜브 영상 링크를 붙여넣으세요", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🔍 영상 구조 분석하기", use_container_width=True)

if analyze_btn:
    if not youtube_url:
        st.warning("유튜브 링크를 입력해 주세요.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("올바른 유튜브 링크가 아닙니다.")
        else:
            with st.spinner(f"[{analyst_ai}]가 영상을 분석하고 공식을 추출하는 중입니다..."):
                try:
                    transcript_text = fetch_youtube_transcript(video_id)
                    prompt = f"""
                    다음은 조회수가 잘 나온 유튜브 야담/민담 영상의 전체 자막입니다.
                    다른 작가 AI가 이 스타일을 그대로 재현할 수 있도록 핵심 비결을 분석해 주세요.
                    
                    1. 초반 15초 후킹(Hook) 기법: 첫 문장을 어떻게 시작해서 호기심을 자극했는가?
                    2. 화법과 문체 특징: 어미 처리(~했더랍니다, ~였지요 등), 호흡의 길이, 분위기 조성 방식
                    3. 기승전결 전개 비율: 발단-위기-절정-결말이 어떤 템포로 배치되었는가?
                    4. 벤치마킹 핵심 공식: 새로운 야담을 쓸 때 그대로 적용할 스토리텔링 공식 3줄 요약

                    [원본 자막]
                    {transcript_text[:15000]}
                    """
                    st.session_state.analysis_result = call_selected_ai(analyst_ai, prompt, is_json=False)
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
    scene_count = st.number_input("분할할 장면 수", min_value=4, max_value=20, value=8)

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
                1. narration(대사): 성우가 읽기 좋게 구수하고 긴장감 있는 사극 구어체('~했더랍니다', '~였지요', '~한 것이었습니다')를 사용하세요.
                2. image_prompt(그림 묘사): 각 장면에 어울리는 조선시대 배경, 인물 표정, 복식, 조명을 영어로 구체적으로 묘사하세요. (단, 화풍 스타일은 시스템에서 자동 추가하므로 장면 내용만 영어로 작성)
                """

                raw_json = call_selected_ai(writer_ai, gen_prompt, is_json=True)
                st.session_state.generated_scenes = json.loads(raw_json)
                st.session_state.scene_media = {}
            except Exception as e:
                st.error(f"대본 생성 오류: {e}")

# ==========================================
# [3단계] 반자동 검수 + 삽화/성우 AI 즉시 제작
# ==========================================
if st.session_state.generated_scenes:
    data = st.session_state.generated_scenes
    st.subheader(f"🎬 추천 제목: {data['title']}")
    st.info(f"💡 **초반 후킹 포인트:** {data['hook_summary']}")
    
    if not openai_key:
        st.caption("💡 왼쪽 사이드바에 **OpenAI API 키**를 추가로 입력하시면, 각 장면마다 **AI 삽화 그리기 & 성우 음성 녹음 버튼**이 활성화됩니다!")

    edited_scenes = []
    for scene in data["scenes"]:
        s_num = scene["scene_number"]
        with st.container(border=True):
            st.markdown(f"### 장면 {s_num}")
            col_text, col_img = st.columns(2)
            
            with col_text:
                new_narration = st.text_area(
                    "내레이션 대본 (수정 가능)",
                    value=scene["narration"],
                    key=f"nar_{s_num}",
                    height=110
                )
            with col_img:
                full_prompt = f"{style_anchor} {scene['image_prompt']}"
                new_img_prompt = st.text_area(
                    "최종 삽화 프롬프트 (화풍 자동 결합됨)",
                    value=full_prompt,
                    key=f"img_{s_num}",
                    height=110
                )

            # OpenAI 키가 연결되어 있으면 삽화 + 성우 음성 바로 생성 버튼 제공
            if openai_key:
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(f"🎨 장면 {s_num} 삽화 그리기", key=f"btn_img_{s_num}"):
                        with st.spinner("조선시대 삽화를 그리는 중..."):
                            try:
                                oai = OpenAI(api_key=openai_key)
                                img_res = oai.images.generate(
                                    model="dall-e-3",
                                    prompt=new_img_prompt,
                                    size="1792x1024",
                                    quality="standard",
                                    n=1
                                )
                                st.session_state.scene_media.setdefault(s_num, {})["image_url"] = img_res.data[0].url
                            except Exception as e:
                                st.error(f"삽화 생성 실패: {e}")
                with b_col2:
                    if st.button(f"🎙️ 장면 {s_num} 성우 녹음하기", key=f"btn_tts_{s_num}"):
                        with st.spinner("성우 목소리를 녹음하는 중..."):
                            try:
                                oai = OpenAI(api_key=openai_key)
                                tts_res = oai.audio.speech.create(
                                    model="tts-1",
                                    voice="onyx",
                                    input=new_narration
                                )
                                st.session_state.scene_media.setdefault(s_num, {})["audio_bytes"] = tts_res.content
                            except Exception as e:
                                st.error(f"음성 생성 실패: {e}")

                # 생성된 미디어 미리보기 출력
                media = st.session_state.scene_media.get(s_num, {})
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if "image_url" in media:
                        st.image(media["image_url"], caption=f"장면 {s_num} 생성 삽화", use_container_width=True)
                with m_col2:
                    if "audio_bytes" in media:
                        st.audio(media["audio_bytes"], format="audio/mp3")
                        st.download_button(
                            label=f"⬇️ 장면 {s_num} 음성(.mp3) 다운로드",
                            data=media["audio_bytes"],
                            file_name=f"scene_{s_num}.mp3",
                            mime="audio/mp3",
                            key=f"dl_audio_{s_num}"
                        )

            edited_scenes.append({
                "scene_number": s_num,
                "narration": new_narration,
                "final_image_prompt": new_img_prompt
            })

    final_output = {
        "title": data["title"],
        "scenes": edited_scenes
    }
    st.download_button(
        label="📥 전체 대본 & 프롬프트 기획안 저장 (JSON)",
        data=json.dumps(final_output, ensure_ascii=False, indent=2),
        file_name="yadam_script.json",
        mime="application/json"
    )
