import re
import time
import json
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="야담·민담 반자동 제작기", page_icon="📜", layout="wide")
st.title("📜 AI 야담·민담 벤치마킹 & 반자동 대본 생성기")

# 내 API 키로 사용 가능한 Gemini 모델 목록 자동 조회
@st.cache_data(ttl=600)
def get_available_models(key: str):
    try:
        client = genai.Client(api_key=key)
        models = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            # 텍스트/대본 생성이 가능한 gemini 모델만 필터링
            if "gemini" in name.lower() and not any(x in name.lower() for x in ["embedding", "imagen", "veo", "aqa", "tts", "audio"]):
                models.append(name)
        # gemini-3.8-flash가 있으면 맨 위로 정렬
        models.sort(key=lambda x: (0 if "3.8-flash" in x else 1, x))
        return models if models else ["gemini-3.8-flash"]
    except Exception:
        return ["gemini-3.8-flash"]

# 사이드바: API 키 및 모델 선택
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API 키 입력", type="password", placeholder="AIzaSy...")
    st.caption("Google AI Studio에서 발급받은 API 키를 입력하세요.")
    
    st.divider()
    st.subheader("🤖 AI 모델 선택")
    if api_key:
        available_models = get_available_models(api_key)
    else:
        available_models = ["gemini-3.8-flash"]
        
    selected_model = st.selectbox(
        "내 키에서 사용 가능한 모델 목록 (자동 조회됨)",
        options=available_models,
        index=0
    )
    st.caption("💡 503 혼잡 에러가 뜨면 목록의 다른 모델로 바꿔서 눌러보세요.")
    
    st.divider()
    st.subheader("🎨 고정 화풍 설정 (Style Anchor)")
    style_anchor = st.text_area(
        "모든 장면 그림 프롬프트 앞에 자동 결합됩니다.",
        value="Traditional Korean folk painting style, Joseon Dynasty historical illustration, moody cinematic lighting, ink wash and soft watercolor texture, 16:9 aspect ratio --",
        height=100
    )

# 서버 일시 혼잡(503) 시 자동 재시도 함수
def generate_with_retry(client, model_name, contents, config=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model_name, contents=contents, config=config)
            return client.models.generate_content(model=model_name, contents=contents)
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(3)
                continue
            raise e

# 2. 유튜브 영상 ID 추출 및 자막 가져오기 함수
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
        raise RuntimeError(f"자막을 가져오지 못했습니다. (자막이 꺼져 있거나 비공개 영상일 수 있습니다): {e}")

# 3. Gemini 구조화 출력을 위한 데이터 모델 정의
class Scene(BaseModel):
    scene_number: int = Field(description="장면 번호 (1부터 순차 증가)")
    narration: str = Field(description="전기수(옛날 이야기꾼) 말투의 한국어 내레이션 대사")
    image_prompt: str = Field(description="해당 장면을 묘사하는 구체적인 영문 이미지 생성 프롬프트 (화풍 태그 제외, 인물/배경/행동 중심)")

class YadamScript(BaseModel):
    title: str = Field(description="클릭을 유도하는 매력적인 야담 유튜브 제목")
    hook_summary: str = Field(description="초반 15초 시선 끌기(후킹) 전략 요약")
    scenes: list[Scene] = Field(description="장면별로 분할된 대본 및 삽화 프롬프트 리스트")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "generated_scenes" not in st.session_state:
    st.session_state.generated_scenes = None

# ==========================================
# [1단계] 레퍼런스 유튜브 영상 분석
# ==========================================
st.header("1단계: 벤치마킹할 야담 영상 분석")
col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input("유튜브 영상 링크를 붙여넣으세요", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🔍 영상 구조 분석하기", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("왼쪽 사이드바에 Gemini API 키를 먼저 입력해 주세요.")
    elif not youtube_url:
        st.warning("유튜브 링크를 입력해 주세요.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("올바른 유튜브 링크가 아닙니다.")
        else:
            with st.spinner(f"[{selected_model}] 모델로 자막을 추출하고 이야기 구조를 분석하는 중입니다..."):
                try:
                    transcript_text = fetch_youtube_transcript(video_id)
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
                    다음은 조회수가 잘 나온 유튜브 야담/민담 영상의 전체 자막입니다.
                    이 영상이 시청자를 끝까지 붙잡아 두는 비결을 아래 항목별로 날카롭게 분석해 주세요.
                    
                    1. 초반 15초 후킹(Hook) 기법: 첫 문장을 어떻게 시작해서 호기심을 자극했는가?
                    2. 화법과 문체 특징: 어미 처리(~했더랍니다, ~였지요 등), 호흡의 길이, 분위기 조성 방식
                    3. 기승전결 전개 비율: 발단-위기-절정-결말이 어떤 템포로 배치되었는가?
                    4. 벤치마킹 핵심 공식: 새로운 야담을 쓸 때 그대로 적용할 스토리텔링 공식 3줄 요약

                    [원본 자막]
                    {transcript_text[:15000]}
                    """
                    response = generate_with_retry(client, selected_model, prompt)
                    st.session_state.analysis_result = response.text
                except Exception as e:
                    st.error(f"오류 발생: {e}")

if st.session_state.analysis_result:
    with st.expander("📊 분석된 스토리텔링 공식 보기 (클릭하여 접기/펴기)", expanded=True):
        st.markdown(st.session_state.analysis_result)

st.divider()

# ==========================================
# [2단계] 새로운 야담 대본 & 장면 자동 분할
# ==========================================
st.header("2단계: 새로운 야담 대본 및 삽화 프롬프트 생성")

c1, c2 = st.columns([3, 1])
with c1:
    new_topic = st.text_input(
        "만들고 싶은 새로운 야담 소재나 키워드를 입력하세요",
        placeholder="예: 한양으로 과거 시험 보러 가던 가난한 선비가 산속 폐가에서 만난 여인의 비밀"
    )
with c2:
    scene_count = st.number_input("분할할 장면 수", min_value=4, max_value=20, value=8)

generate_btn = st.button("✍️ 야담 대본 및 장면 생성하기", type="primary")

if generate_btn:
    if not api_key:
        st.error("왼쪽 사이드바에 Gemini API 키를 먼저 입력해 주세요.")
    elif not new_topic:
        st.warning("새로운 야담 주제를 입력해 주세요.")
    else:
        with st.spinner(f"[{selected_model}] 모델이 조선시대 전기수가 되어 대본을 쓰고 장면을 나누는 중입니다..."):
            try:
                client = genai.Client(api_key=api_key)
                
                reference_guide = (
                    f"아래 분석된 인기 영상의 성공 공식을 반드시 적용하세요:\n{st.session_state.analysis_result}"
                    if st.session_state.analysis_result
                    else "초반 15초 안에 강렬한 미스터리나 기이한 사건을 던져 시청자의 시선을 사로잡으세요."
                )

                gen_prompt = f"""
                당신은 조선시대 저잣거리에서 사람들을 울고 웃기는 최고의 이야기꾼(전기수)입니다.
                아래 주제와 가이드를 바탕으로 몰입감 넘치는 야담 대본을 작성하고, 총 {scene_count}개의 장면으로 나누어 주세요.

                [주제]
                {new_topic}

                [적용할 스타일 가이드]
                {reference_guide}

                [작성 규칙]
                1. narration(대사): 성우가 읽기 좋게 구수하고 긴장감 있는 사극 구어체('~했더랍니다', '~였지요', '~한 것이었습니다')를 사용하세요.
                2. image_prompt(그림 묘사): 각 장면에 어울리는 조선시대 배경, 인물 표정, 복식, 조명을 영어로 구체적으로 묘사하세요. (단, 화풍 스타일은 시스템에서 자동 추가하므로 장면 내용만 영어로 작성)
                """

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=YadamScript,
                    temperature=0.8
                )
                response = generate_with_retry(client, selected_model, gen_prompt, config=config)
                
                result_data = json.loads(response.text)
                st.session_state.generated_scenes = result_data
            except Exception as e:
                st.error(f"대본 생성 중 오류 발생: {e}")

# ==========================================
# [3단계] 반자동 검수 및 수정 영역
# ==========================================
if st.session_state.generated_scenes:
    data = st.session_state.generated_scenes
    st.subheader(f"🎬 추천 제목: {data['title']}")
    st.info(f"💡 **초반 후킹 포인트:** {data['hook_summary']}")
    
    st.write("아래 각 장면의 **대본과 그림 프롬프트를 직접 수정**할 수 있습니다.")
    
    edited_scenes = []
    for scene in data["scenes"]:
        with st.container(border=True):
            st.markdown(f"**장면 {scene['scene_number']}**")
            col_text, col_img = st.columns(2)
            
            with col_text:
                new_narration = st.text_area(
                    "내레이션 대본 (수정 가능)",
                    value=scene["narration"],
                    key=f"nar_{scene['scene_number']}",
                    height=110
                )
            with col_img:
                full_prompt = f"{style_anchor} {scene['image_prompt']}"
                new_img_prompt = st.text_area(
                    "최종 삽화 프롬프트 (화풍 자동 결합됨)",
                    value=full_prompt,
                    key=f"img_{scene['scene_number']}",
                    height=110
                )
            edited_scenes.append({
                "scene_number": scene["scene_number"],
                "narration": new_narration,
                "final_image_prompt": new_img_prompt
            })

    final_output = {
        "title": data["title"],
        "scenes": edited_scenes
    }
    st.download_button(
        label="📥 수정 완료된 대본 & 프롬프트 파일 저장 (JSON)",
        data=json.dumps(final_output, ensure_ascii=False, indent=2),
        file_name="yadam_script.json",
        mime="application/json"
    )