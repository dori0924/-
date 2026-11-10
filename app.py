"""AI 관상가 - 얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델로 관상을 풀이해 주는 Streamlit 앱."""

import base64
import io
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_MB = 10
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
# 파일 용량이 작아도 픽셀 수가 매우 크면 디코딩 시 메모리를 수백 MB 쓰므로 별도로 제한한다.
MAX_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85
MAX_OUTPUT_TOKENS = 1500
REQUEST_TIMEOUT_SEC = 60
# 세션당 API 호출 상한 (PRD 11. 비용 리스크). 실제 비용 상한은 OpenAI 대시보드 사용 한도로 설정할 것.
MAX_ANALYSES_PER_SESSION = 10

DISCLAIMER = "본 결과는 재미를 위한 것이며 과학적 근거가 없습니다. 업로드한 사진은 저장되지 않습니다."

SOURCE_UPLOAD = "사진 업로드"
SOURCE_CAMERA = "카메라 촬영"

# ---------------------------------------------------------------------------
# 프롬프트 (PRD 5.2, 5.3)
# ---------------------------------------------------------------------------
RETRY_MARKER = "[재업로드]"

SYSTEM_PROMPT = f"""\
당신은 친근하고 유쾌한 전통 관상가입니다. 사용자가 보낸 얼굴 사진을 보고 \
전통 관상학의 관점에서 재미로 즐길 수 있는 풀이를 한국어로 들려줍니다.

사진에 보이는 이마, 눈썹, 눈, 코, 입, 귀, 얼굴형 같은 외형적 특징을 관상학 용어로 풀이하는 것이 당신의 일입니다. \
인물이 누구인지 알 필요는 없으며, 보이는 얼굴 특징만으로 풀이하면 됩니다.

[반드시 지킬 규칙]
- 긍정적이고 존중하는 말투를 사용하세요. 외모를 비하하거나 조롱하지 마세요.
- 인종, 민족, 종교, 성적 지향, 건강 상태, 장애 등 민감한 속성을 추정하거나 언급하지 마세요.
- 사진 속 인물의 이름이나 신원을 추정하거나 언급하지 마세요.
- 이 풀이는 엔터테인먼트 목적이며 과학적 근거가 없다는 전제로 작성하세요.

[여러 사람이 보이는 사진]
가장자리에 다른 사람이 일부 보이거나 여러 명이 있어도, 가장 크게·가운데에 보이는 한 사람의 얼굴을 풀이하세요.

[풀이가 불가능한 사진]
사진에 사람의 얼굴이 전혀 없거나, 얼굴이 가려지거나 너무 흐려서 눈·코·입을 구분할 수 없을 때만 \
풀이를 하지 말고 첫 줄을 반드시 {RETRY_MARKER} 로 시작해서 두 줄로 답하세요.
첫 줄: {RETRY_MARKER} 뒤에 이 사진에서 실제로 보이는 내용을 근거로 한 이유 한 문장
둘째 줄: 정면을 바라보는 한 사람의 얼굴 사진을 다시 올려 주세요.
얼굴이 또렷하게 보인다면 이 형식을 쓰지 말고 반드시 풀이하세요.

[출력 형식]
얼굴 분석이 가능하면 아래 7개 섹션을 마크다운 소제목(###)으로 순서대로 작성하세요.

### 첫인상 요약
한두 문장.

### 부위별 풀이
이마, 눈썹, 눈, 코, 입, 귀, 턱/얼굴형을 각각 글머리표로 짧게 풀이. 사진에서 보이지 않는 부위는 "사진으로 확인하기 어려움"이라고 적으세요.

### 성격과 기질

### 재물운 / 직업운

### 대인관계·연애운

### 조언
건강하게 지내기 위한 가벼운 생활 조언 수준으로만 작성하세요.

### 한 줄 총평
"""

USER_PROMPT = "이 사진 속 인물의 관상을 재미로 풀이해 주세요."


class ImageTooLargeError(Exception):
    """픽셀 수가 MAX_PIXELS를 넘는 이미지."""


# ---------------------------------------------------------------------------
# 설정 로딩
# ---------------------------------------------------------------------------
def _get_secret(name: str) -> str | None:
    """st.secrets에서 값을 읽는다. secrets.toml이 없는 로컬 환경에서는 None을 반환한다."""
    try:
        value = st.secrets.get(name)
    except Exception:  # secrets.toml이 없으면 StreamlitSecretNotFoundError 등이 발생
        return None
    return str(value) if value else None


def load_config() -> tuple[str, str]:
    """API 키와 모델명을 st.secrets → .env 순서로 읽는다. 키가 없으면 앱을 중단한다."""
    load_dotenv()
    api_key = _get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = _get_secret("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL

    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 Secrets에 `OPENAI_API_KEY`를 설정해 주세요."
        )
        st.stop()
    return api_key, model


# ---------------------------------------------------------------------------
# 이미지 처리 (메모리에서만 처리, 디스크 저장 없음)
# ---------------------------------------------------------------------------
def open_image(data: bytes) -> Image.Image:
    """바이트에서 이미지를 열고 EXIF 회전을 보정한다.

    Image.open은 헤더만 읽으므로, 픽셀 데이터를 디코딩(load)하기 전에 크기를 검사한다.
    손상된 파일이면 UnidentifiedImageError, 너무 크면 ImageTooLargeError.
    """
    image = Image.open(io.BytesIO(data))
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise ImageTooLargeError(f"{width}x{height}")
    image.load()
    return ImageOps.exif_transpose(image)


def to_rgb(image: Image.Image) -> Image.Image:
    """투명도가 있는 이미지는 흰 배경에 합성하고, 그 외는 RGB로 변환한다."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def prepare_image(image: Image.Image) -> str:
    """RGB 변환 → 긴 변 최대 1024px 리사이즈 → JPEG → base64 data URL."""
    img = to_rgb(image)
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# OpenAI 관상 분석
# ---------------------------------------------------------------------------
def analyze_face(image: Image.Image, api_key: str, model: str) -> str:
    """이미지를 OpenAI Vision 모델에 보내 관상 분석 결과(마크다운)를 반환한다.

    openai 예외는 호출한 쪽에서 처리한다.
    """
    data_url = prepare_image(image)
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC, max_retries=1)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    if not content:
        refusal = getattr(message, "refusal", None)
        raise ValueError(refusal or "모델이 빈 응답을 반환했습니다.")
    return content


def describe_error(error: Exception) -> str:
    """예외를 사용자에게 보여줄 한국어 메시지로 바꾼다. (PRD 9. 예외 상황 처리)"""
    if isinstance(error, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. `OPENAI_API_KEY` 설정을 확인해 주세요."
    if isinstance(error, openai.RateLimitError):
        return "요청 한도를 초과했습니다(429). 잠시 후 다시 시도하거나 OpenAI 사용량/결제 상태를 확인해 주세요."
    if isinstance(error, openai.APITimeoutError):
        return "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.BadRequestError):
        return f"요청이 거부되었습니다. 다른 사진으로 시도해 주세요. ({error.message})"
    if isinstance(error, openai.APIError):
        return f"OpenAI API 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. ({error.message})"
    if isinstance(error, ValueError):
        return f"분석 결과를 받지 못했습니다. 다른 사진으로 시도해 주세요. ({error})"
    return f"알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. ({type(error).__name__})"


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
def init_state() -> None:
    # result: {"image_id": 분석한 이미지 식별자, "text": 결과} 또는 None
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("analysis_count", 0)
    # 입력 위젯 key에 붙이는 번호. 바꾸면 업로드/카메라 위젯이 새로 만들어져 비워진다.
    st.session_state.setdefault("input_nonce", 0)


def reset_session() -> None:
    """'다른 사진으로 다시 보기' 버튼 콜백: 결과와 입력 위젯을 초기화한다. (호출 횟수는 유지)"""
    st.session_state.result = None
    st.session_state.input_nonce += 1


def current_result(image_id: str) -> str | None:
    """현재 이미지에 대한 결과만 반환한다. 이미지가 바뀌면 이전 결과는 보이지 않는다."""
    result = st.session_state.result
    if result and result["image_id"] == image_id:
        return result["text"]
    return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* 제목: 테마 색상 사용, 가운데 정렬 */
.gwansang-hero { text-align: center; padding: 0.5rem 0 0.25rem; }
.gwansang-hero h1 { font-size: 2.4rem; margin-bottom: 0.25rem; letter-spacing: 0.05em; }
.gwansang-hero p { opacity: 0.8; margin: 0; }
/* 결과 카드 안 소제목 구분선 */
.st-key-result_card h3 {
    font-size: 1.15rem;
    padding-bottom: 0.3rem;
    margin-top: 1.1rem;
    border-bottom: 1px solid rgba(155, 44, 44, 0.35);
}
/* 업로더 기본 안내("NMB per file")는 서버 제한값이라 실제 제한(10MB)과 다르므로 숨긴다.
   허용 형식과 용량은 업로더 라벨에 표시한다. */
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none; }
</style>
"""

UPLOAD_TIPS = "📸 **잘 나오는 사진 팁** · 정면을 바라보는 얼굴 · 밝은 조명 · 한 사람만 나온 사진"


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="gwansang-hero"><h1>🔮 AI 관상가</h1>'
        "<p>얼굴 사진 한 장으로 보는 재미있는 관상 풀이</p></div>",
        unsafe_allow_html=True,
    )
    st.info(DISCLAIMER, icon="ℹ️")


def render_sidebar(model: str) -> None:
    with st.sidebar:
        st.header("📜 사용 방법")
        st.markdown(
            "1. **사진 업로드** 또는 **카메라 촬영** 탭에서 얼굴 사진을 준비합니다.\n"
            "2. **관상 보기** 버튼을 누릅니다.\n"
            "3. 관상가의 풀이를 확인합니다."
        )
        st.header("🔒 개인정보 안내")
        st.markdown(
            "사진은 분석을 위해 OpenAI API로만 전송되며, 이 앱의 서버에 **저장되지 않습니다**."
        )
        st.header("⚠️ 면책")
        st.caption(DISCLAIMER)
        st.caption(f"분석 모델: `{model}`")


def load_input(uploaded, source: str) -> tuple[Image.Image | None, str | None]:
    """업로드/촬영된 파일을 검사하고 연다. 문제가 있으면 메시지를 보여주고 이미지는 None."""
    if uploaded is None:
        return None, None

    image_id = f"{source}:{uploaded.file_id}"
    if uploaded.size > MAX_FILE_BYTES:
        st.warning(f"파일 용량이 {MAX_FILE_MB}MB를 초과합니다. 더 작은 사진을 올려 주세요.")
        return None, image_id

    try:
        image = open_image(uploaded.getvalue())
    except (ImageTooLargeError, Image.DecompressionBombError):
        st.warning("사진 해상도가 너무 큽니다. 일반적인 크기의 사진(예: 4000×4000 이하)을 올려 주세요.")
        return None, image_id
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        st.error("이미지를 열 수 없습니다. 손상되지 않은 jpg/png/webp 파일을 올려 주세요.")
        return None, image_id
    return image, image_id


def render_input() -> tuple[Image.Image | None, str | None]:
    """업로드 또는 카메라 탭에서 사진을 받는다. (이미지, 이미지 식별자)를 반환한다.

    - 업로드 탭은 항상 렌더링한다. 렌더링하지 않으면 Streamlit이 위젯 상태를 지워서
      카메라 탭에 다녀오면 올린 사진이 사라진다.
    - 카메라 탭은 선택됐을 때만 렌더링한다. 업로드 탭에 있을 때 카메라 권한을 요청하지 않기 위해서다.
    """
    nonce = st.session_state.input_nonce
    upload_tab, camera_tab = st.tabs(
        [f"🖼️ {SOURCE_UPLOAD}", f"📷 {SOURCE_CAMERA}"], key="input_tab", on_change="rerun"
    )

    with upload_tab:
        uploaded = st.file_uploader(
            f"얼굴 사진을 올려 주세요 (jpg, png, webp · 최대 {MAX_FILE_MB}MB)",
            type=ALLOWED_TYPES,
            key=f"uploader_{nonce}",
        )
        upload_image, upload_id = load_input(uploaded, SOURCE_UPLOAD)
        if upload_image is not None:
            with st.container(horizontal_alignment="center"):
                st.image(upload_image, caption="업로드한 사진", width=280)

    camera_image, camera_id = None, None
    if camera_tab.open:
        with camera_tab:
            shot = st.camera_input("카메라로 얼굴을 촬영하세요", key=f"camera_{nonce}")
            camera_image, camera_id = load_input(shot, SOURCE_CAMERA)

    st.caption(UPLOAD_TIPS)
    if camera_tab.open:
        return camera_image, camera_id
    return upload_image, upload_id


def is_retry_response(result: str) -> bool:
    """재업로드 안내 응답인지 판단한다. 모델이 표시를 빠뜨려도 7개 섹션(###)이 없으면 안내로 본다."""
    return RETRY_MARKER in result or "###" not in result


def render_result(result: str) -> None:
    if is_retry_response(result):
        st.warning(result.replace(RETRY_MARKER, "").strip(), icon="🙈")
    else:
        st.subheader("📜 관상 풀이")
        with st.container(border=True, key="result_card"):
            st.markdown(result)
        st.caption(DISCLAIMER)
    st.button("🔄 다른 사진으로 다시 보기", on_click=reset_session, width="stretch")


def run_analysis(image: Image.Image, image_id: str, api_key: str, model: str) -> None:
    st.session_state.analysis_count += 1
    with st.spinner("관상가가 얼굴을 살펴보는 중입니다..."):
        try:
            text = analyze_face(image, api_key, model)
        except Exception as error:  # noqa: BLE001 - 사용자에게 메시지로 보여준다
            st.error(describe_error(error))
            return
    st.session_state.result = {"image_id": image_id, "text": text}
    # 버튼은 분석 전에 그려졌으므로, 다시 실행해서 '관상 보기' 버튼을 비활성 상태로 갱신한다.
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    init_state()
    render_header()
    api_key, model = load_config()
    render_sidebar(model)

    image, image_id = render_input()
    if image is None:
        return

    remaining = MAX_ANALYSES_PER_SESSION - st.session_state.analysis_count
    already_done = current_result(image_id) is not None
    clicked = st.button(
        "🔮 관상 보기",
        type="primary",
        width="stretch",
        disabled=already_done or remaining <= 0,
        help="이 사진은 이미 풀이했습니다. 다른 사진으로 시도해 보세요." if already_done else None,
    )
    if remaining <= 0 and not already_done:
        st.warning(f"이번 접속에서 사용할 수 있는 분석 횟수({MAX_ANALYSES_PER_SESSION}회)를 모두 사용했습니다.")
    if clicked:
        run_analysis(image, image_id, api_key, model)

    result = current_result(image_id)
    if result:
        render_result(result)
    elif remaining > 0:
        st.caption(f"남은 분석 횟수: {remaining}/{MAX_ANALYSES_PER_SESSION}")


if __name__ == "__main__":
    main()
