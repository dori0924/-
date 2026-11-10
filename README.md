# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델이 전통 관상학 관점으로 풀이해 주는 Streamlit 앱입니다.
결과는 재미를 위한 것이며 과학적 근거가 없습니다. 업로드한 사진은 저장되지 않습니다.

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 실제 OPENAI_API_KEY 입력
streamlit run app.py
```

브라우저에서 http://localhost:8501 로 접속합니다.

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 푸시합니다. (`.env`는 `.gitignore`로 제외됩니다)
2. https://share.streamlit.io 에서 **New app** → 저장소 선택 → Main file path: `app.py`
3. **Advanced settings → Secrets**에 아래 내용을 입력합니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```

4. **Deploy**를 누릅니다.

## 환경변수

| 이름 | 설명 | 기본값 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `OPENAI_MODEL` | 이미지 입력을 지원하는 OpenAI 모델 | `gpt-4o-mini` |
