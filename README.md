# ⚖️ AI 규제·트렌드 RAG 챗봇

> **EU AI Act · 국내 AI 기본법 · NIST AI RMF** 등 AI 규제 문서 기반  
> LangChain + LangGraph + Tavily 웹 검색을 활용한 도메인 특화 질의응답 챗봇

---

## 🎯 프로젝트 개요

기업의 AI 도입 시 준수해야 할 국내외 규제 문서를 RAG(Retrieval-Augmented Generation) 방식으로 인덱싱하고, LangGraph로 멀티턴 대화와 실시간 웹 검색을 연동하여 AI 규제 전문 컨설턴트처럼 동작하는 챗봇입니다.

| 항목 | 내용 |
|------|------|
| **목적** | AI 규제 문서 기반 QA 자동화 및 최신 트렌드 검색 |
| **핵심 기술** | LangChain · LangGraph · ChromaDB · OpenAI · Tavily |
| **지원 문서** | EU AI Act, 국내 AI 기본법, NIST AI RMF 1.0, NIST Generative AI Profile |
| **UI** | Streamlit 웹 인터페이스 |

---

## ✨ 주요 기능

### 1. LangGraph 멀티턴 대화
이전 대화를 기억하며 연속적인 질의응답이 가능합니다.
```
사용자: EU AI Act 고위험 AI 요건이 뭐야?
챗봇: (답변)
사용자: 그럼 한국이랑 비교하면?  ← 이전 맥락 기억
챗봇: (EU AI Act와 한국 AI 기본법 비교 답변)
```

### 2. 실시간 웹 검색 연동 (Tavily)
최신 뉴스·트렌드 관련 질문은 자동으로 웹 검색 후 답변에 반영합니다.
- 문서 기반 질문 → RAG만 사용
- 최신 동향 질문 → RAG + Tavily 웹 검색 병행

### 3. 문서 출처 하이라이팅
답변에 사용된 문서의 파일명, 페이지, 핵심 내용을 노란 박스로 표시합니다.

### 4. 웹 검색 출처 URL 표시
웹 검색 시 참고한 뉴스 기사 URL을 함께 제공합니다.

---

## 🏗️ 아키텍처

```
📄 PDF 문서 (docs/)
     ↓ PyPDFLoader
📝 Document Chunks (chunk_size=1000, overlap=200)
     ↓ OpenAI text-embedding-3-small
🗄️ ChromaDB 벡터 저장소

           질문 입력
               ↓
    [LangGraph 에이전트]
               ↓
    웹 검색 필요 여부 판단
       ↙             ↘
   RAG만 사용      RAG + Tavily 웹 검색
       ↘             ↙
        GPT-4o-mini
        + 규제 전문가 프롬프트
        + 대화 히스토리
               ↓
     Streamlit UI (출처 하이라이팅 + URL)
```

---

## 📁 디렉토리 구조

```
ai-regulation-rag/
├── app.py                  # Streamlit UI 메인
├── src/
│   └── rag_pipeline.py     # LangGraph RAG 파이프라인
├── docs/                   # PDF 문서 폴더
│   ├── eu_ai_act.pdf
│   ├── ai_basic_law_korea.pdf
│   ├── nist_ai_rmf.pdf
│   └── nist_gen_ai_profile.pdf
├── data/
│   └── chroma_db/          # 벡터 DB (자동 생성)
├── .env                    # API 키 관리 (gitignore)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 설치 및 실행

### 1. 레포지토리 클론

```bash
git clone https://github.com/YOUR_ID/ai-regulation-rag.git
cd ai-regulation-rag
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```
TAVILY_API_KEY=tvly-xxxx
```

> OpenAI API Key는 실행 후 Streamlit 사이드바에서 입력합니다.

### 4. 문서 추가

`docs/` 폴더에 PDF 파일을 넣습니다.

| 문서 | 다운로드 |
|------|---------|
| EU AI Act | [EUR-Lex 공식](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) |
| 국내 AI 기본법 | [국가법령정보센터](https://www.law.go.kr) |
| NIST AI RMF 1.0 | [NIST 공식](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf) |
| NIST Generative AI Profile | [NIST 공식](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf) |

### 5. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 → 사이드바에 OpenAI API Key 입력 → 질문 시작!

---

## 💬 예시 질의응답

**Q: EU AI Act에서 고위험 AI 시스템의 요건은?**
> EU AI Act Article 9에 따르면, 고위험 AI 시스템은 리스크 관리 시스템을 갖추어야 하며...
> 📄 출처: eu_ai_act.pdf (p.42)

**Q: 2026년 최신 AI 규제 뉴스 알려줘**
> (Tavily 웹 검색 결과 반영)
> 🌐 웹 검색 출처: https://...

---

## 🛠️ 기술 스택

| 레이어 | 기술 |
|--------|------|
| RAG 오케스트레이션 | LangChain 0.3 |
| 에이전트 흐름 제어 | LangGraph 0.3 |
| 벡터 저장소 | ChromaDB |
| 임베딩 | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| 웹 검색 | Tavily Search API |
| UI | Streamlit |
| 문서 파싱 | PyPDF |

---

## 🔑 핵심 설계 포인트

- **LangGraph 조건부 분기**: 질문 유형에 따라 RAG 단독 vs RAG+웹검색 자동 선택
- **MMR Retriever**: 유사 청크 중복 방지로 답변 다양성 향상
- **멀티턴 히스토리**: 최근 4턴 대화를 참고해 질문 재구성 후 검색
- **도메인 특화 프롬프트**: AI 규제 전문가 페르소나 + 웹 검색 결과 강제 반영
- **벡터 DB 캐싱**: 최초 1회만 임베딩 후 재사용으로 API 비용 절약
- **출처 하이라이팅**: 답변 근거 투명성 확보 + 웹 검색 URL 표시
