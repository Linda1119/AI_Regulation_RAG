"""
AI 규제·트렌드 RAG 챗봇 v2 - Streamlit UI
- LangGraph 멀티턴 대화
- 웹 검색 연동
- 문서 출처 하이라이팅
실행: streamlit run app.py
"""

import streamlit as st
import os
from src.rag_pipeline import initialize_pipeline

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="AI 규제·트렌드 RAG 챗봇 v2",
    page_icon="⚖️",
    layout="wide",
)

st.title("📜AI 규제·트렌드 RAG 챗봇")
st.caption("EU AI Act · 국내 AI 기본법 · NIST AI RMF 기반 Agentic RAG")

# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="입력한 키는 세션 동안만 사용됩니다.",
    )

    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        help="입력한 키는 세션 동안만 사용됩니다.",
    )

    st.divider()
    st.subheader("📚 로드된 문서")
    docs_path = "./docs"
    if os.path.exists(docs_path):
        pdfs = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"- {pdf}")
        else:
            st.warning("docs/ 폴더에 PDF가 없습니다.")

    st.divider()
    st.subheader("💡 예시 질문")
    example_questions = [
        "EU AI Act의 고위험 AI 시스템 요건은?",
        "국내 AI 기본법 주요 내용 요약",
        "NIST AI RMF 프레임워크 구조 설명",
        "AI 규제 위반 시 제재는?",
        "최신 글로벌 AI 규제 동향은?",       # 웹 검색 트리거
        "2025년 AI 규제 변화 뭐가 있어?",    # 웹 검색 트리거
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["input_question"] = q

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.rerun()


# ── LangGraph 에이전트 초기화 (캐싱) ─────────────────────
@st.cache_resource(show_spinner="RAG 파이프라인 초기화 중...")
def get_agent(api_key: str, tavily_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    return initialize_pipeline()


# ── 세션 상태 초기화 ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # UI 표시용 메시지
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []    # LangGraph 히스토리


# ── 출처 하이라이팅 함수 ──────────────────────────────────
def render_sources_with_highlight(source_docs: list):
    """
    출처 문서를 하이라이팅 형태로 표시합니다.
    파일명, 페이지, 핵심 내용 미리보기를 함께 보여줍니다.
    """
    if not source_docs:
        return

    with st.expander("📄 참고 문서 출처 (클릭하여 내용 확인)", expanded=False):
        seen = set()
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page", "?")
            key = f"{source}_p{page}"

            if key in seen:
                continue
            seen.add(key)

            filename = os.path.basename(source) if source != "N/A" else "N/A"
            page_num = page + 1 if isinstance(page, int) else page

            # 출처 헤더
            st.markdown(f"**📌 {filename}** — p.{page_num}")

            # 핵심 내용 하이라이팅 (배경색 강조)
            preview = doc.page_content[:300].replace("\n", " ")
            st.markdown(
                f"""<div style="
                    background-color: #fff9c4;
                    border-left: 4px solid #f9a825;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 0.85em;
                    color: #333;
                    margin-bottom: 8px;
                ">{preview}...</div>""",
                unsafe_allow_html=True,
            )


# ── 이전 대화 출력 ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 웹 검색 사용 여부 배지
        if msg.get("used_web"):
            st.caption("🌐 웹 검색 결과 포함")

        # 출처 하이라이팅
        if msg.get("source_docs"):
            render_sources_with_highlight(msg["source_docs"])


# ── 메인 입력 ─────────────────────────────────────────────
question = st.chat_input("AI 규제나 트렌드에 대해 질문하세요.")

if "input_question" in st.session_state:
    question = st.session_state.pop("input_question")

if question:
    if not api_key or not tavily_key:
        st.error("사이드바에 OpenAI API Key와 Tavily API Key를 입력해주세요.")
        st.stop()

    # 사용자 메시지 출력
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # LangGraph 에이전트 실행
    with st.chat_message("assistant"):
        with st.spinner("문서 검색 및 답변 생성 중..."):
            agent = get_agent(api_key, tavily_key)

            # LangGraph 상태로 실행 (대화 히스토리 전달)
            result = agent.invoke({
                "messages": st.session_state.agent_messages,
                "question": question,
                "context": "",
                "web_results": "",
                "answer": "",
                "source_documents": [],
                "use_web": False,
            })

        # 대화 히스토리 업데이트
        st.session_state.agent_messages = result["messages"]

        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        used_web = result.get("use_web", False)
        web_sources = result.get("web_sources", [])

        st.markdown(answer)

        if used_web and web_sources:
            st.caption("웹 검색 출처:")
            for src in web_sources:
                st.caption(f"- {src}")

        render_sources_with_highlight(source_docs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source_docs": source_docs,
        "used_web": used_web,
    })
