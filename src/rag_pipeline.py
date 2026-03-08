"""
AI 규제·트렌드 RAG 파이프라인 v2
LangChain + LangGraph + ChromaDB + OpenAI 기반
- LangGraph 멀티턴 대화 (이전 대화 기억)
- 웹 검색 연동 (최신 규제 뉴스 실시간 검색)
- 문서 출처 하이라이팅
"""

import os
from pathlib import Path
from typing import List, TypedDict, Annotated
import operator
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END


# ── 1. 문서 로드 ──────────────────────────────────────────
def load_documents(docs_dir: str = "./docs") -> List:
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"✅ 총 {len(documents)}개 페이지 로드 완료")
    return documents


# ── 2. 청크 분할 ──────────────────────────────────────────
def split_documents(documents: List) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 총 {len(chunks)}개 청크로 분할 완료")
    return chunks


# ── 3. 벡터 DB 구축 ──────────────────────────────────────
def build_vectorstore(chunks: List, persist_dir: str = "./data/chroma_db") -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        print("✅ 기존 벡터 DB 로드 중...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        print("🔨 벡터 DB 새로 구축 중...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        print("✅ 벡터 DB 구축 완료")

    return vectorstore


# ── 4. LangGraph 상태 정의 ────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    context: str
    web_results: str
    web_sources: list
    answer: str
    source_documents: list
    use_web: bool


# ── 5. LangGraph 노드 정의 ────────────────────────────────
def build_langgraph_agent(vectorstore: Chroma):
    """
    LangGraph로 멀티턴 RAG 에이전트를 구성합니다.
    
    흐름:
    질문 입력 → [웹 검색 필요 판단] → RAG 검색 → (웹 검색) → 답변 생성
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10},
    )
    web_search = TavilySearchResults(max_results=3)

    # ── 노드 1: 웹 검색 필요 여부 판단 ──
    def decide_web_search(state: AgentState) -> AgentState:
        """
        질문이 최신 뉴스/트렌드 관련이면 웹 검색 활성화
        규제 문서 내용 질문이면 RAG만 사용
        """
        question = state["question"]
        decision_prompt = f"""다음 질문이 최신 뉴스, 최근 동향, 실시간 정보가 필요한지 판단하세요.
질문: {question}

최신 정보가 필요하면 'yes', 문서 내용만으로 충분하면 'no'만 답하세요."""

        result = llm.invoke(decision_prompt)
        use_web = "yes" in result.content.lower()
        return {**state, "use_web": use_web}

    # ── 노드 2: RAG 문서 검색 ──
    def retrieve_documents(state: AgentState) -> AgentState:
        """벡터 DB에서 관련 문서 청크를 검색합니다."""
        question = state["question"]

        # 대화 히스토리를 반영한 질문 재구성
        if len(state["messages"]) > 0:
            history_text = "\n".join([
                f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in state["messages"][-4:]  # 최근 4턴만 참조
            ])
            rewrite_prompt = f"""이전 대화를 참고해서 현재 질문을 독립적으로 이해할 수 있게 재작성하세요.
이전 대화:
{history_text}

현재 질문: {question}
재작성된 질문 (한 문장):"""
            rewritten = llm.invoke(rewrite_prompt)
            search_query = rewritten.content
        else:
            search_query = question

        docs = retriever.invoke(search_query)
        context = "\n\n".join([
            f"[출처: {doc.metadata.get('source', 'N/A')} | p.{doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for doc in docs
        ])
        return {**state, "context": context, "source_documents": docs}

# ── 노드 3: 웹 검색 (선택적) ──
    def search_web(state: AgentState) -> AgentState:
        """최신 AI 규제 뉴스를 웹에서 검색합니다."""
        if not state.get("use_web", False):
            return {**state, "web_results": "", "web_sources": []}

        query = f"AI 규제 {state['question']} 2025"
        try:
            results = web_search.invoke(query)
            # URL 출처 따로 추출
            web_sources = [
                f"{r.get('title', 'N/A')} - {r.get('url', 'N/A')}"
                for r in results if isinstance(r, dict)
            ]
            web_results = f"[최신 웹 검색 결과]\n{str(results)[:1500]}" if results else ""
        except Exception:
            web_results = ""
            web_sources = []
        return {**state, "web_results": web_results, "web_sources": web_sources}
    
# ── 노드 4: 답변 생성 ──
    def generate_answer(state: AgentState) -> AgentState:
        """RAG 문서 + 웹 검색 + 대화 히스토리를 종합해 답변합니다."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 AI 규제 및 기술 트렌드 전문 컨설턴트입니다.

규칙:
- 제공된 문서 내용을 근거로 답변하세요.
- 웹 검색 결과가 있으면 반드시 해당 내용을 답변에 포함하세요.
- 웹 검색 결과가 있으면 문서에 없더라도 웹 검색 결과를 활용해서 답변하세요.
- 관련 규제명, 조항번호, 출처를 가능한 한 포함하세요.
- 한국어로 답변하되, 원문 용어는 그대로 사용하세요.
- 이전 대화 내용을 참고해 일관성 있게 답변하세요.

[검색된 문서]
{context}

[웹 검색 결과]
{web_results}

주의: web_results가 비어있지 않으면 반드시 답변에 반영하세요."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm
        result = chain.invoke({
            "context": state["context"],
            "web_results": state.get("web_results", "없음"),
            "history": state["messages"],
            "question": state["question"],
        })

        new_messages = [
            HumanMessage(content=state["question"]),
            AIMessage(content=result.content),
        ]

        return {
            **state,
            "answer": result.content,
            "messages": state["messages"] + new_messages,
        }

    # ── 라우터: 웹 검색 여부에 따라 분기 ──
    def route_web_search(state: AgentState) -> str:
        return "search_web" if state.get("use_web", False) else "generate_answer"

    # ── LangGraph 그래프 구성 ──
    graph = StateGraph(AgentState)

    graph.add_node("decide_web_search", decide_web_search)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("search_web", search_web)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("decide_web_search")
    graph.add_edge("decide_web_search", "retrieve_documents")
    graph.add_conditional_edges(
        "retrieve_documents",
        route_web_search,
        {
            "search_web": "search_web",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("search_web", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ── 6. 파이프라인 초기화 ──────────────────────────────────
def initialize_pipeline(docs_dir: str = "./docs", db_dir: str = "./data/chroma_db"):
    """전체 RAG + LangGraph 파이프라인을 초기화합니다."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"'{docs_dir}' 폴더에 PDF 문서가 없습니다.")
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks, db_dir)
    agent = build_langgraph_agent(vectorstore)
    return agent
