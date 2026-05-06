"""Search client abstraction.

The researcher agent depends on this client instead of calling a search provider
directly. If TAVILY_API_KEY is configured, it uses Tavily real web search.
Otherwise, it falls back to deterministic local results for lab reproducibility.
"""

from __future__ import annotations

import importlib
from typing import Any

from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Search provider client with Tavily support and local fallback."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return search results from Tavily when configured, else local fallback."""

        load_dotenv()
        settings = get_settings()

        tavily_api_key = getattr(settings, "tavily_api_key", None)
        if tavily_api_key:
            try:
                return self._search_tavily(
                    query=query,
                    max_results=max_results,
                    api_key=tavily_api_key,
                )
            except Exception as exc:  # pragma: no cover - external provider dependent
                fallback_sources = self._local_search(query=query, max_results=max_results)
                for source in fallback_sources:
                    source.metadata["provider_fallback"] = f"Tavily failed: {exc.__class__.__name__}"
                return fallback_sources

        return self._local_search(query=query, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        """Search Tavily and map results into the lab SourceDocument schema."""

        tavily_module = importlib.import_module("tavily")
        tavily_client_cls: Any = getattr(tavily_module, "TavilyClient")
        tavily_client: Any = tavily_client_cls(api_key=api_key)

        response: dict[str, Any] = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        results = response.get("results", [])
        sources: list[SourceDocument] = []

        for index, item in enumerate(results, start=1):
            title = str(item.get("title") or f"Tavily result {index}").strip()
            url = item.get("url")
            content = str(item.get("content") or item.get("snippet") or "").strip()
            score = item.get("score")

            if not content:
                content = "No snippet was returned by Tavily for this result."

            sources.append(
                SourceDocument(
                    title=title,
                    url=str(url) if url else None,
                    snippet=content,
                    metadata={
                        "rank": index,
                        "score": score if score is not None else 0,
                        "provider": "tavily",
                    },
                )
            )

        if not sources:
            return [
                SourceDocument(
                    title="No Tavily results",
                    url=None,
                    snippet="Tavily returned no results for this query.",
                    metadata={"rank": 1, "score": 0, "provider": "tavily"},
                )
            ]

        return sources

    def _local_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Deterministic local fallback used for tests and offline demos."""

        documents = [
            SourceDocument(
                title="Building effective agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Reliable agent systems should use simple workflows first, clear tool boundaries, "
                    "and evaluator feedback before adding complex autonomous behavior."
                ),
                metadata={
                    "rank": 1,
                    "score": 3.5,
                    "tags": ["agent", "workflow", "guardrail", "evaluation"],
                    "provider": "local",
                },
            ),
            SourceDocument(
                title="Production guardrails for LLM systems",
                url="https://platform.openai.com/docs/guides/safety-best-practices",
                snippet=(
                    "Production LLM systems need input validation, output validation, rate limits, "
                    "fallbacks, monitoring, and human review for high-risk decisions."
                ),
                metadata={
                    "rank": 2,
                    "score": 3.0,
                    "tags": ["guardrail", "validation", "fallback", "monitoring", "safety"],
                    "provider": "local",
                },
            ),
            SourceDocument(
                title="OpenAI Agents SDK orchestration and handoffs",
                url="https://developers.openai.com/agents",
                snippet=(
                    "Agent orchestration benefits from explicit handoffs, typed inputs and outputs, "
                    "tool-use boundaries, and traceable execution."
                ),
                metadata={
                    "rank": 3,
                    "score": 2.5,
                    "tags": ["openai", "handoff", "orchestration", "agent", "trace"],
                    "provider": "local",
                },
            ),
            SourceDocument(
                title="LangGraph workflow concepts",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "Graph-based orchestration models agent work as nodes, edges, state transitions, "
                    "conditional routing, and stop conditions."
                ),
                metadata={
                    "rank": 4,
                    "score": 1.0,
                    "tags": ["langgraph", "graph", "state", "routing", "workflow"],
                    "provider": "local",
                },
            ),
            SourceDocument(
                title="LangSmith tracing for LLM applications",
                url="https://docs.smith.langchain.com/",
                snippet=(
                    "Tracing helps teams inspect each model call, tool call, latency, error, "
                    "and intermediate decision in an LLM application."
                ),
                metadata={
                    "rank": 5,
                    "score": 1.0,
                    "tags": ["trace", "observability", "latency", "debug"],
                    "provider": "local",
                },
            ),
        ]

        query_terms = {term.lower() for term in query.split() if len(term) > 3}

        def score(document: SourceDocument) -> float:
            haystack = f"{document.title} {document.snippet} {' '.join(document.metadata.get('tags', []))}".lower()
            lexical_score = sum(1 for term in query_terms if term in haystack)
            base_score = float(document.metadata.get("score", 0))
            return lexical_score + base_score

        ranked = sorted(documents, key=score, reverse=True)
        return ranked[:max_results]