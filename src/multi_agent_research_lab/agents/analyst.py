"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            state.analysis_notes = "Analysis could not run because research notes are missing."
            state.errors.append("Analyst received empty research notes.")
            metadata = {"has_research_notes": False}
        else:
            sources_block = self._format_sources(state)

            system_prompt = (
                "You are the Analyst agent in a multi-agent research workflow. "
                "Your job is to analyze collected evidence, identify key claims, "
                "assess source quality, and highlight risks. "
                "Use only the provided sources and research notes. "
                "Do not invent citations. Cite sources using [S1], [S2], etc."
            )

            user_prompt = "\n\n".join(
                [
                    f"User query:\n{state.request.query}",
                    f"Audience:\n{state.request.audience}",
                    f"Research notes:\n{state.research_notes}",
                    f"Sources:\n{sources_block}",
                    (
                        "Write concise analysis notes with these sections:\n"
                        "1. Key claims\n"
                        "2. Evidence quality\n"
                        "3. Risks / weak evidence\n"
                        "4. What the writer should emphasize\n\n"
                        "Keep it grounded and cite relevant sources using [S1], [S2], etc."
                    ),
                ]
            )

            response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

            # If provider fails, LLMClient returns the deterministic offline fallback.
            # For the lab, a structured local analysis is more useful than a generic fallback string.
            if response.content.startswith("Offline LLM fallback:"):
                state.analysis_notes = self._local_analysis(state)
                provider_mode = "fallback_local_analysis"
            else:
                state.analysis_notes = response.content
                provider_mode = "real_llm"

            metadata = {
                "has_research_notes": bool(state.research_notes),
                "provider_mode": provider_mode,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }

            state.add_trace_event(
                "agent.analyst.llm_completed",
                {
                    "provider_mode": provider_mode,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes or "",
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "agent.analyst.completed",
            {"analysis_chars": len(state.analysis_notes or ""), "error_count": len(state.errors)},
        )
        return state

    def _format_sources(self, state: ResearchState) -> str:
        if not state.sources:
            return "No sources available."

        lines: list[str] = []
        for index, source in enumerate(state.sources, start=1):
            url = source.url or "no-url"
            score = source.metadata.get("score", "n/a")
            lines.append(
                f"[S{index}] Title: {source.title}\n"
                f"URL: {url}\n"
                f"Snippet: {source.snippet}\n"
                f"Score: {score}"
            )
        return "\n\n".join(lines)

    def _local_analysis(self, state: ResearchState) -> str:
        claim_lines: list[str] = []
        risk_lines: list[str] = []

        for index, source in enumerate(state.sources, start=1):
            claim_lines.append(f"- Claim {index}: {source.snippet} [S{index}]")
            if source.metadata.get("score", 0) == 0:
                risk_lines.append(f"- Source [S{index}] may be weak because it has no lexical match.")

        if not risk_lines:
            risk_lines.append(
                "- Main risk: local/mock search is useful for lab reproducibility "
                "but not enough for a real literature review."
            )

        return "\n".join(
            [
                "Analysis notes:",
                "Key claims:",
                *claim_lines,
                "Evidence quality:",
                f"- Sources collected: {len(state.sources)}",
                f"- Citation coverage target: cite all {len(state.sources)} sources in the final answer where relevant.",
                "Risks / weak evidence:",
                *risk_lines,
            ]
        )