"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with source references."""

        if not state.analysis_notes:
            state.errors.append("Writer received empty analysis notes.")

        sources_block = self._format_sources(state)

        system_prompt = (
            "You are the Writer agent in a multi-agent research workflow. "
            "Write a clear, evidence-backed final answer for technical learners. "
            "Use only the provided research notes, analysis notes, and sources. "
            "Do not invent sources. Cite sources using [S1], [S2], etc."
        )

        user_prompt = "\n\n".join(
            [
                f"User query:\n{state.request.query}",
                f"Audience:\n{state.request.audience}",
                f"Research notes:\n{state.research_notes or 'No research notes available.'}",
                f"Analysis notes:\n{state.analysis_notes or 'No analysis notes available.'}",
                f"Sources:\n{sources_block}",
                (
                    "Write the final answer in Markdown with these sections:\n"
                    "# Research answer: <query>\n"
                    "## Executive summary\n"
                    "## Evidence-backed points\n"
                    "## Recommendation\n"
                    "## Limitations\n"
                    "## Sources\n\n"
                    "Every important claim should cite sources using [S1], [S2], etc. "
                    "The Sources section must list every provided source."
                ),
            ]
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        if response.content.startswith("Offline LLM fallback:"):
            state.final_answer = self._local_final_answer(state)
            provider_mode = "fallback_local_writer"
        else:
            state.final_answer = self._ensure_sources_section(response.content, state)
            provider_mode = "real_llm"

        metadata = {
            "source_count": len(state.sources),
            "word_count": len(state.final_answer.split()),
            "provider_mode": provider_mode,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        }

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata=metadata,
            )
        )

        state.add_trace_event(
            "agent.writer.llm_completed",
            {
                "provider_mode": provider_mode,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.add_trace_event(
            "agent.writer.completed",
            {"answer_chars": len(state.final_answer), "source_count": len(state.sources)},
        )
        return state

    def _format_sources(self, state: ResearchState) -> str:
        if not state.sources:
            return "No sources available."

        lines: list[str] = []
        for index, source in enumerate(state.sources, start=1):
            url = source.url or "no-url"
            lines.append(f"[S{index}] Title: {source.title}\nURL: {url}\nSnippet: {source.snippet}")
        return "\n\n".join(lines)

    def _source_lines(self, state: ResearchState) -> list[str]:
        source_lines = []
        for index, source in enumerate(state.sources, start=1):
            url = source.url or "no-url"
            source_lines.append(f"[S{index}] {source.title} - {url}")
        return source_lines or ["No sources available."]

    def _ensure_sources_section(self, content: str, state: ResearchState) -> str:
        """Ensure final answer includes a source list for citation coverage."""

        answer = content.strip()
        if "## Sources" in answer:
            return answer

        return "\n".join(
            [
                answer,
                "",
                "## Sources",
                *self._source_lines(state),
            ]
        )

    def _local_final_answer(self, state: ResearchState) -> str:
        source_lines = self._source_lines(state)

        evidence_summary = "\n".join(
            f"- {source.snippet} [S{index}]" for index, source in enumerate(state.sources, start=1)
        )
        if not evidence_summary:
            evidence_summary = (
                "- No external evidence was collected; treat this answer as low confidence."
            )

        return "\n".join(
            [
                f"# Research answer: {state.request.query}",
                "",
                "## Executive summary",
                (
                    "A good multi-agent research system should split work into routing, "
                    "evidence collection, analysis, and writing. This makes the workflow "
                    "easier to debug than a single large prompt, especially when trace "
                    "and benchmark data are recorded."
                ),
                "",
                "## Evidence-backed points",
                evidence_summary,
                "",
                "## Recommendation",
                (
                    "Use the multi-agent workflow when the task needs evidence gathering, "
                    "intermediate reasoning, and auditable handoffs. Use a single-agent "
                    "baseline when the task is simple, low-risk, or latency/cost is more "
                    "important than decomposition."
                ),
                "",
                "## Limitations",
                (
                    "This lab version uses deterministic local search unless external API "
                    "keys are configured, so real-world claims should be rechecked with "
                    "live search or a trusted literature database."
                ),
                "",
                "## Sources",
                *source_lines,
            ]
        )
