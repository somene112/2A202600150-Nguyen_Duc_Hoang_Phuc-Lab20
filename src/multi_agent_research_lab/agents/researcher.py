"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self.search_client.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources

        if not sources:
            state.research_notes = "No sources were found. Continue with a cautious fallback answer."
            state.errors.append("Researcher found no sources.")
        else:
            lines = ["Research notes:"]
            for index, source in enumerate(sources, start=1):
                url = source.url or "no-url"
                score = source.metadata.get("score", "n/a")
                lines.append(f"[S{index}] {source.title} ({url}) - {source.snippet} score={score}")
            state.research_notes = "\n".join(lines)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event(
            "agent.researcher.completed",
            {"source_count": len(sources), "titles": [source.title for source in sources]},
        )
        return state