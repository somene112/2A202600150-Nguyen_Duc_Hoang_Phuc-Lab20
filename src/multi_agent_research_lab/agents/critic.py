"""Optional critic agent implementation for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and citation-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        if not state.final_answer:
            finding = "Critic could not run because final answer is missing."
            state.errors.append(finding)
        else:
            missing = [
                f"S{index}"
                for index, _source in enumerate(state.sources, start=1)
                if f"[S{index}]" not in state.final_answer
            ]
            if missing:
                finding = f"Citation check: missing citations for {', '.join(missing)}."
                state.errors.append(finding)
            else:
                finding = "Citation check passed: all collected sources are referenced in the final answer."

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC, content=finding, metadata={"error_count": len(state.errors)}
            )
        )
        state.add_trace_event("agent.critic.completed", {"finding": finding})
        return state
