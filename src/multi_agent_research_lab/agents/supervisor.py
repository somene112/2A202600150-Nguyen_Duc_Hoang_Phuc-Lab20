"""Supervisor / router implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        1. research if sources/notes are missing;
        2. analyze if research exists but analysis is missing;
        3. write if analysis exists but final answer is missing;
        4. stop when final answer exists or max iterations is reached.
        """

        settings = get_settings()
        if state.iteration >= settings.max_iterations:
            route = "done"
            if not state.final_answer:
                state.final_answer = self._fallback_answer(state)
                state.errors.append("Max iterations reached; returned fallback answer.")
        elif not state.sources or not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event(
            "supervisor.route_decision",
            {
                "route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state

    def _fallback_answer(self, state: ResearchState) -> str:
        if state.analysis_notes:
            return f"Fallback answer from available analysis:\n\n{state.analysis_notes}"
        if state.research_notes:
            return f"Fallback answer from available research notes:\n\n{state.research_notes}"
        return "Fallback answer: the workflow stopped before enough evidence was collected."
