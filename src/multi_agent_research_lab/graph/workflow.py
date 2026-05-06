"""Workflow orchestration for the multi-agent lab.

This implementation keeps a graph-shaped orchestration in plain Python so the lab
runs without optional LangGraph dependencies. The returned `build()` structure can
be translated to LangGraph nodes/edges later without changing agent internals.
"""

from __future__ import annotations

from time import perf_counter

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, SupervisorAgent, WriterAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.agents: dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }

    def build(self) -> dict[str, object]:
        """Return a graph specification with nodes, edges, and stop condition."""

        return {
            "nodes": ["supervisor", *self.agents.keys(), "done"],
            "edges": {
                "supervisor": ["researcher", "analyst", "writer", "done"],
                "researcher": ["supervisor"],
                "analyst": ["supervisor"],
                "writer": ["supervisor"],
            },
            "stop_condition": "supervisor routes to done or max_iterations/timeout is reached",
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        settings = get_settings()
        started = perf_counter()
        state.add_trace_event("workflow.started", {"query": state.request.query, "graph": self.build()})

        while True:
            if perf_counter() - started > settings.timeout_seconds:
                state.errors.append("Workflow timeout reached.")
                if not state.final_answer:
                    state.final_answer = "Workflow stopped because timeout was reached before a final answer."
                break

            with trace_span("supervisor", {"iteration": state.iteration}) as supervisor_span:
                state = self.supervisor.run(state)
            state.add_trace_event("span.supervisor", supervisor_span.copy())

            route = state.route_history[-1]
            if route == "done":
                break

            agent = self.agents.get(route)
            if agent is None:
                raise AgentExecutionError(f"Supervisor returned unknown route: {route}")

            try:
                with trace_span(f"agent.{route}", {"iteration": state.iteration}) as agent_span:
                    state = agent.run(state)
                state.add_trace_event(f"span.agent.{route}", agent_span.copy())
            except Exception as exc:
                state.errors.append(f"{route} failed: {exc.__class__.__name__}: {exc}")
                if route == "researcher" and not state.research_notes:
                    state.research_notes = "Researcher failed; no sources were collected."
                elif route == "analyst" and not state.analysis_notes:
                    state.analysis_notes = "Analyst failed; using research notes directly."
                elif route == "writer" and not state.final_answer:
                    state.final_answer = state.analysis_notes or state.research_notes or "Writer failed."
                    break

        state.add_trace_event(
            "workflow.completed",
            {
                "iterations": state.iteration,
                "routes": state.route_history,
                "error_count": len(state.errors),
                "has_final_answer": bool(state.final_answer),
                "latency_seconds": round(perf_counter() - started, 4),
            },
        )
        return state