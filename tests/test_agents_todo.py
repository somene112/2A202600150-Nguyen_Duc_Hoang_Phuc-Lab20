from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"


def test_worker_agents_fill_state() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent research workflows"))
    state = ResearcherAgent().run(state)
    state = AnalystAgent().run(state)
    state = WriterAgent().run(state)
    assert state.sources
    assert state.research_notes
    assert state.analysis_notes
    assert state.final_answer
    assert "[S1]" in state.final_answer


def test_workflow_produces_final_answer_and_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state of the art"))
    result = MultiAgentWorkflow().run(state)
    assert result.final_answer
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert any(event["name"] == "workflow.completed" for event in result.trace)
