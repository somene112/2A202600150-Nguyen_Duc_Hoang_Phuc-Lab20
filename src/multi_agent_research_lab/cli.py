"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_single_agent(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    sources = SearchClient().search(query, max_results=request.max_sources)
    state.sources = sources
    evidence = "\n".join(
        f"- {source.snippet} [S{index}]" for index, source in enumerate(sources, start=1)
    )
    source_list = "\n".join(
        f"[S{index}] {source.title} - {source.url or 'no-url'}"
        for index, source in enumerate(sources, start=1)
    )
    state.final_answer = "\n".join(
        [
            f"# Single-agent baseline: {query}",
            "",
            "This baseline performs search and writes the answer in one step. It is simple and fast, but it does not expose separate research and analysis handoffs.",
            "",
            "## Evidence",
            evidence,
            "",
            "## Sources",
            source_list,
        ]
    )
    state.add_trace_event("baseline.completed", {"source_count": len(sources)})
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a deterministic single-agent baseline."""

    _init()
    state = _run_single_agent(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    result = MultiAgentWorkflow().run(state)
    console.print(Panel.fit(result.final_answer or "No final answer", title="Multi-Agent Answer"))
    console.print_json(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Report path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Compare single-agent and multi-agent workflows and write a report."""

    _init()
    _single_state, single_metrics = run_benchmark("single-agent", query, _run_single_agent)
    _multi_state, multi_metrics = run_benchmark(
        "multi-agent",
        query,
        lambda item: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=item))),
    )
    report = render_markdown_report([single_metrics, multi_metrics])
    store = LocalArtifactStore(output.parent)
    path = store.write_text(output.name, report)
    console.print(Panel.fit(report, title=f"Benchmark written to {path}"))


if __name__ == "__main__":
    app()
