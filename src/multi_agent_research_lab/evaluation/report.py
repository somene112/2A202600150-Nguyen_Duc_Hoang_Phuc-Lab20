"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "This report compares a single-agent baseline with the multi-agent workflow using latency, estimated cost, quality, citation coverage, source count, and error count.",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation coverage | Sources | Errors | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        notes = item.notes.replace("|", "/")
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {coverage} | {item.source_count} | {item.error_count} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Single-agent is faster and simpler, but it gives less visibility into intermediate research and analysis.",
            "- Multi-agent is more auditable because each role writes to shared state and trace events show the handoff path.",
            "- Use multi-agent only when decomposition, evidence tracking, or reviewability is worth the extra orchestration cost.",
            "",
            "## Failure modes and fixes",
            "- Weak or missing sources: use a stronger search provider and validate citation coverage.",
            "- Infinite loops: enforce max_iterations and explicit `done` routing.",
            "- High latency/cost: benchmark against a single-agent baseline and remove unnecessary agents.",
        ]
    )
    return "\n".join(lines) + "\n"