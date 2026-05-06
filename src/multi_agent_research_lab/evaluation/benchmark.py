"""Benchmark helpers for single-agent vs multi-agent runs."""

import os
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, heuristic quality, citation coverage, and errors."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    final_answer = state.final_answer or ""
    cited_sources = sum(1 for index, _source in enumerate(state.sources, start=1) if f"[S{index}]" in final_answer)
    citation_coverage = cited_sources / len(state.sources) if state.sources else 0.0
    quality_score = _heuristic_quality_score(state, citation_coverage)
    estimated_cost = _estimate_cost_usd(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        source_count=len(state.sources),
        error_count=len(state.errors),
        notes=_metric_notes(state),
    )
    return state, metrics


def _heuristic_quality_score(state: ResearchState, citation_coverage: float) -> float:
    score = 0.0
    if state.final_answer:
        score += 2.0
    if state.sources:
        score += 2.0
    if state.research_notes:
        score += 1.5
    if state.analysis_notes:
        score += 1.5
    score += citation_coverage * 2.0
    if not state.errors:
        score += 1.0
    return round(min(score, 10.0), 1)


def _estimate_cost_usd(state: ResearchState) -> float:
    """Estimate provider cost from explicit cost metadata or token metadata.

    This is a lab estimate, not billing-grade accounting.
    You can override the rates through environment variables:

    LAB_INPUT_USD_PER_1M=0.15
    LAB_OUTPUT_USD_PER_1M=0.60
    """

    explicit_costs = [
        float(result.metadata["cost_usd"])
        for result in state.agent_results
        if "cost_usd" in result.metadata and result.metadata["cost_usd"] is not None
    ]
    if explicit_costs:
        return round(sum(explicit_costs), 6)

    input_rate_per_1m = float(os.getenv("LAB_INPUT_USD_PER_1M", "0.15"))
    output_rate_per_1m = float(os.getenv("LAB_OUTPUT_USD_PER_1M", "0.60"))

    input_tokens = 0
    output_tokens = 0

    for result in state.agent_results:
        input_tokens += int(result.metadata.get("input_tokens") or 0)
        output_tokens += int(result.metadata.get("output_tokens") or 0)

    estimated_cost = (input_tokens / 1_000_000 * input_rate_per_1m) + (
        output_tokens / 1_000_000 * output_rate_per_1m
    )

    return round(estimated_cost, 6)


def _metric_notes(state: ResearchState) -> str:
    token_notes = _token_notes(state)

    if state.errors:
        base = "; ".join(state.errors)
    else:
        base = f"routes={state.route_history}; trace_events={len(state.trace)}"

    if token_notes:
        return f"{base}; {token_notes}"

    return base


def _token_notes(state: ResearchState) -> str:
    input_tokens = 0
    output_tokens = 0
    provider_modes: list[str] = []

    for result in state.agent_results:
        input_tokens += int(result.metadata.get("input_tokens") or 0)
        output_tokens += int(result.metadata.get("output_tokens") or 0)

        provider_mode = result.metadata.get("provider_mode")
        if provider_mode:
            provider_modes.append(str(provider_mode))

    parts = []
    if input_tokens or output_tokens:
        parts.append(f"input_tokens={input_tokens}")
        parts.append(f"output_tokens={output_tokens}")
    if provider_modes:
        parts.append(f"provider_modes={provider_modes}")

    return "; ".join(parts)