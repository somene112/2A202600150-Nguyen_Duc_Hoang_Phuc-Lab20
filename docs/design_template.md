# Design Template - Multi-Agent Research System

## Problem

Xây dựng một research assistant nhận câu hỏi nghiên cứu, tìm nguồn, phân tích bằng chứng và viết câu trả lời cuối cùng có citation. Hệ thống phải chạy được trong lab mà không bắt buộc có API key, đồng thời vẫn dễ mở rộng sang OpenAI/Tavily/LangGraph.

## Why multi-agent?

Single-agent có thể nhanh hơn, nhưng dễ bị “một prompt làm tất cả”: khó biết nguồn nào được lấy, phân tích trung gian ra sao, và lỗi xảy ra ở bước nào. Multi-agent phù hợp hơn cho task nghiên cứu vì có thể tách trách nhiệm thành routing, research, analysis và writing. Nhờ shared state và trace, nhóm có thể debug từng bước và benchmark với baseline.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn bước tiếp theo và dừng workflow đúng lúc | `ResearchState` hiện tại | route: `researcher`, `analyst`, `writer`, `done` | Loop vô hạn, route sai, dừng quá sớm |
| Researcher | Tìm nguồn và ghi research notes | query, `max_sources` | `sources`, `research_notes` | Nguồn yếu, không có source, search provider lỗi |
| Analyst | Rút ra claims, đánh giá evidence, nêu risk | `sources`, `research_notes` | `analysis_notes` | Phân tích thiếu evidence, không flag nguồn yếu |
| Writer | Tổng hợp final answer có citation | `analysis_notes`, `sources` | `final_answer` | Hallucination, thiếu citation, trả lời lan man |
| Critic (optional) | Kiểm tra citation coverage sau khi viết | `final_answer`, `sources` | finding + lỗi nếu thiếu citation | Check quá đơn giản, false positive |

## Shared state

- `request`: lưu query, audience, số nguồn tối đa.
- `iteration`: chống loop vô hạn.
- `route_history`: giải thích đường đi của workflow.
- `sources`: danh sách nguồn để citation.
- `research_notes`: ghi chú từ Researcher.
- `analysis_notes`: insight từ Analyst.
- `final_answer`: câu trả lời cuối cùng.
- `agent_results`: log output từng agent.
- `trace`: event-level trace để debug.
- `errors`: lưu lỗi/fallback để report minh bạch.

## Routing policy

```text
start
  -> supervisor
  -> researcher if sources/research_notes missing
  -> supervisor
  -> analyst if analysis_notes missing
  -> supervisor
  -> writer if final_answer missing
  -> supervisor
  -> done
```

Điều kiện dừng: route `done`, quá `MAX_ITERATIONS`, hoặc quá `TIMEOUT_SECONDS`.

## Guardrails

- Max iterations: cấu hình qua `MAX_ITERATIONS`, mặc định 6.
- Timeout: cấu hình qua `TIMEOUT_SECONDS`, mặc định 60 giây.
- Retry/fallback: workflow bắt lỗi từng agent và tạo fallback tối thiểu nếu thiếu output.
- Validation: schema Pydantic cho query, source, metrics; writer luôn cố gắng cite `[S1]`, `[S2]`.
- Trace: mỗi agent ghi trace event và workflow ghi span duration.

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| Research GraphRAG state-of-the-art | latency, quality, citation coverage | Multi-agent có trace rõ hơn baseline |
| Compare single-agent and multi-agent workflows | quality, source count, errors | Multi-agent giải thích role tốt hơn |
| Summarize production guardrails for LLM agents | citation coverage, failure mode | Câu trả lời nêu được validation, fallback, monitoring |

Chạy benchmark:

```bash
python -m multi_agent_research_lab.cli benchmark --query "Research GraphRAG state-of-the-art"
```