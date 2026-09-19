# Agentic RAG ideas for the PubMed project

Both supplied lessons are useful introductory design references. They support
improving the answer workflow, while the project's no-SQL data architecture
remains intact. The adaptations below are engineering proposals, not measured
accuracy or performance improvements for this corpus.

## What the two lessons contribute

Educative's [From RAG to Agentic RAG](https://www.educative.io/courses/agentic-rag/from-rag-to-agentic-rag)
describes retrieval as a tool that can be invoked within a larger workflow. Its
useful ideas here are breaking complex requests into subquestions and selecting
tools appropriate to the question. Its broad suggestion that a fixed RAG
pipeline cannot handle comparisons should not be read as a universal result:
a well-designed fixed workflow can retrieve both sides and compare them.

Educative's [The Anatomy of an Agent](https://www.educative.io/courses/agentic-rag/the-anatomy-of-an-agent-components-and-core-logic)
separates model, tools, planning and memory. Its memory discussion is relevant
to remembering previously retrieved evidence instead of repeating the same work.
The project's authoritative knowledge remains its versioned source records,
facts and passages; the vector store is one derived retrieval representation.

Both lesson bodies, including the second page's memory section, were readable.
They are course explanations rather than biomedical benchmark evidence. Their
example SQL tools, model/provider choices and free-tier descriptions are not
project requirements. We are not adopting those commercial choices or treating
their model-performance statements as validated comparisons.

## How this maps to existing code

`pipeline/agent.py` already has model-selected tool calls, accumulated
observations, provider selection and a six-step limit. `pipeline/tools.py`
already exposes a named tool registry and structured aggregation functions.
The useful work is to make routing, evidence handling and stopping behavior
more reliable, not to rename the project or introduce an additional framework.

The [ReAct paper](https://arxiv.org/abs/2210.03629) studies combining model
reasoning with actions and feedback from external sources. It provides primary
research support for exploring tool-based sequences, but its results do not
establish biomedical correctness for the current model and collection.

## Proposed adaptations

| Proposal | Concrete behavior in this project | Evaluation |
|---|---|---|
| Question routing | A PMID lookup fetches that record; a count calls a structured tool; an evidence question retrieves passages | Route accuracy and end-to-end time |
| Separate comparison searches | Gather evidence for topic A and topic B with the same requested constraints | Coverage and support for both sides |
| Limited corrective retrieval | When a required aspect has no support, make one targeted additional retrieval attempt, then disclose any remaining gap | Added support versus extra latency and tokens |
| Structured request state | Remember filters, selected records and tool outputs without treating prior generated prose as a source | Fewer duplicate calls; correct follow-up scope |
| Defined stopping rules | Enforce call/time/token limits and detect repeated unproductive calls | Completion rate, timeout rate and supported fallback behavior |

The [Corrective RAG paper](https://arxiv.org/abs/2401.15884) investigates evaluating
retrieved information and choosing corrective actions. This motivates a bounded
evidence check experiment here; it does not mean copying the paper's entire
system. In particular, its web-retrieval route is not part of this project
proposal. The assistant must retain the supplied-corpus scope unless external
sources are explicitly requested and identified.

These checks cannot be based on a model's self-reported confidence alone.
Track which question aspects have supporting passages and verify IDs, source
versions and required filters. A human-reviewed answer set is still necessary.

## Accuracy and speed are separate goals

Additional searches or model calls may help a difficult question, while adding
latency and cost. Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
distinguishes fixed workflows from model-directed agents and recommends adding
complexity only when its value is demonstrated. This December 2024 article now
notes that its tooling landscape has evolved; this review uses its architectural
tradeoff, not its product recommendations.

Our performance proposals are to render deterministic answers directly, reuse
identical tool results, bound retries, and fetch only the evidence needed. A
cross-request cache must include the corpus snapshot and filters, and must not
share private state across users. Independent searches can be tested in parallel,
but a CPU-constrained machine may become slower under extra concurrency.

Example: for a request to compare findings about two interventions in the supplied
literature after a specified year, resolve each intervention, preserve the year
filter, retrieve and read evidence for each, and then produce a cited comparison.
If one side lacks evidence, report that gap. Do not convert absence from the
retrieved passages into a claim that no evidence exists anywhere.

## Decision

Keep the existing eleven implementation changes. Add the limited adaptive
workflow after data correctness and evidence delivery are fixed. Compare it with
the corrected fixed workflow on the same held-out queries and snapshot. Retain
extra steps only where they improve support enough to justify measured cost.

No automatic multi-agent design, new database, fine-tuning, unbounded agent loop,
web browsing, or LLM-based processing of every XML record is required. A planner
cannot compensate for outdated facts, missing passages or incorrect counts.

The durable implementation checklist is [PROJECT_PLAN.md](../PROJECT_PLAN.md).
All proposed additions remain pending implementation and evaluation.
