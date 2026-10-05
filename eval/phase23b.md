# Phase 2.3B Result Profiling

## Scope

The profiling tool reads completed D03–D05 Development artifacts. No question
loader, provider client, Run creation, resume, retry, Judge, or production filter
is part of offline characterization.

```powershell
.venv/Scripts/python.exe -m eval.profile_v2 `
  --experiment-dir eval/runs/dev-baseline-v2-attributed `
  --output eval/runs/phase23b-characterization/profile.json
```

## Identity and units

Result IDs identify raw query occurrences. Tool-scoped original URL dedup retains
the first result and explicitly links other occurrences to that representative.
Only selected results enter the processing funnel and replay denominator.
Rank is per-query rank. Provider scores retain their original values without
cross-query calibration.

Exactly one terminal processing outcome belongs to each selected result.
No-raw-content outcomes have no logical model call. Timeout and error outcomes
may have an unchanged raw-content fallback disposition; that disposition is
not an additional outcome. Broken or duplicate result/query/summary/model
identities cause profiling to fail rather than infer a missing link.

Summary objective v2 hashes the normalized actual template, date, and output
schema. Content is separately hashed as the exact summary input. Task topic,
Research brief, and query are absent from the current summary prompt.

Persisted source URLs and Task/Attempt Evidence associations are business units.
Neither unit represents a one-to-one continuation of Tavily results.

## Missingness and derived observations

Complete token sums remain null whenever an included invocation lacks usage.
Known observed subtotals and missing-call counts are separate fields. Retry cost
coverage remains unknown. A skipped result contributes zero calls and known
zero processing cost. Parallel accumulated latency is separate from Research
wall-clock latency.

Duplicate categories overlap. Within-Task and cross-Task scopes can both
apply and cannot be added together. Historical domain signals are recomputed
from saved domain values in an independent selected-arrival view; original
records remain unchanged. Arrival order does not establish summary completion
order or the availability of a reusable prior output.

## Shadow replay

Every replay policy records `local_counterfactual=true` and
`full_agent_causal_estimate=false`. Exact-content policies require the same
objective and actual input hash. Same-URL policies are diagnostics, not safe
deletion claims. Rank, empirical score cutoffs, and domain occurrence caps are
diagnostic sweeps. Raw length remains observation-only.

`local_replay_retention_cost_frontier` contains nondominated retained-result
ratios and known-observed token-avoidance subtotal ratios. Missing token costs
are explicitly excluded from that subtotal axis. This frontier has no quality
axis and does not predict downstream queries, rounds, reports, or billing.

## Current execution audit and stop

The independent report and machine-readable artifacts reside in
`eval/runs/phase23b-characterization/`.

- The single Qwen probe returned HTTP 200 and 31 observed tokens; visible output
  was empty and all 16 output tokens were reasoning.
- D03 and the original D04 retry completed with strict result-summary links.
- An earlier D03 failed at model initialization without a provider invocation.
  Its original journal was overwritten by a later helper; original bytes are
  unavailable and this loss is recorded explicitly.
- An extra D04 was mistakenly created after an incorrect API-unavailability
  inference. The interrupt request did not prevent its completion. Its costs
  and identity are preserved separately and excluded from valid profiles.
- The original D04 completed but retains incorrect historical failure metadata.
  An append-only correction audit records the authoritative completed status.
- D05 was not created after the infrastructure deviations. Only two distinct
  completed v2 questions are valid. Live candidate configurations are empty;
  heuristic thresholds remain undetermined.

The phase is partially completed and stopped, not accepted as a complete
three-question real-data run. No Holdout or new D01/D02 run was executed.
Further paid collection requires a separately authorized stage after execution
identity and finalization acceptance. Existing question, baseline, Judge,
prompt, rubric, and prior Development result bytes remain frozen.
