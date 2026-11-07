# Status update: inspired implementation authorized

The review below records why a faithful DNH-HOPE implementation remains blocked.
The user subsequently authorized **DNH-inspired v1**. It is implemented, tested
and evaluated separately; see [DNH_INSPIRED.md](DNH_INSPIRED.md). Statements below
about not implementing an inspired substitute describe the earlier decision and
are superseded by that authorization. They do not imply access to author code.

# DNH fidelity review and forecasting protocol

## Verified sources (2026-11-06)

Jafari A. A., Ozcinar C., Anbarjafari G. (2026), *Dynamic nested hierarchies:
self-evolving machine learning architectures for lifelong learning*, Frontiers
in Artificial Intelligence 9, DOI 10.3389/frai.2026.1804338, published 14 May 2026.
The full publisher HTML and JATS XML were retrieved successfully after publication
of the environment. See sources/dnh.html, dnh.xml and paper_manifest.json.
XML is necessary: equations were absent from the visible HTML text extraction.
The full text, core method Sections 2–3, experimental setup 5.1 and availability
statement were inspected. No supplementary-material element or supplementary
attachment link was found in the retrieved article/XML. The availability
statement mentions article/supplementary material generically; absence of a link
does not prove that no material exists elsewhere.

The article links Mammoth for baseline implementations, not DNH code. GitHub
repository searches for the exact method name and DNH-HOPE returned zero matches
(see sources/implementation_search.json); this bounded search is not proof that
no code exists. No author-linked official DNH implementation has been verified.
No contact was sent to the authors.

## Compact correspondence and fidelity gate

| Mechanism | Paper locator | What is specified | Remaining implementation choice; current status |
|---|---|---|---|
| Nested optimization and memory | Section 2.2, Eq. (3)–(4) | Time-varying DAG of associative memories; local optimization conditioned on preceding memory/context | Concrete local objective, context tensors and differentiable graph-search procedure unspecified for scalar forecasting; not implemented |
| Meta-objective | Section 3.1, Eq. (8) | Task loss + lambda_s * (level count + sum widths) + lambda_d * Drift(delta_t) | Drift estimator, coefficients and dependency on candidate graph needed; as written an exogenous drift term does not select a graph; not implemented |
| Growth | Sections 2.3.2, 3.2.1 | Meta-loss above tau for k consecutive steps; copy last parameters plus alpha*c*c-transpose; new frequency rho times parent (rho=.5); add edge | k/tau/alpha, context normalization, compatible tensor shapes, max-depth/trigger interactions needed; not implemented |
| Pruning | Section 3.2.2 | Outer-loss gradient norm below epsilon for k steps; low-rank merge into predecessor; bypass edges | Rank, merge operator, output-preserving reparameterization and root protection not specified; not implemented |
| Frequencies | Section 2.2, Eq. (5) | Positive gradient-plus-momentum update to positive real-valued frequencies | Differentiating a discrete update schedule, positivity/bounds and scheduling conversion unspecified; Eq. (5) uses plus sign, not silently changed to descent; not implemented |
| Self-modifying memory | Section 3.3, Eq. (9) | Gated memory update M + sigmoid(W_g*q) elementwise (v*k-transpose - M) | Key/value/query maps, dimensions and integration with nested inner optimization needed; not implemented |
| EAdam and optimizer states | Section 3.4, Eq. (10) | Per-level learning rate eta_base*f_level/f_1 | Moment/step-counter creation, transfer/merge and deletion during topology changes not specified; Section 5.1 also reports AdamW; not implemented |
| Fixed hierarchy and ablations | Section 5.4 ablation discussion | Disable structural mechanisms to isolate contribution | Requires same faithful base implementation and compute budget; not run |

A stacked network or heuristic growing network would omit defining mechanisms:
nested inner problems, graph meta-optimization, memory state and learned update
frequencies. No such surrogate has been silently substituted. No DNH model,
DNH-inspired model, DNH result or growth/pruning verification is present yet.
Reported paper results are claims of the paper, not independently replicated.

## Questions that materially affect method identity

1. Is author reference code available, and which commit should resolve ambiguities?
2. What differentiable estimator/surrogate produces d L_meta / d f through
   discrete update scheduling? What enforces positive bounded frequencies?
3. What exact rank/merge operator and optimizer-moment/step transfer are used on
   pruning and growth? Which k, tau, epsilon, alpha and meta-loss weights apply?
4. User decision: retain the faithful-DNH blocker until author code or
   clarifications are available. Complete baseline work; do not implement an inspired substitute.

Numerical forecasting would additionally need a development-only lag window,
scalar regression head/loss, causal context construction, and a memory reset
policy per customer. No cross-series pooling is authorized for the primary
comparison. All neural seeds and structural ablations remain unrun pending this
method decision. It is not legitimate to infer DNH-vs-ARIMA performance now.

## Historical baseline protocol (superseded for new runs)

The description below records the original AIC-based selector. New runs use
chronological development validation; see [PROTOCOL_REVISION.md](PROTOCOL_REVISION.md).

One-step-ahead chronological prefix development/evaluation; no normalization,
interpolation or winsorization. Seasonal lag 24; MASE denominator from development
only. ARIMA order search is bounded by config and uses converged development AIC;
record warnings/nonconvergence. Across differencing orders, AIC is a pragmatic
selector, not optimality evidence; final studies may add development-only
chronological validation. Selected order fixed, parameters refit only after the
configured count of revealed observations. State updates after every target.
All models receive the same available target history. Expanding-history ARIMA
and lag-based naive models have different retention requirements, not exclusive
observations. Invalid ARIMA forecasts fall back to persistence with an event.
Failed state updates reset ARIMA and retry fitting using revealed history after
the next forecast. Missing evaluation targets remain unscored, not imputed.

Selection is lexicographic by series ID using development availability and length
only, before model performance. This is deterministic, not representative sampling.
Quick/pilot/full use chronological prefixes; they do not span the complete dataset.
A longer seasonal warm-up and later chronological evaluation are sensible explicit
extensions, not silently changed protocol. Scale-normalized macro averages weight
customers equally. No IID uncertainty intervals across adjacent hours are used.
For a later full study consider within-customer paired block bootstrap with block
length sensitivity and report customer variability separately from neural seeds.
