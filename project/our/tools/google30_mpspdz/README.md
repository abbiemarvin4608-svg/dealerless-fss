# MP-SPDZ Google-30 单时间步工具

`prepare.py` freezes the ticket-01 historical mixed-width graph and creates private fixture inputs. `secret.mpc` implements its linear, LUT, reciprocal, signed truncation, state-update and public-scalar operations as secret `sint` values. `validate_sci.py` builds and runs the independent SCI reference harness; `run.py` runs compiled real two-party semi2k and records resource-bounded timings. `test_*.py` tests only the public parser/oracle seams.

The generated model fixture is deliberately small (single time step, three cases) and does not claim full-network performance. The performance path keeps hidden state secret. Diagnostic output is client-only and exists solely for exact integer comparison.

`vector_oracle.py` is a plaintext integer replay of the independent frozen graph.
Its `--activation-mode` selects `sci` (the preserved default), `sigmoid-only`,
`tanh-only`, or `folded`. The latter three replace only the selected complete
activation with the paper's Q12 correction LUT and keep other integer operations
unchanged. `--diagnostics` writes the existing 15 first-step tensors;
`--hidden-state` accepts a signed-int16 `.npy` initial state for reproducible
nonzero-state replay. All modes still execute 99 steps and the classification
head. `folded_semantics.py` independently freezes the 1024 entries using
80-digit Decimal arithmetic; TanH output rescaling uses signed floor.

`validate_folded_oracle.py` checks all native input encodings and the Q12
boundaries, compares scalar and vector replay, checks batch=128 with zero and
nonzero states, and crosschecks the preserved SCI graph against prior real-SCI
diagnostics. Its three-sample full-sequence smoke is numerical validation,
separate from a full validation-set quality evaluation and secure performance.

`summarize_folded_quality.py` combines complete `sci` and `folded` oracle
results against the frozen 6798-row valid manifest and FP32 predictions. It
records the shared sample order, checks predictions against head logits, and
reports the unchanged 1.0 percentage-point quality threshold separately from
secure correctness and performance. A failed model-quality gate remains a
valid completed evaluation.
