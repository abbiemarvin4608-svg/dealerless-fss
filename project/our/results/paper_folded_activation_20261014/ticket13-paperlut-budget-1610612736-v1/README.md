# Budget-only diagnostic

This directory is a fresh ten-query PaperLUT diagnostic with
`generation_budget_bytes=1610612736`. The launch did not fix the three allocator
variables from the frozen final plan, so this run is excluded from the final
matched comparison.

The final two-method comparison is recorded in
[`ticket13-interface-projection-matched`](../ticket13-interface-projection-matched/).
`OUTER_LAUNCH.json` preserves the diagnostic launch identity and its environment
limitation. The original logs, result, and round trace remain unchanged.
