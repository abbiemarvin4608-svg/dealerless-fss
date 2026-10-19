# Ticket 08 focused validation

The agreed seams are artifact eligibility and the externally reported cost
summary. Real runtime correctness/material tests belong to tickets 01–07;
this ticket qualifies their final diagnostics and reports new formal samples.
The five local tests use independent expected examples, without mocked
protocols or a replay of the native implementation.

- A real archived row whose actual backend argv or native preprocessing
  environment disagrees with the declaration/config cannot qualify. The test
  overlays only commands.json in a temporary directory and observes the full
  public audit/report result, without modifying archived raw records.
- A SCI_ours row without shared preprocessing cannot qualify.
- A formal row lacking its nonzero-hidden diagnostic cannot qualify.
- In a three-query worked example, party online times [1,9,2] and [8,2,3]
  produce aligned maxima [8,9,3] and median 8. Direct query totals have median
  13; adding phase medians would be wrong.
- A complete session counts actual setup + query_total + application_io +
  teardown byte differences once. The worked example expects sends [110,228],
  opposite receives [228,110], and total 338, excluding overlapping first_query
  and offline/online records.

The first three qualification/summary slices were observed red then green
before the final dataset audit. The complete-session slice ran red on
2026-10-12 with this exact command and failure, before adding the field:

```text
python3 -m unittest test_preprocessed_report.PreprocessedReportTests.test_session_communication_counts_setup_queries_and_teardown_once
ERROR: test_session_communication_counts_setup_queries_and_teardown_once
KeyError: 'session_total_bytes'
Ran 1 test in 0.000s
FAILED (errors=1)
```

The actual execution-contract test also ran red before its qualification fix:

```text
FAIL: test_actual_run_contract_disagreement_rejects_artifacts (change=backend_argument)
AssertionError: True is not false
FAIL: test_actual_run_contract_disagreement_rejects_artifacts (change=preprocessing_environment)
AssertionError: True is not false
Ran 1 test in 8.260s
FAILED (failures=2)
```

After the qualification fix, both changed-command subcases reject, the row
measurements and all comparison changes are null, and the unmodified archived
24-row dataset remains eligible. No execution records or samples changed.

All five tests pass; saved output
is in test-green.log. The final archived audit outputs 8 qualified formal rows
and 16 qualified diagnostic rows (audit-green.log). Source/log/measurement
recollection is performed by the frozen archived runner. No online lambda term
is subtracted from old results and no stage medians are added to produce the
direct query or session costs.
