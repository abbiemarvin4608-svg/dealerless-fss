# Ticket 08 lightweight read-only review

Reviewed `/tmp/sci-preprocessed-08` audit source and final archived result set. No protocol source was edited, no compilation or protocol benchmark was run, and all negative checks used memory-only read injection. Only this review note was written.

## Final measured evidence

The archive has exactly 8 formal rows and 16 default/nonzero-hidden diagnostic rows. Every row is complete and eligible. All three prepared backends have requested/effective native preprocessing true and both parties' four native qualification flags true; default SCI has these modes/flags false and no per-query native stock.

Independently recomputed all formal per-query phase time samples as aligned maxima over the two parties, then their medians. They agree with ELIGIBILITY.json. Each direct query_total is its own measured interval, rather than a sum of phase medians. Every phase's send/receive balance agrees across parties.

Independently summed the non-overlapping application_io, setup, query_total and teardown counters. All 8 session byte totals agree with final archived measurements. first_query, offline, online and output were not counted again. Every session endpoint phase is recorded once per party; useful native material totals are identical across both parties and every query. Strict input/ready/online/output have zero native and bridge generation deltas for all diagnostic and formal rows.

COMPARISON.json equals regeneration from final ELIGIBILITY.json, and REPORT.zh.md equals final report generation. The Beaver+segmented FSS online speedups over newly preprocessed SCI are 1.86962077698x in a4-sequential and 1.45061742686x in a2-parallel; online communication falls 58.5587814% in both. The report also correctly preserves increased offline and direct query_total costs and the conversion-only backend's lack of latency improvement.

Runtime driver source places prepare, seal_prepare, synchronization and material_ready before begin_input and actual query/hidden reads. The archived collector validates full native OT/triple plans, generation/consumption balances and both epoch guards. The setup/query/output boundaries in archived driver source agree with the report. Exact -O3 compile/link command evidence exists in ticket07_contexts/build/build_commands.json. All current runner/reference source hashes agree with their archived hash lists.

## Failure handling

Memory-only replacement of formal SCI_ours-a2-parallel result status with timeout leaves its status as timeout, measurements null, complete/eligible false and all comparison changes null. Failure report retains the failed row and makes no speedup claim.

## Concrete audit gap discovered

At review time actual commands.json was checked only for taskset affinity and binary prefix. The audit checked declared command/native mode and recorded config, but did not compare actual runtime position arguments or commands.json's actual environment against that contract.

A memory-only injection replaced actual SCI_ours-a2-parallel runtime backend argument with SCI and set actual SDCF_NATIVE_PREPROCESSING=0 / SDCF_BACKEND=SCI. audit_matrix still returned both global and row eligible=true. The fake command record conflicts with its unchanged raw logs; eligibility should reject such evidence rather than qualify it.

Independent inspection confirms all 24 *current* archived actual argument vectors and native/backend/worker/graph/CGGM environment values match their config, so this finding does not invalidate the current numerical result. The implementer was notified to add actual argv/environment checks and a focused rejection regression before finalizing. This note records the reproduced pre-fix gap; a post-fix addendum should state the regression result if repaired.


## Post-review fix

The audit now compares both parties' complete actual argv, all explicitly saved
child environment updates, and the recorded configuration against the declared
runner flags. Frozen fixture hashes are also read and verified. A public-artifact
negative test copies only commands.json into a temporary overlay, independently
changes backend argv or disables native preprocessing in one environment, and
requires the row and entire report to reject, with null row costs and null
comparison changes. Both subcases failed before the fix and pass after it.
The original 24-row artifact audit remains fully eligible; no raw execution
record, protocol source, binary or measured sample changed.
