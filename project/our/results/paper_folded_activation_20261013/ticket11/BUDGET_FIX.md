# Historical SCI small-table DPF budget repair

The initial frozen cohort in `formal/` retains nine qualified rows and the
failed `parallel-a2-SCI_Beaver_segmented_DPF_LUT` row. Both parties aborted
query0 preparation with `DPF resident materials exceed reserved budget`.
The old declaration reserved 48766976 bytes, while four prepared slots retained
53899776 bytes. Party VmPeak remained below the 1.5 GiB limit.

The native reservation and planned-material estimate assumed 256 bytes per
material. The current material retains 336 bytes of metadata, so 76800 requests
need 6144000 more bytes than that assumption. The bridge now exposes
`segfss_sci_dpf_material_metadata_bytes()`, using the actual
`DpfPublicLutMaterial::residentBytes()` of an empty material. Native planning
and reservation use the same packed-selector plus actual-metadata calculation.
Public tables, allocator allowance and the final actual-resident check remain
included in the budget contract.

The focused `test_native_dpf_budget` regression calls real native planning and
preparation entry points with network stubs; resident estimates come from the
real packed selection vector and material class. Before the repair all nine
sigmoid/tanh/combined checks failed, including the original preparation error.
After the repair all nine checks passed. Seven adjacent inventory/overlay
checks also passed. The new bridge's exported function independently returns
336 bytes.

The rebuilt bridge and runtime are under
`/tmp/paper-folded-budget-fixed-{core,runtime}`. The default and nonzero hidden
state diagnostics in `diagnostic-budget-fixed/` each passed two fresh batch128
queries: all 15 tensors and 172928 integers per query match the independent
SCI integer oracle. DPF retained bytes are 53899776 per party/query, within the
new 54910976-byte reservation. Joint input-before-material preparation, zero
online generation/loading/wait, causal traffic, four CPUs and 1.5 GiB address
space checks passed. Maximum party VmPeak is 828833792 bytes.

The repaired common build needs its own frozen cohort. `formal/` and its
original plan are retained; none of its samples enter the new cohort's
statistics. Selective folded diagnostics and `requalify.py` document which
unchanged historical qualifications are reused and their original identities.
The new ten-row cohort must complete before any required reverse-order fresh
groups are selected. The original three-group cap and failure stopping rule
are unchanged.
