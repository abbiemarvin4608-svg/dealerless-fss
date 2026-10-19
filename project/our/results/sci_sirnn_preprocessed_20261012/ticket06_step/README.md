# Ticket 06: complete shared single-step preprocessing

The public two-party FastGRNN single-step runner was executed at batch 128,
with two fresh queries for each of SCI, SCI_preprocessed, SCI_ours and
SCI_Beaver_segmented. Both default and frozen nonzero hidden states pass.
Each row exposes 15 diagnostic tensors and 172928 integers per query;
all prepared rows match the newly executed default SCI row exactly.

The default-hidden public integration test passed in 41.191 seconds and the
nonzero-hidden test passed in 41.561 seconds. These are correctness runs,
not ticket 08's formal ten-query performance cohorts. SDCF-09 remains deferred.

## Fair material boundaries

All prepared rows execute the same graph callable on independently allocated
zero public planning buffers. Actual model, input, hidden state and temporary
values are not copied into planning. Every NATIVE_PLAN_PURITY record has zero
communication, native/bridge generation delta and PRG-block delta.

SCI_preprocessed uses unchanged native SCI computation. SCI_ours prepares
remaining native multiplication and activation protocols. The combined row
prepares remaining native activation protocols and keeps Beaver/FSS's own
physical offline generation inside its measured offline interval. It does
not retain duplicated native stock for replaced graph calls or for OT consumed
while generating Beaver/FSS material.

Public OT plans include complete COT message-width vectors. Both parties
validate exact call identities, widths and inventories. Standard/correlated
comparison bit-triples are physically completed before material-ready.
MATERIAL_READY precedes every query-input and optional hidden-state read.
All strict input, ready, online and output phase records have zero native
and bridge generation deltas. Input-related OT choice/correlation corrections
and protocol payloads remain measured online. Stocks are consumed once and
balanced separately for each query.

## Resource And Command Evidence

The diagnostic cohort uses one native activation worker, sequential graph
execution, a cGGM limit of four, server CPUs 0,2,4,10 and client CPUs 6,8,12,16.
Each party is limited to 1610612736 bytes of address space. Its shared flags
match the final frozen optimizations: contiguous material, fixed-4 segments,
fused domains, paired extensions, packed bits, online AES, Beaver algebra,
generation tile limit 0, budget 1610612736 and reserve 536870912 bytes.
Offline AES, mapped Beaver kernels, prefetch and offline overlap are disabled.

Each row directory preserves commands.json, config.json, result.json,
parties.json, schedule.json and both protocol logs. The result collector
requires the complete preprocessing qualification for all three prepared rows.
The build directory preserves private SCI overlays, generated driver/library
sources, source snapshots, exact compiler/link commands and runtime manifest.

Runtime binary SHA-256:
2814f8f4fcb35bafbf61c8555153cb208473e5acaf35827bd03d92f869ab6076.
Bridge SHA-256:
b9d753d1df474f23c2a32484dc4bd3e65a242d04478caf5c10518ec07d476f0a.

Run the approved integration interface with:

```sh
flock /tmp/sci-preprocessed-build.lock env \
  SDCF_SHARED_STEP_BINARY=/tmp/sci-preprocessed-06-runtime/sdcf-runtime \
  SDCF_SHARED_STEP_EVIDENCE=/absolute/new-evidence \
  python3 -m unittest test_shared_preprocessed_step
```

Use the segmented_dcf_runtime tools directory as working directory. The fixture
and optional hidden path can be supplied with SDCF_SHARED_STEP_FIXTURE and
SDCF_SHARED_STEP_HIDDEN_FILE.
