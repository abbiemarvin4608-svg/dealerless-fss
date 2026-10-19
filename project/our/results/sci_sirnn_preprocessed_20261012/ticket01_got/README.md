# Ticket 01: native binary GOT lifecycle

The retained logs exercise actual two-party SCI SplitIKNP random OT generation
before input, followed by online selection correction and masked-message
adaptation. They cover two fresh queries, 9 13-bit messages and 3 7-bit messages
per query, and the original default GOT path as a control. Both modes select
the same correct messages. Twelve logical OTs generate 256 SIMD rows; 244
padding rows are discarded. The prepared receiver sends 3 bytes of packed
choice corrections online and one byte for synchronization per query.

Failure cases reject wrong contracts, unused material, repeat consumption,
unfinished random OT outputs, and extension after preparation during input,
online, or output. Expected failure exit codes are retained as 2, not results.
These are protocol correctness and accounting evidence, not performance rows
or a complete SCI_preprocessed implementation.

Reproduce from the repository root:

```sh
SDCF_GOT_EVIDENCE_DIR=/tmp/ticket01-got-evidence \
  python3 -m unittest discover \
  -s project/our/tools/segmented_dcf_runtime -p test_prepared_got.py
```

The test builds private SCI header overlays and never changes the shared SCI
installation. The compiler input is the committed test_prepared_got.cpp and
the same runtime material lifecycle/measurement headers used by the existing
FastGRNN entry. Build commands in each result refer to the test's temporary
binary; the source and reproducible build procedure are retained in the repo.

A default SCI batch=128 single-step diagnostic was built at
/tmp/sci-preprocessed-native01 and run at
/tmp/sci-preprocessed-native01-diagnostic under the frozen model/input contract.
All 15 tensor rows and 172928 integer values match the frozen native diagnostic.
Native online extension remains allowed in the control and is metered as 298
generation events. SDCF-09 remains deferred.
