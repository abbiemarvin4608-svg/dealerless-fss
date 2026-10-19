# Precomputed Activation Step Comparison

This experiment executes one complete FastGRNN recurrent step from the
Google-30 generated graph. It does not execute the other 98 steps or the
classifier and cannot establish end-to-end Top-1 accuracy.

The commands use the independent ticket-02 model and evaluation records. The
generated graph and converted model header are preserved under ticket-07 and
match its existing prepared contract. The runner rejects historical graph
schemas paired with independent weight narrowing, and `--artifact-contract`
checks the model and graph against that established artifact. The historical
ticket-01 trace contains only one record and is not the Batch=128 input file.

```sh
python3 project/our/tools/google30_hybrid/build.py \
  --out /tmp/fss_step_precomputed \
  --bridge-lib project/our/build/libsegfss_sci_bridge.so \
  --generated project/our/results/ticket07_hybrid_full/sources/sirnn_fixed.cpp
python3 project/our/tools/google30_hybrid/run_step.py \
  --binary /tmp/fss_step_precomputed/sci-step-hook \
  --schema /tmp/fss_step_precomputed/schema.json \
  --output project/our/results/fss_sigmoid_tanh_google30_precomputed/batch1 \
  --model-header project/our/results/ticket07_hybrid_full/sources/model_fixed.h \
  --input-csv project/our/results/ticket02_independent_quality/heldout_input/X.csv \
  --artifact-contract project/our/results/ticket07_hybrid_full/prepared_contract.json \
  --variant independent --nseg 2 --batch 1 --repeats 3 --timeout 900
```

For Batch=128, use a new output directory and add
`--batch 128 --repeats 3 --diagnostic-cases 0 --port 41300 --timeout 1800`.
The measured batch contains the
first timestep from each of the first 128 Google-30 records, with zero initial
hidden states. Repeated queries use the same frozen inputs and fresh one-use
preprocessing. Each backend calls Sigmoid/TanH once on all `100 * batch`
elements; other step operations execute per sample.

SCI setup/base OT and the additional FSS bridge setup are recorded separately.
Every query generates its complete input-independent activation materials
before loading the client's private input and hidden state. The
online interval includes input loading, the remaining SCI operations, FSS
operations, and final synchronization. It contains no diagnostic output
reconstruction. An assertion rejects FSS offline generation during online
execution. Three separate Batch=1 diagnostics retain the original boundary
fixtures and record the SCI-versus-FSS integer differences.
Per-MSB metadata traces and dispatch prints are disabled in measured secure
queries. Their replacement coverage is validated from bridge counters and
the public one-use material schedule.

The native graph's selected 16-bit MSB calls occur inside Sigmoid/TanH. Whole
activation replacement removes those calls and uses the FSS activation's own
sign/clipping BitDec instead. No additional native-MSB material schedule is
generated; the driver rejects unexpected native-MSB hook calls. Extensions
outside the activation continue to use SCI's wrap and B2A protocols.

`result.json` contains both parties' phase measurements, online and offline
medians, activation subtotals including SCI sign extension, the FSS LUT-only
subtotal, and peak process RSS. Communication sums SCI and bridge sent bytes
from both parties exactly once. SCI and bridge round counters are kept
separate; they are protocol counters rather than measured network RTTs.

The FSS activation uses the frozen f=12/q=10 LUT contract and changes SCI's
native EXP/div approximation. A faster step is not sufficient evidence of a
speedup at equal model quality. Rebuild the bridge library before building the
driver because its statistics ABI now includes the LUT subtotal.
