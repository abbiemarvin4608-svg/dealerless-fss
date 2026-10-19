"""Short diagnostic throughput check for choosing validation batch size."""
import argparse
import json
import time
from pathlib import Path

import torch

from model import DEFAULT_FRAC, IntegerResNet, load_reference, wrap


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    reference, _ = load_reference(args.model)
    model = IntegerResNet(reference, args.model, args.bits, args.frac).cuda()
    del reference
    results = dict(model=args.model, bits=args.bits, frac=args.frac,
                   evidence_scope="diagnostic_oracle_throughput_only",
                   device=torch.cuda.get_device_name(), warmup_iterations=1, timed_iterations=2,
                   includes_fp32_or_data_loading=False, batches=[])
    batches = list(args.batches)
    while batches:
        batch = batches.pop(0)
        codes = None
        try:
            generator = torch.Generator().manual_seed(20261015)
            codes = wrap(torch.randint(-(4 << args.frac), 4 << args.frac,
                (batch, 3, 224, 224), generator=generator), args.bits).cuda()
            model.forward_codes(codes)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(2):
                start = time.perf_counter()
                model.forward_codes(codes)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            average = sum(samples) / len(samples)
            row = dict(batch=batch, status="passed", seconds=samples,
                       mean_seconds=average, images_per_second=batch / average,
                       estimated_50000_oracle_seconds=50000 * average / batch,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except torch.cuda.OutOfMemoryError as error:
            row = dict(batch=batch, status="out_of_memory", reason=str(error))
            if batch == 8 and 4 not in [r["batch"] for r in results["batches"]] and 4 not in batches:
                batches.insert(0, 4)
        results["batches"].append(row)
        print(json.dumps(row), flush=True)
        del codes
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet50", "resnet152"), required=True)
    parser.add_argument("--bits", type=int, choices=sorted(DEFAULT_FRAC), default=32)
    parser.add_argument("--frac", type=int)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frac is None:
        args.frac = DEFAULT_FRAC[args.bits]
    run(args)


if __name__ == "__main__":
    main()
