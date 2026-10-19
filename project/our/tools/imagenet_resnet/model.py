"""Exact integer ResNet and the graph consumed by secure_driver.

Floating point is used to load/fold/encode constants and encode the input.
Every graph operator consumes and returns signed int64 integer codes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet50_Weights, ResNet152_Weights, resnet50, resnet152,
)

PRECISION_DIR = Path(__file__).resolve().parents[1] / "precision"
if str(PRECISION_DIR) not in sys.path:
    sys.path.insert(0, str(PRECISION_DIR))
from uniform_integer_gemm import CachedIntegerGemmWeights, conv2d_gemm, integer_im2col, linear_gemm
from uniform_fixed_point import _native_ops


DEFAULT_FRAC = {8: 0, 16: 4, 20: 6, 24: 8, 28: 10, 32: 12}
ARITHMETIC = "dot_mod_2^bits_signed_floor_shift_then_Q_bias_wrap"
MODEL_SPECS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
}


def wrap(codes, bits):
    residues = torch.bitwise_and(codes.to(torch.int64), (1 << bits) - 1)
    return torch.where(residues >= (1 << (bits - 1)), residues - (1 << bits), residues)


def encode(values, bits, frac):
    """Entry/constant encoding: IEEE round-to-nearest, ties-to-even, then wrap."""
    values = torch.round(values.detach().to(torch.float64) * (1 << frac))
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite input or model constant")
    return wrap(torch.remainder(values, 1 << bits).to(torch.int64), bits)


def fold_conv_bn(conv, bn):
    if not bn.track_running_stats:
        raise ValueError("BN folding requires frozen running statistics")
    weight = conv.weight.detach().to(torch.float64)
    mean, variance = bn.running_mean.detach().double(), bn.running_var.detach().double()
    gamma = bn.weight.detach().double() if bn.affine else torch.ones_like(mean)
    beta = bn.bias.detach().double() if bn.affine else torch.zeros_like(mean)
    factor = gamma / torch.sqrt(variance + bn.eps)
    bias = conv.bias.detach().double() if conv.bias is not None else torch.zeros_like(mean)
    return weight * factor[:, None, None, None], beta + (bias - mean) * factor


def load_reference(name, weights_path=None):
    builder, weights = MODEL_SPECS[name]
    model = builder(weights=weights if weights_path is None else None).eval()
    if weights_path is not None:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    source_path = Path(weights_path) if weights_path is not None else (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name)
    model._weights_identity = {"source": "custom_state_dict" if weights_path is not None else "torchvision_IMAGENET1K_V1",
                               "file": source_path.name, "sha256": file_digest(source_path)}
    return model, weights.transforms()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def ring_matmul_cpu(left, right, bits):
    """Exact modular dot using 16-bit limbs; each int64 partial dot is bounded."""
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise ValueError("CPU ring oracle requires CPU integer tensors")
    if left.shape[-1] > ((1 << 63) - 1) // (65535 * 65535):
        raise OverflowError("16-bit limb dot cannot be proven int64-safe")
    mask = (1 << bits) - 1
    lo_a, lo_b = left & 65535, right & 65535
    dot = (lo_a @ lo_b) & mask
    if bits > 16:
        hi_a, hi_b = (left >> 16) & 65535, (right >> 16) & 65535
        cross = ((hi_a @ lo_b) & mask) + ((lo_a @ hi_b) & mask)
        dot = (dot + ((cross & mask) << 16)) & mask
    return wrap(dot, bits)


def finish_dot(dot, bias, bits, frac):
    result = torch.div(wrap(dot, bits), 1 << frac, rounding_mode="floor")
    return wrap(result + bias, bits)


class IntegerParameter(nn.Module):
    def __init__(self, weight, bias, bits, frac):
        super().__init__()
        self.register_buffer("weight", encode(weight, bits, frac))
        self.register_buffer("bias", encode(bias, bits, frac))
        self.packed = CachedIntegerGemmWeights(self.weight)


class IntegerResNet(nn.Module):
    def __init__(self, reference, name="resnet50", bits=32, frac=None, batch=1,
                 collect_overflow=False):
        super().__init__()
        if bits not in DEFAULT_FRAC:
            raise ValueError("bits must be one of 8,16,20,24,28,32")
        self.bits = int(bits)
        self.frac = DEFAULT_FRAC[bits] if frac is None else int(frac)
        if not 0 <= self.frac < bits:
            raise ValueError("require 0 <= frac < bits")
        if batch < 1:
            raise ValueError("batch must be positive")
        self.name, self.batch = name, int(batch)
        self.overflow = None
        if collect_overflow:
            from width_stats import RingOverflowStats
            self.overflow = RingOverflowStats(bits)
        self.parameters_by_node = nn.ModuleDict()
        self.graph = {"model": name, "batch": self.batch, "bits": bits,
                      "frac": self.frac, "input": "input", "output": "fc",
                      "tensors": {"input": [batch, 3, 224, 224]}, "nodes": [],
                      "arithmetic": ARITHMETIC,
                      "encoding": "torch.round_ties_even_then_wrap",
                      "weights_identity": getattr(reference, "_weights_identity", {"source": "supplied_reference"}),
                      "bn_folding": "float64_before_quantization"}
        self._offset = 0
        current = self._conv("conv1", "input", reference.conv1, reference.bn1)
        current = self._node("relu", "relu", [current])
        current = self._node("maxpool", "maxpool", [current], kernel=3, stride=2, padding=1)
        for stage in range(1, 5):
            for index, block in enumerate(getattr(reference, "layer%d" % stage)):
                prefix = "layer%d.%d" % (stage, index)
                identity = current
                result = current
                for layer in range(1, 4):
                    result = self._conv(prefix + ".conv%d" % layer, result,
                                        getattr(block, "conv%d" % layer), getattr(block, "bn%d" % layer))
                    if layer < 3:
                        result = self._node(prefix + ".relu%d" % layer, "relu", [result])
                if block.downsample is not None:
                    identity = self._conv(prefix + ".downsample.0", current,
                                          block.downsample[0], block.downsample[1])
                result = self._node(prefix + ".add", "add", [result, identity])
                current = self._node(prefix + ".relu3", "relu", [result])
        current = self._node("avgpool", "avgpool", [current], kernel=7, stride=1, padding=0)
        self._linear("fc", current, reference.fc)

    @staticmethod
    def _key(name):
        return name.replace(".", "__")

    def release_weight_codes(self):
        """Discard redundant int64 weights after export, before CUDA transfer.

        Packed digits retain exact CUDA weights. CPU execution and parameter
        export require an instance whose original weight codes are retained.
        """
        released = 0
        for parameter in self.parameters_by_node.values():
            if parameter.weight is not None:
                released += parameter.weight.numel() * parameter.weight.element_size()
                parameter.weight = None
        return released

    def _node(self, name, op, inputs, shape=None, **attributes):
        source = self.graph["tensors"][inputs[0]]
        if shape is None:
            shape = list(source)
            if op == "maxpool":
                shape[-2:] = [(v + 2 * attributes["padding"] - attributes["kernel"]) // attributes["stride"] + 1
                              for v in source[-2:]]
            elif op == "avgpool":
                shape[-2:] = [1, 1]
            elif op == "add" and source != self.graph["tensors"][inputs[1]]:
                raise ValueError("residual tensor shapes disagree")
        node = dict(name=name, op=op, inputs=inputs, output=name, shape=shape)
        node.update(attributes)
        self.graph["nodes"].append(node)
        self.graph["tensors"][name] = shape
        return name

    def _parameter(self, name, weight, bias):
        if self.overflow is not None:
            self.overflow.observe_encoding("parameter_encoding", weight, self.frac)
            self.overflow.observe_encoding("parameter_encoding", bias, self.frac)
        param = IntegerParameter(weight, bias, self.bits, self.frac)
        self.parameters_by_node[self._key(name)] = param
        attributes = dict(weight_offset=self._offset, weight_shape=list(weight.shape))
        self._offset += weight.numel()
        attributes.update(bias_offset=self._offset, bias_count=bias.numel())
        self._offset += bias.numel()
        return attributes

    def _conv(self, name, source, conv, bn):
        if conv.groups != 1 or _pair(conv.dilation) != (1, 1) or conv.padding_mode != "zeros":
            raise ValueError("export supports standard torchvision ResNet convolutions")
        weight, bias = fold_conv_bn(conv, bn)
        kernel, stride, padding = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        shape = self.graph["tensors"][source]
        output_shape = [shape[0], weight.shape[0]] + [
            (v + 2 * padding - kernel) // stride + 1 for v in shape[-2:]]
        return self._node(name, "conv2d", [source], output_shape,
                          kernel=kernel, stride=stride, padding=padding,
                          **self._parameter(name, weight, bias))

    def _linear(self, name, source, linear):
        bias = linear.bias.detach().double() if linear.bias is not None else torch.zeros(linear.out_features)
        return self._node(name, "linear", [source], [self.batch, linear.out_features],
                          **self._parameter(name, linear.weight.detach().double(), bias))

    def _dot(self, node, value):
        parameter = self.parameters_by_node[self._key(node["name"])]
        empty = parameter.bias.new_empty(0)
        stats = getattr(self, "overflow", None)
        combine = _native_ops().combine_limb_dots if value.is_cuda else None
        if stats is not None and value.is_cuda:
            combine = stats.wrap_combiner(combine)
        if value.is_cuda:
            if node["op"] == "conv2d":
                result, _ = conv2d_gemm(value, parameter.packed, empty,
                    (node["stride"],) * 2, (node["padding"],) * 2, (1, 1),
                    self.bits, self.frac, self.bits, combine,
                    arithmetic_mode="ring", overflow_mode="wrap")
                result = result + parameter.bias.view(1, -1, 1, 1)
                if stats is not None:
                    stats.observe("bias_add", result)
                return wrap(result, self.bits)
            result, _ = linear_gemm(value.flatten(1), parameter.packed, empty,
                self.bits, self.frac, self.bits, combine,
                arithmetic_mode="ring", overflow_mode="wrap")
            result = result + parameter.bias
            if stats is not None:
                stats.observe("bias_add", result)
            return wrap(result, self.bits)
        if parameter.weight is None:
            raise RuntimeError("CPU integer execution unavailable: weight codes were released for CUDA")
        if node["op"] == "linear":
            matrix, right = value.flatten(1), parameter.weight.T
            dot = ring_matmul_cpu(matrix, right, self.bits)
            if stats is not None:
                stats.observe_cpu_dot(matrix, right)
                stats.observe("bias_add", torch.div(dot, 1 << self.frac, rounding_mode="floor") + parameter.bias)
            return finish_dot(dot, parameter.bias, self.bits, self.frac)
        matrix, (batch, height, width) = integer_im2col(value, (node["kernel"],) * 2,
            (node["stride"],) * 2, (node["padding"],) * 2, (1, 1))
        dot = ring_matmul_cpu(matrix, parameter.weight.flatten(1).T, self.bits)
        if stats is not None:
            stats.observe_cpu_dot(matrix, parameter.weight.flatten(1).T)
            stats.observe("bias_add", torch.div(dot, 1 << self.frac, rounding_mode="floor") + parameter.bias)
        output = finish_dot(dot, parameter.bias, self.bits, self.frac)
        return output.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def execute_node(self, node, values):
        inputs = [values[key] for key in node["inputs"]]
        value = inputs[0]
        op = node["op"]
        if op in ("conv2d", "linear"):
            return self._dot(node, value)
        if op == "relu":
            return value.clamp_min(0)
        if op == "add":
            result = value + inputs[1]
            if getattr(self, "overflow", None) is not None:
                self.overflow.observe("residual_add", result)
            return wrap(result, self.bits)
        if op == "maxpool":
            p, k, s = node["padding"], node["kernel"], node["stride"]
            padded = F.pad(value, (p, p, p, p), value=-(1 << (self.bits - 1)))
            return padded.unfold(2, k, s).unfold(3, k, s).amax(dim=(-1, -2))
        if op == "avgpool":
            if tuple(value.shape[-2:]) != (7, 7):
                raise ValueError("canonical ImageNet ResNet requires a 7x7 final spatial tensor")
            summed = value.sum(dim=(-1, -2), keepdim=True)
            if getattr(self, "overflow", None) is not None:
                self.overflow.observe("avgpool_sum_before_division", summed)
            summed = wrap(summed, self.bits)
            return torch.div(summed, 49, rounding_mode="floor")
        raise ValueError("unknown integer graph operator: " + op)

    def forward_codes(self, codes, capture=None):
        if codes.dtype != torch.int64 or tuple(codes.shape[1:]) != (3, 224, 224):
            raise ValueError("integer ResNet expects int64 [N,3,224,224] codes")
        values = {"input": wrap(codes, self.bits)}
        remaining = {}
        for node in self.graph["nodes"]:
            for name in node["inputs"]:
                remaining[name] = remaining.get(name, 0) + 1
        if capture is not None:
            capture["input"] = values["input"].detach().clone()
        for node in self.graph["nodes"]:
            output = self.execute_node(node, values)
            values[node["output"]] = output
            if capture is not None:
                capture[node["output"]] = output.detach().clone()
            for name in node["inputs"]:
                remaining[name] -= 1
                if remaining[name] == 0:
                    del values[name]
        return values[self.graph["output"]]

    def forward(self, inputs, capture=None):
        if self.overflow is not None:
            self.overflow.observe_encoding("input_encoding", inputs, self.frac)
        return self.forward_codes(encode(inputs, self.bits, self.frac), capture=capture)

    def export(self, directory):
        if any(parameter.weight is None for parameter in self.parameters_by_node.values()):
            raise RuntimeError("parameter export unavailable: weight codes were released for CUDA")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        graph_path = directory / "graph.json"
        if graph_path.exists() and json.loads(graph_path.read_text()) != self.graph:
            raise ValueError("export directory already contains another model/configuration")
        parameter_path = directory / "parameters.bin"
        with parameter_path.open("wb") as stream:
            for node in self.graph["nodes"]:
                if node["op"] not in ("conv2d", "linear"):
                    continue
                parameter = self.parameters_by_node[self._key(node["name"])]
                for tensor in (parameter.weight, parameter.bias):
                    residues = (tensor.detach().cpu() & ((1 << self.bits) - 1)).numpy()
                    stream.write(residues.astype(np.dtype("<u8"), copy=False).tobytes(order="C"))
        if parameter_path.stat().st_size != self._offset * 8:
            raise RuntimeError("exported parameter offsets disagree with binary length")
        graph_path.write_text(json.dumps(self.graph, indent=2) + "\n", encoding="utf-8")
        return graph_path, parameter_path

    @torch.inference_mode()
    def export_audit(self, directory, codes):
        """Export one actual integer input and every graph boundary for auditing."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if codes.shape[0] != self.batch:
            raise ValueError("audit batch must match exported graph batch")
        capture = {}
        if codes.is_cuda:
            torch.cuda.synchronize(codes.device)
        started = time.perf_counter()
        output = self.forward_codes(codes, capture=capture)
        if codes.is_cuda:
            torch.cuda.synchronize(codes.device)
        elapsed = time.perf_counter() - started
        index = {"model": self.name, "bits": self.bits, "frac": self.frac,
                 "dtype": "uint64LE_ring_residues", "evidence_scope": "diagnostic",
                 "oracle_seconds_with_capture": elapsed,
                 "estimated_50000_oracle_seconds": elapsed * 50000 / self.batch,
                 "tensors": {}}
        for position, (name, tensor) in enumerate(capture.items()):
            filename = "input.bin" if name == "input" else "%04d.bin" % position
            residues = (tensor.cpu() & ((1 << self.bits) - 1)).numpy()
            (directory / filename).write_bytes(residues.astype(np.dtype("<u8"), copy=False).tobytes())
            index["tensors"][name] = {"file": filename, "shape": list(tensor.shape),
                                      "elements": tensor.numel()}
        (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="resnet50")
    parser.add_argument("--bits", type=int, choices=sorted(DEFAULT_FRAC), default=32)
    parser.add_argument("--frac", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--audit-image", type=Path)
    parser.add_argument("--audit-dummy", action="store_true")
    parser.add_argument("--audit-device", default="cuda")
    args = parser.parse_args()
    reference, transform = load_reference(args.model, args.weights_path)
    integer = IntegerResNet(reference, args.model, args.bits, args.frac, args.batch)
    graph_path, parameter_path = integer.export(args.export_dir)
    if args.audit_image is not None and args.audit_dummy:
        parser.error("choose only one audit input source")
    if args.audit_image is not None or args.audit_dummy:
        integer.to(args.audit_device)
        if args.audit_image is not None:
            from PIL import Image
            with Image.open(args.audit_image) as image:
                inputs = transform(image.convert("RGB")).unsqueeze(0).repeat(args.batch, 1, 1, 1)
            codes = encode(inputs.to(args.audit_device), integer.bits, integer.frac)
        else:
            generator = torch.Generator().manual_seed(20261015)
            codes = wrap(torch.randint(-(4 << integer.frac), 4 << integer.frac,
                (args.batch, 3, 224, 224), generator=generator), integer.bits).to(args.audit_device)
        with torch.inference_mode():
            integer.forward_codes(codes)
        integer.export_audit(args.export_dir / "audit", codes)
    print(json.dumps({"graph": str(graph_path), "parameters": str(parameter_path),
                      "nodes": len(integer.graph["nodes"]), "bits": integer.bits, "frac": integer.frac}))


if __name__ == "__main__":
    main()
