"""Small numerical-contract tests; pretrained/full-data evaluation is separate."""
import json
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torchvision.models import resnet50

from model import IntegerResNet, encode, finish_dot, fold_conv_bn, ring_matmul_cpu, wrap


class IntegerContractTests(unittest.TestCase):
    def test_round_ties_even_and_wrap(self):
        values = torch.tensor([-129., -128., -2.5, -1.5, -.5, .5, 1.5, 2.5, 128.])
        self.assertEqual(encode(values, 8, 0).tolist(), [127, -128, -2, -2, 0, 0, 2, 2, -128])

    def test_cpu_dot_against_big_integer(self):
        generator = torch.Generator().manual_seed(19)
        for bits in (8, 16, 20, 24, 28, 32):
            left = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (3, 19), generator=generator)
            right = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (19, 4), generator=generator)
            actual = ring_matmul_cpu(left, right, bits)
            expected = []
            for row in left.tolist():
                expected.append([sum(a * b for a, b in zip(row, column)) for column in right.T.tolist()])
            expected = [[((v + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1)) for v in row] for row in expected]
            self.assertEqual(actual.tolist(), expected)

    def test_bias_is_added_after_signed_floor(self):
        self.assertEqual(finish_dot(torch.tensor([-3, 127]), torch.tensor([5, 5]), 8, 1).tolist(), [3, 68])
        self.assertEqual(finish_dot(torch.tensor([127]), torch.tensor([100]), 8, 0).tolist(), [-29])

    def test_true_bn_fold(self):
        torch.manual_seed(9)
        conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=True).double()
        bn = torch.nn.BatchNorm2d(4).double().eval()
        bn.running_mean.copy_(torch.randn(4, dtype=torch.float64))
        bn.running_var.copy_(torch.rand(4, dtype=torch.float64) + .1)
        value = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        weight, bias = fold_conv_bn(conv, bn)
        self.assertTrue(torch.allclose(bn(conv(value)), torch.nn.functional.conv2d(value, weight, bias, padding=1), atol=1e-12, rtol=1e-12))

    def test_release_weight_codes_removes_only_redundant_cuda_storage(self):
        from model import IntegerParameter
        generator = torch.Generator().manual_seed(73)
        for bits in (8, 16, 20, 24, 28, 32):
            model = IntegerResNet.__new__(IntegerResNet)
            torch.nn.Module.__init__(model)
            model.bits, model.frac, model.overflow = bits, 0, None
            weight = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1),
                                   (5, 19), generator=generator)
            bias = torch.arange(5, dtype=torch.int64)
            parameter = IntegerParameter(weight.double(), bias.double(), bits, 0)
            model.parameters_by_node = torch.nn.ModuleDict({"test": parameter})
            node = dict(name="test", op="linear")
            model.graph, model._offset = {"nodes": [node]}, weight.numel() + bias.numel()
            value = torch.randint(-100, 100, (2, 19), generator=generator)
            model._dot(node, value)
            packed_before = {name: tensor.clone() for name, tensor in parameter.packed.named_buffers()}
            before_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
            with tempfile.TemporaryDirectory() as directory:
                _, original_path = model.export(Path(directory) / "original")
                self.assertEqual(original_path.stat().st_size, model._offset * 8)
                released = model.release_weight_codes()
                self.assertEqual(released, weight.numel() * weight.element_size())
                self.assertIsNone(parameter.weight)
                after_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
                self.assertEqual(before_bytes - after_bytes, released)
                self.assertTrue(torch.equal(parameter.bias, bias))
                for name, tensor in parameter.packed.named_buffers():
                    self.assertTrue(torch.equal(tensor, packed_before[name]))
                with self.assertRaisesRegex(RuntimeError, "weight codes were released"):
                    model._dot(node, value)
                with self.assertRaisesRegex(RuntimeError, "weight codes were released"):
                    model.export(Path(directory) / "released")
                self.assertFalse((Path(directory) / "released").exists())
                self.assertEqual(model.release_weight_codes(), 0)

    def test_graph_export_and_non_dot_contract(self):
        model = IntegerResNet(resnet50(weights=None).eval(), bits=8, frac=0)
        self.assertEqual(sum(node["op"] == "conv2d" for node in model.graph["nodes"]), 53)
        self.assertFalse(any("bn" in node["op"] for node in model.graph["nodes"]))
        self.assertEqual(model.graph["tensors"]["fc"], [1, 1000])
        pooled = model.execute_node(dict(op="avgpool", inputs=["x"]), {"x": torch.full((1, 1, 7, 7), 127, dtype=torch.int64)})
        self.assertEqual(pooled.item(), 1)
        with tempfile.TemporaryDirectory() as directory:
            graph_path, params_path = model.export(directory)
            graph = json.loads(graph_path.read_text())
            params = np.memmap(params_path, mode="r", dtype="<u8")
            first = graph["nodes"][0]
            expected = model.parameters_by_node["conv1"].weight.flatten().numpy() & 255
            np.testing.assert_array_equal(params[:len(expected)], expected)
            self.assertEqual(params_path.stat().st_size, model._offset * 8)
            graph["model"] = "another_model"
            graph_path.write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, "another model"):
                model.export(directory)

    def test_accuracy_status_and_formal_sample_gate(self):
        from evaluate import _initial_state, accuracy_summary
        identity = dict(model="resnet50", bits=32, frac=12, requested_samples=50000, diagnostic=False)
        state = _initial_state(identity)
        state["processed"] = 49999
        self.assertEqual(accuracy_summary(state)["evidence_scope"], "diagnostic")
        state["processed"], state["status"] = 50000, "complete"
        self.assertEqual(accuracy_summary(state)["evidence_scope"], "formal")
        state["status"], state["reason"] = "failed", "test interruption"
        self.assertEqual(accuracy_summary(state)["status"], "failed")
        self.assertEqual(accuracy_summary(state)["reason"], "test interruption")
        self.assertEqual(accuracy_summary(state)["evidence_scope"], "diagnostic")
        state["identity"]["diagnostic"] = True
        self.assertEqual(accuracy_summary(state)["evidence_scope"], "diagnostic")

    def test_evaluator_resumes_completed_batches_after_failure(self):
        import evaluate as evaluator

        class Dataset:
            classes = ["class0", "class1"]
            class_to_idx = {"class0": 0, "class1": 1}
            samples = [("unit_test_image_%d" % i, 1) for i in range(4)]

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return torch.tensor([index], dtype=torch.float32), 1

        class Reference(torch.nn.Module):
            _weights_identity = {"source": "unit_test", "sha256": "test_only"}

            def forward(self, inputs):
                output = torch.zeros(len(inputs), 10)
                output[:, 1] = 2
                return output

        class Integer(Reference):
            fail_on_second = True

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.calls = 0

            def export(self, directory):
                directory.mkdir(parents=True, exist_ok=True)
                graph, params = directory / "graph.json", directory / "parameters.bin"
                graph.write_text('{"unit_test":true}')
                params.write_bytes(b"unit_test_only")
                return graph, params

            def forward(self, inputs):
                self.calls += 1
                if self.calls == 2 and self.fail_on_second:
                    raise RuntimeError("unit test interruption")
                return super().forward(inputs).to(torch.int64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(device="cpu", model="resnet50", weights_path=None,
                val_dir=root / "val", limit=4, bits=16, frac=4, output_dir=root / "run",
                batch_size=2, workers=0, checkpoint_every=1, restart=False)
            with mock.patch.object(evaluator, "ImageFolder", return_value=Dataset()), \
                    mock.patch.object(evaluator, "load_reference", side_effect=lambda *a: (Reference(), None)), \
                    mock.patch.object(evaluator, "IntegerResNet", Integer), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "unit test interruption"):
                    evaluator.evaluate(args)
                failed = json.loads((args.output_dir / "accuracy.json").read_text())
                self.assertEqual((failed["status"], failed["samples"]), ("failed", 2))
                Integer.fail_on_second = False
                evaluator.evaluate(args)
            checkpoint = torch.load(args.output_dir / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["processed"], 4)
            self.assertEqual(checkpoint["integer_top1_correct"], 4)
            self.assertEqual(len(checkpoint["integer_predictions"]), 4)
            result = json.loads((args.output_dir / "accuracy.json").read_text())
            self.assertEqual((result["status"], result["evidence_scope"]), ("complete", "diagnostic"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_limb_conv_matches_literal_cpu_at_all_widths(self):
        from model import IntegerParameter
        model = IntegerResNet.__new__(IntegerResNet)
        torch.nn.Module.__init__(model)
        generator = torch.Generator().manual_seed(31)
        for bits, frac in ((8, 0), (16, 4), (20, 6), (24, 8), (28, 10), (32, 12)):
            model.bits, model.frac = bits, frac
            weight = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (5, 3, 3, 3), generator=generator)
            bias = torch.randint(-100, 100, (5,), generator=generator)
            parameter = IntegerParameter(weight.double() / (1 << frac), bias.double() / (1 << frac), bits, frac)
            model.parameters_by_node = torch.nn.ModuleDict({"test": parameter})
            value = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (2, 3, 5, 5), generator=generator)
            node = dict(name="test", op="conv2d", stride=1, padding=1, kernel=3)
            expected = model._dot(node, value)
            model.release_weight_codes()
            model.cuda()
            actual = model._dot(node, value.cuda()).cpu()
            self.assertTrue(torch.equal(actual, expected), "CPU/CUDA mismatch for bits=%d" % bits)


if __name__ == "__main__":
    unittest.main()
