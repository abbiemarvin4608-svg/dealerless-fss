#!/usr/bin/env python3
"""Public plaintext oracle contract for the frozen folded activations."""
import unittest

import numpy as np

import vector_oracle as oracle
import sci_semantics


class FoldedActivationTests(unittest.TestCase):
    def test_native_folded_outputs_and_floor_rounding(self):
        inputs = np.asarray([-32768, -128, -1, 0, 1, 128, 32767])
        np.testing.assert_array_equal(
            oracle.activation(inputs, "sigmoid", activation_mode="folded"),
            [0, 4404, 8192, 8192, 8192, 11980, 16384],
        )
        np.testing.assert_array_equal(
            oracle.activation(inputs, "tanh", activation_mode="folded"),
            [-128, -98, -1, 0, 1, 97, 128],
        )

    def test_modes_preserve_all_pre_activation_tensors(self):
        rng = np.random.default_rng(20261013)
        model = {
            name: rng.integers(-40, 41, size=shape, dtype=np.int64)
            for name, shape in sci_semantics.SHAPES.items()
        }
        model['Bg'] *= 256
        model['Bh'] *= 256
        x = rng.integers(-2048, 2049, size=(3, 32), dtype=np.int64)
        h = rng.integers(-128, 129, size=(3, 100), dtype=np.int64)
        _, baseline = oracle.step(x, h, model, trace=True)
        for mode in oracle.ACTIVATION_MODES:
            actual_h, values = oracle.step(x, h, model, trace=True, activation_mode=mode)
            self.assertEqual(len(values), 15)
            for name in ('tmp19', 'tmp21', 'tmp23', 'tmp25', 'tmp26', 'tmp27', 'tmp29'):
                np.testing.assert_array_equal(values[name], baseline[name])
            for case in range(len(x)):
                expected = sci_semantics.step(
                    x[case].tolist(), h[case].tolist(),
                    {name: tensor.tolist() for name, tensor in model.items()},
                    variant='independent', activation_mode=mode,
                )
                for name in values:
                    np.testing.assert_array_equal(values[name][case], expected[name])
            np.testing.assert_array_equal(actual_h, values['H'])

    def test_full_sequence_head_uses_the_selected_integer_graph(self):
        rng = np.random.default_rng(128)
        model = {
            name: rng.integers(-80, 81, size=shape, dtype=np.int64)
            for name, shape in sci_semantics.SHAPES.items()
        }
        model['Bg'] *= 256
        model['Bh'] *= 256
        encoded = rng.integers(-2048, 2049, size=(1, 99, 32), dtype=np.int64)
        for mode in oracle.ACTIVATION_MODES:
            prediction, logits, traces = oracle.infer(encoded, model, trace=True, activation_mode=mode)
            self.assertEqual(len(traces), 99)
            scalar_model = {name: values.tolist() for name, values in model.items()}
            h = [0] * 100
            for t in range(99):
                expected = sci_semantics.step(
                    encoded[0, t].tolist(), h, scalar_model,
                    variant='independent', activation_mode=mode,
                )
                for name in traces[t]:
                    np.testing.assert_array_equal(traces[t][name][0], expected[name])
                h = expected['H']
            fc = sci_semantics.matmul(h, scalar_model['FC'], 16, 8, 8, 16)
            expected_logits = [
                sci_semantics.signed(value + sci_semantics.trunc0(bias, 3), 8)
                for value, bias in zip(fc, scalar_model['FCbias'][0])
            ]
            np.testing.assert_array_equal(logits[0], expected_logits)
            self.assertEqual(int(prediction[0]), int(np.argmax(expected_logits)))


if __name__ == "__main__":
    unittest.main()
