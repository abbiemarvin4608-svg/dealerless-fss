#!/usr/bin/env python3
"""Evaluate only the frozen checkpoint on the predeclared official valid split."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reproduction', type=Path, required=True)
    args = parser.parse_args()
    root, repro = args.run.resolve(), args.reproduction.resolve()
    if not (root / 'frozen_quantization.json').is_file():
        raise ValueError('Freeze quantization before final evaluation')
    sys.path.insert(0, str(repro.parent / 'EdgeML-sirnn/tf'))
    from edgeml_tf.graph.rnn import FastGRNNCell
    model = root / 'fp32_model'
    mean, std = [np.load(str(model / (name + '.npy'))) for name in ['mean', 'std']]
    raw = np.load(str(repro / 'data/Google-30-processed/valid.npy'), mmap_mode='r')
    tf.set_random_seed(42)
    inputs = tf.placeholder(tf.float32, [None, 99, 32])
    cell = FastGRNNCell(100, wRank=16, uRank=35)
    output, _ = tf.nn.static_rnn(cell, tf.unstack(inputs, 99, 1), dtype=tf.float32)
    logits = tf.matmul(output[-1], np.load(str(model / 'FC.npy'))) + np.load(str(model / 'FCbias.npy'))
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1, device_count={'GPU': 0})
    result = np.empty((len(raw), 30), dtype=np.float32)
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        for variable, name in zip(cell.getVars(), ['W1', 'W2', 'U1', 'U2', 'Bg', 'Bh', 'zeta', 'nu']):
            session.run(variable.assign(np.load(str(model / (name + '.npy')))))
        for start in range(0, len(raw), 256):
            normalized = ((raw[start:start + 256, 1:] - mean) / std).astype(np.float32)
            result[start:start + 256] = session.run(logits, {inputs: normalized.reshape(-1, 99, 32)})
    np.save(str(root / 'fp32_logits.npy'), result)
    np.save(str(root / 'fp32_predictions.npy'), result.argmax(1))
    np.save(str(root / 'labels.npy'), np.asarray(raw[:, 0], dtype=np.int64))
    print('FP32 complete: %d samples' % len(raw), flush=True)

if __name__ == '__main__':
    main()
