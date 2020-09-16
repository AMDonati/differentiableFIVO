import numpy as np
import os
import argparse


def load_npz(path):
    x = np.load(path, allow_pickle=True)
    samples = x[()]['samples']
    prefix = x[()]['prefixes']
    return samples, prefix

def stack_samples(data_path, seq_len=24):
    list_samples = []
    for i in range(seq_len):
        path = os.path.join(data_path, "samples_preflen{}_sufflen1.npz".format(i))
        sample, prefix = load_npz(path)
        list_samples.append(sample)
    samples = np.stack(list_samples, axis=0)
    # save samples and prefix
    np.save(os.path.join(data_path, "samples_unistep.npy"), samples)
    np.save(os.path.join(data_path, "prefix.npy"), prefix)

    return samples, prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, default='output/fivo/synthetic_1')
    args = parser.parse_args()
    samples, prefix = stack_samples(args.data_path)