import numpy as np
import os
import argparse


def load_npz(path):
    x = np.load(path, allow_pickle=True)
    samples = x[()]['samples']
    prefix = x[()]['prefixes']
    return samples, prefix

def stack_samples(data_path, seq_len=24, split="test"):
    list_samples = []
    for i in range(seq_len):
        path = os.path.join(data_path, "samples_preflen{}_sufflen1_{}.npz".format(i, split))
        sample, prefix = load_npz(path)
        list_samples.append(sample)
    samples = np.stack(list_samples, axis=0)
    # save samples and prefix
    np.save(os.path.join(data_path, "samples_unistep_{}.npy".format(split)), samples)
    np.save(os.path.join(data_path, "prefix_{}.npy".format(split)), prefix)
    return samples, prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, default='output/elbo/synthetic_1_bs_4')
    parser.add_argument("-split", type=str, default="test")
    args = parser.parse_args()
    samples, prefix = stack_samples(args.data_path, split=args.split)