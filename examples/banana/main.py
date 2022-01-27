import argparse
import os
import pickle
import time

import numpy as np
import tqdm

import iliad
from odyssey.banana import Banana, generate_data

parser = argparse.ArgumentParser(description='Convergence effects in Riemannian manifold HMC')
parser.add_argument('--thresh', type=float, default=1e-6, help='Convergence threshold')
parser.add_argument('--max-iters', type=int, default=1000, help='Maximum number of fixed point iterations')
parser.add_argument('--step-size', type=float, default=0.03, help='Integration step-size')
parser.add_argument('--num-steps', type=int, default=20, help='Number of integration steps')
parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to generate')
parser.add_argument('--check-prob', type=float, default=0.001, help='Probability of checking reversibility and volume preservation')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

np.random.seed(args.seed)

sigma_y = 2.0
sigma_theta = 2.0
theta, y = generate_data(0.5, sigma_y, sigma_theta, 100)
distr = Banana(y, sigma_y, sigma_theta)

def experiment(
        step_size,
        num_steps,
        proposal,
        num_samples,
        check_prob
):
    qo = np.zeros(2)
    sampler = iliad.sample(
        qo,
        step_size,
        num_steps,
        proposal,
        check_prob=check_prob,
    )
    samples = np.zeros([num_samples, len(qo)])
    props = np.zeros([num_samples, len(qo)])
    pbar = tqdm.tqdm(total=num_samples, position=0, leave=True)
    elapsed = 0.0
    for i in range(num_samples):
        s = next(sampler)
        elapsed += s.elapsed
        samples[i] = s.sample
        props[i] = s.proposal
        d = proposal.info.asdict()
        pbar.set_postfix(d)
        pbar.update(1)
    return samples, elapsed

def main():
    proposal =  iliad.proposals.RiemannianLeapfrogProposal(
        distr,
        args.thresh,
        args.max_iters,
    )
    samples, elapsed = experiment(
        args.step_size,
        args.num_steps,
        proposal,
        args.num_samples,
        args.check_prob
    )
    metrics = iliad.summarize(samples, elapsed=elapsed)

if __name__ == '__main__':
    main()

