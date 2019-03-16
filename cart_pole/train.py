"""Actor-Critic Agent for Cart Pole."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
import gym


parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type=int, default=1000,
                    help="Number of episodes.")


def _random_agent_play(num_episodes):
    env = gym.make("CartPole-v1")

    for i in range(num_episodes):
        observation = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)


if __name__ == "__main__":
    config = parser.parse_args()
    _random_agent_play(config.num_episodes)
