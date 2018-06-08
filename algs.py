"""
This file has implementations of bandit algorithms.

Each algorithm takes as input the history: A list with "wins" and "pulls" tuples for each arm
"""

import math
import random


def mean_reward(x):
    return x[0] / max(1, x[1])

def eps_greedy(history):
    """ epsilon greedy strategy for agent """

    # pick random arm with no history
    not_pulled_arms = [i for i, h in enumerate(history) if h[1] == 0]
    if len(not_pulled_arms) > 0:
        return random.choice(not_pulled_arms)

    # with probability eps select random arm
    if random.random() < 0.05:
        return random.choice(range(len(history)))

    # else pick (one of the) best arm
    best_avg_reward = max([mean_reward(x) for x in history])
    best_arms = [i for i, x in enumerate(history) if mean_reward(x) >= best_avg_reward - 0.00001]
    return random.choice(best_arms)

def ucb1(history):
    """ UCB strategy for agent """

    def upper_bound(x, t, sigma=0.5):
        _, n, p = x
        return math.sqrt(2 * sigma * math.log(t) / n)

    # pick random arm with no history
    not_pulled_arms = [i for i, h in enumerate(history) if h[1] == 0]
    if len(not_pulled_arms) > 0:
        return random.choice(not_pulled_arms)

    # number of arm pulls
    t = sum(x[1] for x in history) + 1

    best_score = max([mean_reward(x) + upper_bound(x, t) + x[2] for x in history])
    best_arms = [i for i, x in enumerate(history)
                 if mean_reward(x) + upper_bound(x, t) + x[2] >= best_score - 0.00001]

    '''
    arms_payoff = []
    for arm in best_arms:
        payoff = history[arm][2]
        arms_payoff.append(payoff)

    best_arms_idx = []
    for i, pay in enumerate(arms_payoff):
        if pay == max(arms_payoff):
            best_arms_idx.append(i)

    n_best_arms = [best_arms[i] for i in best_arms_idx]
    '''

    return random.choice(best_arms)

def ucb2(history):
    """ UCB strategy for agent """

    def upper_bound(x, t, sigma=0.5):
        _, n, p = x
        return math.sqrt(2 * sigma * math.log(t) / n)

    # pick random arm with no history
    not_pulled_arms = [i for i, h in enumerate(history) if h[1] == 0]
    if len(not_pulled_arms) > 0:
        return random.choice(not_pulled_arms)

    # number of arm pulls
    t = sum(x[1] for x in history) + 1

    prob = random.random()

    if prob >= 0.5:
        best_score = max([mean_reward(x) + upper_bound(x, t)  for x in history])
        best_arms = [i for i, x in enumerate(history)
                     if mean_reward(x) + upper_bound(x, t) >= best_score - 0.00001]
    else:
        best_score = max([x[2]  for x in history])
        best_arms = [i for i, x in enumerate(history)
                     if x[2] >= best_score - 0.00001]

    return random.choice(best_arms)

def ucb3(history):
    """ UCB strategy for agent """

    # pick random arm with no history
    not_pulled_arms = [i for i, h in enumerate(history) if h[1] == 0]
    if len(not_pulled_arms) > 0:
        return random.choice(not_pulled_arms)

    # number of arm pulls
    t = sum(x[1] for x in history) + 1


    best_score = max([x[2]  for x in history])
    best_arms = [i for i, x in enumerate(history)
                 if x[2] >= best_score - 0.00001]

    return random.choice(best_arms)

def thompson_sampling(history):
    """ Thompson sampling strategy for agent """
    k = len(history)
    scores = [random.betavariate(x+1, n-x+1) for x, n in history]
    return max(range(k), key=lambda i: scores[i])

