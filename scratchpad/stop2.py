#!/usr/bin/env python3

import numpy as np


######################################################################################
# helpers
class Env:
    def __init__(self):
        self.goal = 14
        self.ns = 15  # number of states

        self.F = np.zeros(shape=[15, 15], dtype=np.int)  # Feasible
        self.F[0, 1] = 1;
        self.F[0, 5] = 1;
        self.F[1, 0] = 1;
        self.F[2, 3] = 1;
        self.F[3, 2] = 1
        self.F[3, 4] = 1;
        self.F[3, 8] = 1;
        self.F[4, 3] = 1;
        self.F[4, 9] = 1;
        self.F[5, 0] = 1
        self.F[5, 6] = 1;
        self.F[5, 10] = 1;
        self.F[6, 5] = 1;
        # self.F[6, 7] = 1; # hole
        # self.F[7, 6] = 1; # hole
        self.F[7, 8] = 1;
        self.F[7, 12] = 1
        self.F[8, 3] = 1;
        self.F[8, 7] = 1;
        self.F[9, 4] = 1;
        self.F[9, 14] = 1;
        self.F[10, 5] = 1
        self.F[10, 11] = 1;
        self.F[11, 10] = 1;
        self.F[11, 12] = 1;
        self.F[12, 7] = 1;
        self.F[12, 11] = 1;
        self.F[12, 13] = 1;
        self.F[13, 12] = 1;
        self.F[14, 14] = 1
        print("F", self.F)

    def GetNextState(self, curr, action):
        if action == 0:
            next = curr - 5
        elif action == 1:
            next = curr + 1
        elif action == 2:
            next = curr + 5
        elif action == 3:
            next = curr - 1
        elif action == 4:
            next = curr
        # assert(next >= 0)
        # print("next", next)

        die = False
        if action == 4:
            reward = 0
            die = True
        elif next < 0 or next >= self.ns or self.F[curr, next] == 0:
            next = curr
            reward = -10
            die = True
        elif next == self.goal:
            reward = 8.5
        else:
            reward = -1

        return next, reward, die

    def get_poss_next_actions(self, s):
        actions = []
        actions.append(0)
        actions.append(1)
        actions.append(2)
        actions.append(3)
        actions.append(4)

        # print("  actions", actions)
        return actions


def get_rnd_next_state(s, env):
    actions = env.get_poss_next_actions(s)

    i = np.random.randint(0, len(actions))
    action = actions[i]
    next_state, reward, die = env.GetNextState(s, action)

    return next_state, action, reward, die


def my_print(Q):
    rows = len(Q);
    cols = len(Q[0])
    print("       up    right  down   left   stop")
    for i in range(rows):
        print("%d " % i, end="")
        if i < 10: print(" ", end="")
        for j in range(cols): print(" %6.2f" % Q[i, j], end="")
        print("")
    print("")


def Walk(start, Q, env):
    curr = start
    i = 0
    totReward = 0
    print(str(curr) + "->", end="")
    while True:
        # print("curr", curr)
        action = np.argmax(Q[curr])
        next, reward, die = env.GetNextState(curr, action)
        totReward += reward

        # print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
        print(str(next) + "->", end="")
        curr = next

        if die: break
        if curr == env.goal: break

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("REWARD", totReward)


######################################################################################
def GetMaxQ(next_s, actions, Q, env):
    # if actions == 4:
    #    return 0

    max_Q = -9999.99
    for j in range(len(actions)):
        nn_a = actions[j]
        nn_s = env.GetNextState(next_s, nn_a)

        q = Q[next_s, nn_a]
        if q > max_Q:
            max_Q = q
    return max_Q


def Trajectory(curr_s, Q, gamma, lrn_rate, env):
    while (True):
        next_s, action, reward, die = get_rnd_next_state(curr_s, env)
        actions = env.get_poss_next_actions(next_s)

        DEBUG = False
        # DEBUG = action == 4
        # DEBUG = curr_s == 0

        max_Q = GetMaxQ(next_s, actions, Q, env)

        if DEBUG:
            print("max_Q", max_Q)
            before = Q[curr_s][action]

        prevQ = ((1 - lrn_rate) * Q[curr_s][action])
        V = lrn_rate * (reward + (gamma * max_Q))
        Q[curr_s][action] = prevQ + V

        if DEBUG:
            after = Q[curr_s][action]
            print("Q", curr_s, reward, before, after)

        if die: break

        curr_s = next_s
        if curr_s == env.goal: break

    if (np.max(Q) > 0):
        score = (np.sum(Q / np.max(Q) * 100))
    else:
        score = (0)

    return score


def Train(Q, gamma, lrn_rate, max_epochs, env):
    scores = []

    for i in range(0, max_epochs):
        curr_s = np.random.randint(0, env.ns)  # random start state
        score = Trajectory(curr_s, Q, gamma, lrn_rate, env)
        scores.append(score)

    return scores


######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    print("Setting up maze in memory")

    # R = np.random.rand(15, 15)  # Rewards
    MOVE_REWARD = 0

    # =============================================================

    Q = np.empty(shape=[15, 5], dtype=np.float)  # Quality
    Q[:] = 0

    print("Analyzing maze with RL Q-learning")
    start = 0;
    gamma = 0.99
    lrn_rate = 0.5
    max_epochs = 10000
    env = Env()

    scores = Train(Q, gamma, lrn_rate, max_epochs, env)
    print("Trained")

    print("The Q matrix is:")
    my_print(Q)

    #
    # plt.plot(scores)
    # plt.show()

    # print("Using Q to go from 0 to goal (14)")
    # Walk(start, goal, Q)

    for start in range(0, env.ns):
        Walk(start, Q, env)

    print("Finished")


if __name__ == "__main__":
    Main()
