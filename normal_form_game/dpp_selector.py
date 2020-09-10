from numpy.random import RandomState
from dppy.finite_dpps import FiniteDPP
import numpy as np

from scipy.spatial.distance import minkowski as dist
from collections import Counter


def dpp_selector(solver, current_player, epsilon=1e-12, dpp_iters=10000, uniform=False):
    epsilon = 1e-12

    meta_game = solver.get_meta_game()
    meta_probabilities = solver.get_meta_strategies()

    # probabilities_enemy = run_dpp(current_player, meta_game, meta_probabilities, dpp_iters=dpp_iters, p=p)
    probabilities_enemy = run_dpp_option2(current_player, meta_game, meta_probabilities, dpp_iters=dpp_iters, uniform=uniform)
    prob_sum = np.sum(probabilities_enemy)

    # If the rectified probabilities are too low / 0, we play against the
    # non-rectified probabilities.
    if prob_sum <= epsilon:
        if current_player == 0:
            probabilities_enemy = meta_probabilities[1]
        else:
            probabilities_enemy = meta_probabilities[0]
    else:
        probabilities_enemy /= prob_sum

    return probabilities_enemy


def run_dpp_option2(current_player, meta_game, meta_probabilities, dpp_iters=10000, uniform=False):

    rng = RandomState(0)

    meta_game = meta_game[current_player]
    if current_player == 0:
        nb_policies = meta_game.shape[1]
        probabilities_enemy = meta_probabilities[1]
    elif current_player == 1:
        nb_policies = meta_game.shape[0]
        meta_game = meta_game.T
        probabilities_enemy = meta_probabilities[0]
    else:
        raise NotImplementedError

    L = np.zeros((nb_policies, nb_policies))

    for n in range(nb_policies):
            for m in range(n, nb_policies):
                L[n, m] = np.dot(meta_game[:, n], meta_game[:, m]) / (np.linalg.norm(meta_game[:, n])
                                                                      * np.linalg.norm(meta_game[:, m]))
                if n != m:
                    L[m, n] = L[n, m]

    DPP = FiniteDPP(kernel_type='likelihood', projection=False, **{'L': L})


    K = np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0]))
    E_cardinality = np.trace(K)


    # Sample and probabilities
    sample_size = 0
    iters = 0
    while sample_size < E_cardinality:
        sample = np.array(DPP.sample_exact(mode='GS', random_state=rng))
        sample_size = len(sample)
        iters += 1
        if iters > dpp_iters:
            sample = np.arange(len(probabilities_enemy))
            print(" ####################### TOO MANY!! #######################################")
            break

    if not uniform:
        if len(sample) == len(probabilities_enemy):
            return probabilities_enemy
        probabilities_enemy_rectified = np.zeros_like(probabilities_enemy)
        probabilities_enemy_rectified[sample] = probabilities_enemy[sample]
        return probabilities_enemy_rectified
    else:
        if len(sample) == len(probabilities_enemy):
            return (1./len(sample))*np.ones_like(probabilities_enemy)
        probabilities_enemy_rectified = np.zeros_like(probabilities_enemy)
        probabilities_enemy_rectified[sample] = (1./len(sample))*np.ones_like(probabilities_enemy)[sample]
        return probabilities_enemy_rectified



if __name__=="__main__":
    iters = 100000
    p = 2
    meta_game = np.load('dpp_example/meta_game.npy')
    meta_probabilities = np.load('dpp_example/meta_probabilities.npy')

    probabilities = run_dpp_option2(0, meta_game, meta_probabilities)
    print(probabilities)























































