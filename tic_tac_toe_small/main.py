import time
import os
import multiprocessing as mp
from collections import namedtuple

# from absl import app
# from absl import flags
import numpy as np
import matplotlib
import argparse

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
from scipy import stats

import generalized_psro
import optimization_oracle
# import pyspiel
from tictactoe import TicTacToe

Result = namedtuple('Result', [
    'meta_games',
    'meta_probabilities',
    'rpps',
    'l_cards_p1',
    'l_cards_p2',
    'lambda_weights',
])


# ------------------------------------------------ Parameters ---------------------------------------------------- #
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--nb_iters', type=int, default=4, help='Number of PSRO iterations')
parser.add_argument('--results_path', type=str, default='results/', help='Save results file path')
prms = vars(parser.parse_args())


def plot_error(data, label=''):
    avg = np.mean(np.array(data), axis=0)
    error = stats.sem(np.array(data))
    plt.plot(avg, label=label)
    plt.fill_between([i for i in range(avg.shape[0])], avg - error, avg + error, alpha=0.3)

def run_psro(params, seed=0):
    game_name = params['game_name']
    number_players = params['number_players']
    sims_per_entry = params['sims_per_entry']
    rectify_training = params['rectify_training']
    training_strategy_selector = params['training_strategy_selector']
    meta_strategy_method = params['meta_strategy_method']
    number_policies_sampled = params['number_policies_sampled']
    number_episodes_sampled = params['number_episodes_sampled']
    rnr_iterations = params['rnr_iterations']
    lambda_weight = params['lambda_weight']

    np.random.seed(seed)

    meta_games = []
    meta_probabilities = []
    rpps = []
    l_cards_p1 = []
    l_cards_p2 = []
    lambda_weights = []

    game = TicTacToe()

    oracle = optimization_oracle.EvolutionaryStrategyOracle(
        number_policies_sampled=number_policies_sampled,
        number_episodes_sampled=number_episodes_sampled,
        n_evolution_tests=50,
        nb_players=number_players)

    g_psro_solver = generalized_psro.GenPSROSolver(game,
                                                   oracle,
                                                   sims_per_entry=sims_per_entry,
                                                   rectify_training=rectify_training,
                                                   training_strategy_selector=training_strategy_selector,
                                                   meta_strategy_method=meta_strategy_method,
                                                   lambda_weight=lambda_weight
                                                   )
    for i in range(rnr_iterations):
        start = time.time()
        g_psro_solver.set_lambda_weight([lambda_weight[0], 1.] if (i>200 and lambda_weight[0]<0.99) else [1., 1.])
        g_psro_solver.iteration()

        meta_games.append(g_psro_solver.get_meta_game)
        meta_probabilities.append(g_psro_solver.get_and_update_meta_strategies())
        rpps.append(meta_probabilities[-1][0] @ meta_games[-1][0] @ meta_probabilities[-1][1])
        l_cards_p1.append(oracle.l_card[0])
        l_cards_p2.append(oracle.l_card[1])
        lambda_weights.append(g_psro_solver._lambda_weight)

        print('Process' + str(os.getpid())+' Iter ' + str(i) + ' in ' + str(time.time() - start) + 's RPP:' + str(rpps[-1]))


    return meta_games, meta_probabilities, rpps, l_cards_p1, l_cards_p2, lambda_weights


def main2(params, num_exps=2, filename='file'):

    # DPP vs PSRO
    params['lambda_weight'] = [0.85, 1.]
    rpps_dpp = []
    l_cards_p1_dpp = []
    l_cards_p2_dpp = []
    meta_games_dpp = []
    meta_probabilities_dpp = []
    for i in range(num_exps):
        meta_games, meta_probabilities, rpps, l_cards_p1, l_cards_p2, lambda_weights = run_psro(params)
        rpps_dpp.append(rpps)
        l_cards_p1_dpp.append(l_cards_p1)
        l_cards_p2_dpp.append(l_cards_p2)
        meta_games_dpp.append(meta_games)
        meta_probabilities_dpp.append(meta_probabilities)

    pickle.dump({
                 'rpps': rpps_dpp,
                 'l_cards_p1': l_cards_p1_dpp,
                 'l_cards_p2': l_cards_p2_dpp,
                 'meta_games': meta_games_dpp,
                 'meta_probabilities': meta_probabilities_dpp,
                 'lambda_weights': lambda_weights,
                 },
                open(os.path.join(prms['results_path'], filename+'_dpp.p'), 'wb'))

    # PSRO vs PSRO
    params['lambda_weight'] = [1., 1.]
    rpps_orig = []
    l_cards_p1_orig = []
    l_cards_p2_orig = []
    meta_games_orig = []
    meta_probabilities_orig = []
    for i in range(num_exps):
        meta_games, meta_probabilities, rpps, l_cards_p1, l_cards_p2, lambda_weights = run_psro(params)
        rpps_orig.append(rpps)
        l_cards_p1_orig.append(l_cards_p1)
        l_cards_p2_orig.append(l_cards_p2)
        meta_games_orig.append(meta_games)
        meta_probabilities_orig.append(meta_probabilities)

    pickle.dump({
                 'rpps': rpps_orig,
                 'l_cards_p1': l_cards_p1_orig,
                 'l_cards_p2': l_cards_p2_orig,
                 'meta_games': meta_games_orig,
                 'meta_probabilities': meta_probabilities_orig,
                 'lambda_weights': lambda_weights,
                 },
                open(os.path.join(prms['results_path'], filename+'_orig.p'), 'wb'))

    plt.figure()
    plot_error(rpps_dpp, label='dpp VS orig')
    plot_error(rpps_orig, label='orig VS orig')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'rpps' + time_string +'.pdf'))

    plt.figure()
    plot_error(l_cards_p1_dpp, label='Diversity p1')
    plot_error(l_cards_p1_orig, label='Diversity orig p1')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'div1' + time_string +'.pdf'))
    plt.figure()
    plot_error(l_cards_p2_dpp, label='Diversity p2')
    plot_error(l_cards_p2_orig, label='Diversity orig p2')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'div2' + time_string +'.pdf'))

    # plt.show()


def main(params_seed):

    params, seed = params_seed
    print('Starting process: ' + str(os.getpid()))
    
    # DPP vs PSRO
    params['lambda_weight'] = [0.7, 1.]
    meta_games, meta_probabilities, rpps, l_cards_p1, l_cards_p2, lambda_weights = run_psro(params, seed=seed)
    result_dpp = Result(
        meta_games=meta_games,
        meta_probabilities=meta_probabilities,
        rpps=rpps,
        l_cards_p1=l_cards_p1,
        l_cards_p2=l_cards_p2,
        lambda_weights=lambda_weights,
    )

    # PSRO vs PSRO
    params['lambda_weight'] = [1., 1.]
    meta_games, meta_probabilities, rpps, l_cards_p1, l_cards_p2, lambda_weights = run_psro(params, seed=seed)
    result_orig = Result(
        meta_games=meta_games,
        meta_probabilities=meta_probabilities,
        rpps=rpps,
        l_cards_p1=l_cards_p1,
        l_cards_p2=l_cards_p2,
        lambda_weights=lambda_weights,
    )

    return (result_dpp, result_orig)



if __name__ == "__main__":
    params = {
        'game_name': 'tic_tac_toe',
        'number_players': 2,

        # PSRO
        'sims_per_entry': 5,  # Number of simulations to run to estimate each element of the game outcome matrix.
        'rectify_training': False,  # A boolean, specifying whether to train only against opponents we beat (True).
        'training_strategy_selector': 'probabilistic',
                                        # How to select the strategies to start training from
                                        #      String value can be:
                                        #        - "probabilistic_deterministic": selects the first
                                        #          policies with highest selection probabilities.
                                        #        - "probabilistic": randomly selects policies with
                                        #           probabilities determined by the meta strategies.
                                        #        - "exhaustive": selects every policy of every player.
                                        #        - "rectified": only selects strategies that have nonzero chance of
                                        #          being selected.
                                        #        - "uniform": randomly selects kwargs["number_policies_selected"]
                                        #           policies with uniform probabilities.
        'meta_strategy_method': 'nash',
                                        # String or callable taking a GenPSROSolver object and
                                        # returning a list of meta strategies (One list entry per player).
                                        #   String value can be:
                                        #       - "uniform": Uniform distribution on policies.
                                        #       - "nash": Taking nash distribution. Only works for 2 player, 0-sum games.
                                        #       - "prd": Projected Replicator Dynamics
        'rnr_iterations': prms['nb_iters'],

        # Oracle parameters
        'number_policies_sampled': 50,  # Number of different opponent policies sampled during evaluation of policy.
        'number_episodes_sampled': 5,  # Number of episodes sampled to estimate the return  of different opponent policies.
        # 'lambda_weight': [0.9, 1.],  # Player 1 does dpp player 2 does not
    }
    time_string = time.strftime("%Y%m%d-%H%M%S")
    filename_data = 'data' + time_string
    filename_plot = 'plots' + time_string
    # num_exps = 2
    num_exps = 8

    print('CPU count:' + str(mp.cpu_count()))
    pool = mp.Pool()
    result = pool.map(main, [(params, i) for i in range(num_exps)])


    rpps_dpp = []
    l_cards_p1_dpp = []
    l_cards_p2_dpp = []
    meta_games_dpp = []
    meta_probabilities_dpp = []
    lambda_weights_dpp = []
    rpps_orig = []
    l_cards_p1_orig = []
    l_cards_p2_orig = []
    meta_games_orig = []
    meta_probabilities_orig = []
    lambda_weights_orig = []
    for r in result:
        result_dpp, result_orig = r
        # DPP
        rpps_dpp.append(result_dpp.rpps)
        l_cards_p1_dpp.append(result_dpp.l_cards_p1)
        l_cards_p2_dpp.append(result_dpp.l_cards_p2)
        meta_games_dpp.append(result_dpp.meta_games)
        meta_probabilities_dpp.append(result_dpp.meta_probabilities)
        lambda_weights_dpp.append(result_dpp.lambda_weights)
        # Orig
        rpps_orig.append(result_orig.rpps)
        l_cards_p1_orig.append(result_orig.l_cards_p1)
        l_cards_p2_orig.append(result_orig.l_cards_p2)
        meta_games_orig.append(result_orig.meta_games)
        meta_probabilities_orig.append(result_orig.meta_probabilities)
        lambda_weights_orig.append(result_orig.lambda_weights)
    pickle.dump({
                 'rpps': rpps_dpp,
                 'l_cards_p1': l_cards_p1_dpp,
                 'l_cards_p2': l_cards_p2_dpp,
                 'lambda_weights': lambda_weights_dpp,
                 },
                open(os.path.join(prms['results_path'], filename_plot+'_dpp.p'), 'wb'))
    pickle.dump({
                 'rpps': rpps_orig,
                 'l_cards_p1': l_cards_p1_orig,
                 'l_cards_p2': l_cards_p2_orig,
                 'lambda_weights': lambda_weights_orig,
                 },
                open(os.path.join(prms['results_path'], filename_plot+'_orig.p'), 'wb'))
    pickle.dump({
                 'meta_games': meta_games_dpp,
                 'meta_probabilities': meta_probabilities_dpp,
                 },
                open(os.path.join(prms['results_path'], filename_data+'_dpp.p'), 'wb'))
    pickle.dump({
                 'meta_games': meta_games_orig,
                 'meta_probabilities': meta_probabilities_orig,
                 },
                open(os.path.join(prms['results_path'], filename_data+'_orig.p'), 'wb'))


    plt.figure()
    plot_error(rpps_dpp, label='dpp VS orig')
    plot_error(rpps_orig, label='orig VS orig')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'rpps' + time_string +'.pdf'))
    plt.figure()
    plot_error(l_cards_p1_dpp, label='Diversity p1')
    plot_error(l_cards_p1_orig, label='Diversity orig p1')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'div1' + time_string +'.pdf'))
    plt.figure()
    plot_error(l_cards_p2_dpp, label='Diversity p2')
    plot_error(l_cards_p2_orig, label='Diversity orig p2')
    plt.legend()
    plt.savefig(os.path.join(prms['results_path'], 'div2' + time_string +'.pdf'))
