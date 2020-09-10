import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
from numpy.random import RandomState
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from dppy.finite_dpps import FiniteDPP
import scipy.linalg as la
import torch
import torch.nn.functional as f
import itertools
import copy

from utils.logger import EpochLogger
from ppo_class import Agent, PPOTrainer
from kuhn_env import infostates, infostate2vector, kuhn_cards, KuhnPoker, get_exploitability, calc_ev

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.random.seed(0)

SAVE_ALL = False
ac_kwargs = dict(hidden_sizes=[64] * 2)  # [nb_neurons] * nb_layers of logits network

class KuhnPop:
    def __init__(self, env, num_learners, nb_games=100, seed=0):
        # Environment
        self.env = env
        self.nb_games = nb_games

        # Population
        self.pop_size = num_learners + 2
        self.num_learners = num_learners
        self.pop = [Agent(env, seed=i+seed, ac_kwargs=ac_kwargs) for i in range(self.pop_size)]

        # Hashed population
        self.hpop = [self.hash_agent(agent) for agent in self.pop]  # This are just dict of state to prob

        # Metagame
        self.metagame = np.zeros((self.pop_size, self.pop_size))
        for i, hagent1 in enumerate(self.hpop):
            for j, hagent2 in enumerate(self.hpop):
                    self.metagame[i, j] = self.get_payoff(hagent1, hagent2)

    def add_new(self):
        self.pop.append(Agent(self.env, seed=self.pop_size, ac_kwargs=ac_kwargs))
        self.hpop.append(self.hash_agent(self.pop[-1]))
        self.pop_size += 1
        metagame = np.zeros((self.pop_size, self.pop_size))
        metagame[:-1, :-1] = self.metagame
        self.metagame = metagame

    def get_metagame(self, k=None):
        if k is None:
            for i, hagent1 in enumerate(self.hpop):
                for j, hagent2 in enumerate(self.hpop):
                    self.metagame[i, j] = self.get_payoff(hagent1, hagent2)
            return self.metagame

        metagame = np.zeros((k, k))
        for i, hagent1 in enumerate(self.hpop):
            if i >= k:
                continue
            for j, hagent2 in enumerate(self.hpop):
                if j >= k:
                    continue
                metagame[i, j] = self.get_payoff(hagent1, hagent2)
        return metagame

    def get_payoff2(self, agent1, agent2):
        payoff = 0.
        nb_games = 0
        obs = self.env.reset()
        player_turn = 0
        swap = False
        while 1:
            if player_turn==0:
                if swap:
                    if isinstance(agent2, dict):
                        a = np.random.choice(2, 1, p=agent2[str(obs[0])])
                    else:
                        a, _, _ = agent2.ac.step(torch.as_tensor(obs[0], dtype=torch.float32))
                else:
                    if isinstance(agent1, dict):
                        a = np.random.choice(2, 1, p=agent1[str(obs[0])])
                    else:
                        a, _, _ = agent1.ac.step(torch.as_tensor(obs[0], dtype=torch.float32))
            else:
                if swap:
                    if isinstance(agent1, dict):
                        a = np.random.choice(2, 1, p=agent1[str(obs[1])])
                    else:
                        a, _, _ = agent1.ac.step(torch.as_tensor(obs[1], dtype=torch.float32))
                else:
                    if isinstance(agent2, dict):
                        a = np.random.choice(2, 1, p=agent2[str(obs[1])])
                    else:
                        a, _, _ = agent2.ac.step(torch.as_tensor(obs[1], dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            obs = next_o
            player_turn = 1 - player_turn

            if d:
                nb_games += 1
                if swap:
                    payoff += r[1]
                else:
                    payoff += r[0]
                obs = self.env.reset()
                player_turn = 0
                swap = not swap  # We swap agents playing order
                if nb_games == self.nb_games:
                    break
        return payoff / self.nb_games

    def get_payoff(self, hagent1, hagent2):
        payoff = 0.
        for cards in itertools.permutations(kuhn_cards):
            payoff += -calc_ev(hagent1, hagent2, cards, '', 0)  # This function returns payoff for second agent
            payoff += calc_ev(hagent2, hagent1, cards, '', 0)  # This function returns payoff for second agent
        return payoff / 12.  # 6 permutations x 2 players

    def hash_agent(self, agent):
        hagent = {}
        for infostate in infostates:
            obs = infostate2vector(infostate)
            with torch.no_grad():
                prob = agent.get_prob(torch.as_tensor(obs, dtype=torch.float32))
            hagent[infostate] = prob / np.sum(prob)
        return hagent

    def update_hashed_agent(self, k):
        hagent = {}
        for infostate in infostates:
            obs = infostate2vector(infostate)
            with torch.no_grad():
                prob = self.pop[k].get_prob(torch.as_tensor(obs, dtype=torch.float32))
            hagent[infostate] = prob / np.sum(prob)
        self.hpop[k] = hagent

    def agg_agents(self, weights):
        # Here weights should be the metanash of the first len(weights) agents
        agg_hashed_agent = {}
        for infostate in infostates:
            for i, w in enumerate(weights):
                prob = self.hpop[i][infostate]
                if infostate not in agg_hashed_agent:
                    agg_hashed_agent[infostate] = w*prob
                else:
                    agg_hashed_agent[infostate] += w*prob

            agg_hashed_agent[infostate] /= np.sum(agg_hashed_agent[infostate])

        return agg_hashed_agent



def L_numpy(L):
    L = L.detach().cpu().clone().numpy()
    evals, _ = la.eigh(L)
    if not np.all(np.array(evals) >= -1e-8):
        L = L + 0.01*np.eye(L.shape[0])
    return L

def ppo_update(env, kuhn_pop, k, lambda_weight=0.1, ppo_kwargs=None, dpp=True, dpp_sample=True, switch=False):

    M = kuhn_pop.get_metagame(k)
    meta_nash, _ = fictitious_play(payoffs=M, iters=1000)
    meta_nash = meta_nash[-1]

    M = f.normalize(torch.from_numpy(M).float(), dim=1, p=2)  # Normalise
    L = M @ M.t()  # Compute kernel
    L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

    if dpp_sample:
        L_np = L_numpy(L)
        DPP = FiniteDPP(kernel_type='likelihood', projection=False, **{'L': L_np})
        sample_size = 0
        while sample_size < L_card:  # Ensure at least as many as expected
            sample = np.array(DPP.sample_exact(mode='GS'))
            sample_size = len(sample)
        sample -= 1
        metanash_rectified = np.zeros_like(meta_nash)
        metanash_rectified[sample] = meta_nash[sample]
        meta_nash = metanash_rectified


    # Create PPO trainer
    agent1 = kuhn_pop.pop[k]
    agent2 = kuhn_pop.agg_agents(meta_nash)
    ppo_trainer = PPOTrainer(env, agent1, agent2, kuhn_pop, lambda_weight=lambda_weight, dpp=dpp, **ppo_kwargs)

    # Train and update in population
    exp_return, _ = ppo_trainer.train(switch=switch)
    kuhn_pop.update_hashed_agent(k)

    return exp_return, L_card




def pipeline_ppo(env, kuhn_pop, iters=5, num_learners=4, ppo_kwargs=None,
                 verbose=False, log=False, logger_dir="experiments/%i" % int(time.time()),
                 dpp=True, dpp_sample=True, switch_div=False ):

    if log:
        # Create logger, everything will be saved to logger_dir
        logger = EpochLogger(output_dir=logger_dir)
        logger.save_config(locals())  # Save configuration in a .json file

    # Compute initial exploitability and init stuff
    emp_game_matrix = kuhn_pop.get_metagame()
    meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
    agg_agent = kuhn_pop.agg_agents(meta_nash[-1])
    exp, _ = get_exploitability(agg_agent)
    exps = [exp]
    exp_returns = []
    L_cards = []
    for i in range(iters):
        switch = False
        if dpp:
            if i % 5 == 0:
                lambda_weight = 0.7
            if switch_div:
                switch = True if i>=i_switch else False
        else:
            lambda_weight = 1.
        dpp_sample_flag = dpp_sample

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = kuhn_pop.pop_size - j - 1

            # Diverse PSRO update
            exp_return, L_card = ppo_update(env, kuhn_pop, k, lambda_weight=lambda_weight,
                                            ppo_kwargs=ppo_kwargs, dpp=dpp, dpp_sample=dpp_sample_flag,
                                            switch=switch)

            if j == num_learners - 1:
                # Updated pop
                kuhn_pop.add_new()


        # calculate exploitability for meta Nash of whole population
        emp_game_matrix = kuhn_pop.get_metagame()
        meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
        agg_agent = kuhn_pop.agg_agents(meta_nash[-1])
        exp, _ = get_exploitability(agg_agent)
        exps.append(exp)
        L_cards.append(L_card)
        exp_returns.append(exp_return)

        if i % 5 == 0:
            print('ITERATION: ', i, ' exp: {:.4f}'.format(exps[-1]), 'L_CARD: {:.3f}'.format(L_card),
                  'lw: {:.3f}'.format(lambda_weight))

        if log:
            # Save diagnostics into logger
            logger.log_tabular('Iter', i)
            logger.log_tabular('exp', exp)
            logger.log_tabular('L_card', L_card)
            logger.log_tabular('pop_size', kuhn_pop.pop_size)
            logger.log_tabular('lambda_weight', lambda_weight)
            if i % 5 == 0 and SAVE_ALL:  # We save all this data every 5 iters in a different file
                logger.save_state({'meta_nash': meta_nash,
                                   'emp_game_matrix': emp_game_matrix,
                                   'exps': exps,
                                   'L_cards': L_cards,
                                   }, itr=i)
            logger.dump_tabular(verbose=False)  # Save and print all

    return exps, L_cards, exp_returns


#Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

#Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


def run_experiments(num_experiments=1, num_threads=20, iters=40, ppo_kwargs=None,
                    grad=False, grad_dpp=False, grad_sample=False, grad_dpp_sample=False,
                    yscale='none', verbose=False, switch_div=False):

    def plot_error(data, label=''):
        avg = np.mean(np.array(data), axis=0)
        error = stats.sem(np.array(data))
        plt.plot(avg, label=label)
        plt.fill_between([i for i in range(avg.shape[0])], avg - error, avg + error, alpha=alpha)

    # Grad variants #
    grad_exps = []
    grad_cardinality = []

    grad_dpp_exps = []
    grad_dpp_cardinality = []

    grad_sample_exps = []
    grad_sample_cardinality = []

    grad_dpp_sample_exps = []
    grad_dpp_sample_cardinality = []

    for i in range(num_experiments):
        print('Experiment: ', i + 1)
        # Create env and pop
        env = KuhnPoker()
        kuhn_pop = KuhnPop(env, num_threads, seed=i)

        #  grad = False, grad_dpp = False, grad_sample = False, grad_dpp_sample = False,
        if grad_sample:
            print('Grad Sample')
            exps, L_cards, exp_returns = pipeline_ppo(copy.deepcopy(env), copy.deepcopy(kuhn_pop), iters=iters, num_learners=num_threads, ppo_kwargs=ppo_kwargs,
                                                         verbose=False, log=False, logger_dir="experiments/%i" % int(time.time()),
                                                         dpp=False, dpp_sample=True)
            grad_sample_exps.append(exps)
            grad_sample_cardinality.append(L_cards)

        if grad_dpp_sample:
            print('Grad DPP & Sample')
            exps, L_cards, exp_returns = pipeline_ppo(copy.deepcopy(env), copy.deepcopy(kuhn_pop), iters=iters, num_learners=num_threads, ppo_kwargs=ppo_kwargs,
                                                         verbose=False, log=False, logger_dir="experiments/%i" % int(time.time()),
                                                         dpp=True, dpp_sample=True)
            grad_dpp_sample_exps.append(exps)
            grad_dpp_sample_cardinality.append(L_cards)

        if grad_dpp:
            print('Grad DPP')
            exps, L_cards, exp_returns = pipeline_ppo(copy.deepcopy(env), copy.deepcopy(kuhn_pop), iters=iters, num_learners=num_threads, ppo_kwargs=ppo_kwargs,
                                                      verbose=False, log=False, logger_dir="experiments/%i" % int(time.time()),
                                                      dpp=True, dpp_sample=False, switch_div=switch_div)
            grad_dpp_exps.append(exps)
            grad_dpp_cardinality.append(L_cards)

        if grad:
            print('Grad no DPP')
            exps, L_cards, exp_returns = pipeline_ppo(copy.deepcopy(env), copy.deepcopy(kuhn_pop), iters=iters, num_learners=num_threads, ppo_kwargs=ppo_kwargs,
                                                         verbose=False, log=False, logger_dir="experiments/%i" % int(time.time()),
                                                         dpp=False, dpp_sample=False)
            grad_exps.append(exps)
            grad_cardinality.append(L_cards)

    if yscale == 'both':
        num_plots = 2
    else:
        num_plots = 2

    alpha = .4
    for j in range(num_plots):
        plt.figure()
        # grad = False, grad_dpp = False, grad_sample = False, grad_dpp_sample = False
        if grad:
            if j == 0:
                plot_error(grad_exps, label='Original')
            elif j == 1:
                plot_error(grad_cardinality, label='Original')

        if grad_dpp:
            if j == 0:
                plot_error(grad_dpp_exps, label='DPP Loss')
            elif j == 1:
                plot_error(grad_dpp_cardinality, label='DPP Loss')

        if grad_dpp_sample:
            if j == 0:
                plot_error(grad_dpp_sample_exps, label='DPP Loss and Sample')
            elif j == 1:
                plot_error(grad_dpp_sample_cardinality, label='DPP Loss and Sample')

        if grad_sample:
            if j == 0:
                plot_error(grad_sample_exps, label='DPP Sample')
            elif j == 1:
                plot_error(grad_sample_cardinality, label='DPP Sample')

        if switch_div:
            plt.title('With switching at 50')
        plt.legend(loc="upper left")

        if yscale == 'both':
            if j == 0:
                plt.yscale('log')
        elif yscale == 'log':
            plt.yscale('log')




if __name__ =="__main__":
    ppo_kwargs = {'pi_lr':3e-2,  # 3e-4, 3e-1
                  'vf_lr':1e-2,  # 1e-3, 1e-1
                  'target_kl':0.01,
                  'clip_ratio':0.2,
                 'train_pi_iters':80,
                  'train_v_iters':80}
    i_switch = 0  # Iteration at which we switch

    run_experiments(num_experiments=10, num_threads=4, iters=100, ppo_kwargs=ppo_kwargs,
                    grad=True,
                    grad_dpp=True,
                    grad_sample=True,
                    grad_dpp_sample=True,
                    yscale='none', switch_div=False)


    plt.show()

























