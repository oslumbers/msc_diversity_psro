import numpy as np
from random import shuffle
from scipy.stats import entropy
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import time
# import nashpy as nash
import random
import os
from scipy import stats
import glob
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import time
import copy
np.set_printoptions(suppress=True)

from scipy.special import softmax
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)
from utils.logger import EpochLogger

dim = 30
payoffs = np.tril(np.random.uniform(-1, 1, (dim,dim)), -1)
payoffs = (payoffs - payoffs.T)

LR = 0.005  # 0.005,  0.01
TRAIN_ITERS = 10
TH = 0.03
SAVE_ALL = False

expected_card = []
sizes = []

class TorchPop:
    def __init__(self, num_learners, n_actions):
        self.pop_size = num_learners + 1
        self.n_actions = n_actions

        self.pop = [np.random.uniform(0, 1, (1, n_actions)) for _ in range(self.pop_size)]
        self.pop = [torch.from_numpy(p/p.sum(axis=1)[:, None]).float() for p in self.pop]

    def pop2numpy(self):
        return torch.cat(self.pop).cpu().detach().numpy()

    def get_pop(self):
        return torch.cat(self.pop)

    def norm(self, k):
        if (self.pop[k] < 0.).any():
            self.pop[k] -= self.pop[k].min()
        self.pop[k] /= torch.sum(self.pop[k], 1)

    def add_new(self):
        with torch.no_grad():
            new = torch.rand((1, self.n_actions), dtype=torch.float32)
            self.pop.append(new/torch.sum(new, 1))
            self.pop_size += 1

def gradient_loss_update(torch_pop, k, agg_strat, payoffs=payoffs, train_iters=10, lambda_weight=0.1, lr=0.1, dpp=True, rectified=False, iter_size=1):

    # Convert to tensor
    payoffs = torch.from_numpy(payoffs).float()
    agg_strat = torch.from_numpy(agg_strat).float()
    payoffs.requires_grad = False
    agg_strat.requires_grad = False

    # Optimiser
    optimiser = optim.Adam([torch_pop.pop[k]], lr=lr)

    for iter in range(train_iters):

        # Make strategy k trainable
        torch_pop.pop[k].requires_grad = True

        # Compute the expected return given that enemy plays agg_strat (using :k first strats)
        exp_payoff = torch_pop.pop[k] @ payoffs @ agg_strat

        # Compute cardinality of pop up until :k UNION training strategy. We use payoffs as features.
        if dpp:
            X = torch.cat(torch_pop.pop[:k+1])  # We add the training strategy to the list of pop
            M = X @ payoffs @ X.t()  # Compute metagame
            M =  f.normalize(M,dim=1,p=2) #  Normalise
            L = M @ M.t()  # Compute kernel
            L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

            # Loss
            loss = -(lambda_weight * exp_payoff + (1 - lambda_weight) * L_card)
        
        elif rectified:
            with torch.no_grad():
                X = torch.cat(torch_pop.pop[:iter_size])  # We add the training strategy to the list of pop
                M = X @ payoffs @ X.t()  # Compute kernel
                M =  f.normalize(M,dim=1,p=2) #  Normalise
                L = M @ M.t()  # Compute kernel
                L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality
            
            loss = -(lambda_weight * exp_payoff)

        else:
            with torch.no_grad():
                X = torch.cat(torch_pop.pop[:k+1])  # We add the training strategy to the list of pop
                M = X @ payoffs @ X.t()  # Compute kernel
                M =  f.normalize(M,dim=1,p=2) #  Normalise
                L = M @ M.t()  # Compute kernel
                L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

            # Loss
            loss = -(lambda_weight * exp_payoff)

        # Optimise !
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Normalise the strategy after updating
        with torch.no_grad():
            torch_pop.norm(k)
    return exp_payoff.item(), L_card.item()

def pipeline_gradient(iters=5, payoffs=payoffs, verbose=False, log=False, rectified=False,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, train_iters=10, dpp=True, logger_dir="experiments/%i" % int(time.time())):

    if log:
        # Create logger, everything will be saved to logger_dir
        logger = EpochLogger(output_dir=logger_dir)
        logger.save_config(locals())  # Save configuration in a .json file

    # Generate population as list of tensor
    dim = payoffs.shape[0]
    torch_pop = TorchPop(num_learners, dim)
    #clone_pop = torch_pop.clone().detach()
    pop = torch_pop.pop2numpy()

    # Compute initial exploitability and init stuff
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    learner_performances = [[.1] for i in range(num_learners + 1)]
    L_card = 0.
    L_cards = []
    exp_payoff = 0.
    restart = False
    for i in range(iters):
        # Define the weighting towards diversity
        lambda_weight = 0. + (0.15 / (1 + np.exp(-0.25*(i-(160/2)))) )  #
        lambda_weight = 1. - lambda_weight
        
        if rectified:
            emp_game_matrix = pop @ payoffs @ pop.T
            averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
            iter_size = pop.shape[0]
            for j in range(pop.shape[0]):
                if averages[-1][j] > 0.1:
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()

                    agg_strat = weights @ pop[:iter_size]
                    exp_payoff, L_card = gradient_loss_update(torch_pop, j, agg_strat,
                         payoffs=payoffs, train_iters=train_iters, lambda_weight=lambda_weight, lr=lr, dpp=False, rectified=rectified, iter_size=iter_size)

                    torch_pop.add_new()
                    pop = torch_pop.pop2numpy()

        else:
            for j in range(num_learners):
                # first learner (when j=num_learners-1) plays against normal meta Nash
                # second learner plays against meta Nash with first learner included, etc.
                k = pop.shape[0] - j - 1

                emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
                meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
                agg_strat = meta_nash[-1] @ pop[:k]  # Aggregate strategies according to metanash

                # Diverse PSRO update
                exp_payoff, L_card = gradient_loss_update(torch_pop, k, agg_strat,
                                     payoffs=payoffs, train_iters=train_iters, lambda_weight=lambda_weight, lr=lr, dpp=dpp)

                torch_pop.add_new()
                pop = torch_pop.pop2numpy()

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)
        L_cards.append(L_card)

        L = linear_kernel(preprocessing.normalize(pop@payoffs@pop.T, norm='l2'))
        card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        if i % 5 == 0:
            print('ITERATION: ', i, ' exp full: {:.4f}'.format(exps[-1]), 'L_CARD: {:.3f}'.format(card),
                  'lw: {:.3f}'.format(lambda_weight))

        if log:
            # Save diagnostics into logger
            logger.log_tabular('Iter', i)
            logger.log_tabular('exp', exp)
            logger.log_tabular('L_card', card)
            logger.log_tabular('pop_size', pop.shape[0])
            logger.log_tabular('lambda_weight', lambda_weight)
            logger.log_tabular('Performance', performance)
            if i % 5 == 0 and SAVE_ALL:  # We save all this data every 5 iters in a different file
                logger.save_state({'meta_nash': meta_nash,
                                   'emp_game_matrix': emp_game_matrix,
                                   'exps': exps,
                                   'L_cards': L_cards,
                                   }, itr=i)
            logger.dump_tabular(verbose=False)  # Save and print all
        
        if np.mean(exps) < 0.05 and dpp==False and rectified==False:
            restart = True
            return pop, exps, learner_performances, L_cards, restart

    return pop, exps, learner_performances, L_cards, restart

def get_br_to_strat(strat, payoffs=payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

def fictitious_play(iters=2000, payoffs=payoffs, verbose=False):
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

def get_exploitability(pop, payoffs, iters=1000):
    emp_game_matrix = pop@payoffs@pop.T
    averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
    strat = averages[-1]@pop  # Aggregate
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat@payoffs@test_br.T
    exp2 = test_br@payoffs@strat
    return exp2 - exp1

def run_experiments(num_complete_experiments=1, num_threads=20, iters=40, dim=60, lr=0.6, thresh=0.001,
                    sequential=False, pipeline=False, pipeline_joint=False, self_play=False,
                    rectified=False, qlearner=False, grad_learn=False, yscale='none', verbose=False, train_iters=10):
    sequential_exps = []
    sequential_cardinality = []

    rectified_exps = []
    rectified_cardinality = []

    pipeline_exps = []
    pipeline_cardinality = []

    pipeline_joint_exps = []
    pipeline_joint_cardinality = []

    pipeline_jointF_exps = []
    pipeline_jointF_cardinality = []

    qlearner_exps = []
    qlearner_cardinality = []

    grad_exps = []
    grad_cardinality = []

    grad_nodpp_exps = []
    grad_nodpp_cardinality = []

    self_play_exps = []
    self_play_cardinality = []
    
    i = 0
    while i <= num_complete_experiments:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        print('Experiments Complete: ', i)
        payoffs = np.tril(np.random.uniform(-1, 1, (dim, dim)), -1)
        payoffs = (payoffs - payoffs.T)

        if grad_learn:
            print('Grad no DPP')
            pop, exps, learner_performances, L_cards, restart = pipeline_gradient(iters=iters, payoffs=payoffs, verbose=verbose,
                                                                num_learners=num_threads,improvement_pct_threshold=thresh,
                                                                lr=LR, train_iters=train_iters, dpp=False)
            if restart == True:
                print('Failed experiment')
            
            elif restart == False:
                grad_nodpp_exps.append(exps)
                grad_nodpp_cardinality.append(L_cards)

                print('Grad DPP')
                pop, exps, learner_performances, L_cards, restart = pipeline_gradient(iters=iters, payoffs=payoffs, verbose=verbose,
                                                                    num_learners=num_threads,improvement_pct_threshold=thresh,
                                                                    lr=LR, train_iters=train_iters, dpp=True)
                grad_exps.append(exps)
                grad_cardinality.append(L_cards)

                print('Grad rec')
                pop, exps, learner_performances, L_cards, restart = pipeline_gradient(iters=iters, payoffs=payoffs, verbose=verbose, rectified=True,
                                                                    num_learners=num_threads,improvement_pct_threshold=thresh,
                                                                    lr=LR, train_iters=train_iters, dpp=False)
                rectified_exps.append(exps)
                rectified_cardinality.append(L_cards)
                
                i += 1


    if yscale == 'both':
        num_plots = 2
    else:
        num_plots = 2

    alpha = .4
    for j in range(num_plots):
        plt.figure()
        if grad_learn:
            if j == 0:
                avg_grad_exps = np.mean(np.array(grad_exps), axis=0)
                error_bars = stats.sem(np.array(grad_exps))
                plt.plot(avg_grad_exps, label='BR + DPP')
                plt.fill_between([i for i in range(avg_grad_exps.shape[0])],
                                 avg_grad_exps - error_bars,
                                 avg_grad_exps + error_bars, alpha=alpha)

            elif j == 1:
                avg_grad_cardinality = np.mean(np.array(grad_cardinality), axis=0)
                error_bars = stats.sem(np.array(grad_cardinality))
                plt.plot(avg_grad_cardinality, label='BR + DPP')
                plt.fill_between([i for i in range(avg_grad_cardinality.shape[0])],
                                 avg_grad_cardinality - error_bars,
                                 avg_grad_cardinality + error_bars, alpha=alpha)

            if j == 0:
                avg_grad_nodpp_exps = np.mean(np.array(grad_nodpp_exps), axis=0)
                error_bars = stats.sem(np.array(grad_nodpp_exps))
                plt.plot(avg_grad_nodpp_exps, label='BR')
                plt.fill_between([i for i in range(avg_grad_nodpp_exps.shape[0])],
                                 avg_grad_nodpp_exps - error_bars,
                                 avg_grad_nodpp_exps + error_bars, alpha=alpha)

            elif j == 1:
                avg_grad_nodpp_cardinality = np.mean(np.array(grad_nodpp_cardinality), axis=0)
                error_bars = stats.sem(np.array(grad_nodpp_cardinality))
                plt.plot(avg_grad_nodpp_cardinality, label='BR')
                plt.fill_between([i for i in range(avg_grad_nodpp_cardinality.shape[0])],
                                 avg_grad_nodpp_cardinality - error_bars,
                                 avg_grad_nodpp_cardinality + error_bars, alpha=alpha)
                
            if j == 0:
                avg_grad_rec_exps = np.mean(np.array(rectified_exps), axis=0)
                error_bars = stats.sem(np.array(rectified_exps))
                plt.plot(avg_grad_rec_exps, label='Rectified BR')
                plt.fill_between([i for i in range(avg_grad_rec_exps.shape[0])],
                                 avg_grad_rec_exps - error_bars,
                                 avg_grad_rec_exps + error_bars, alpha=alpha)

            elif j == 1:
                avg_grad_rec_cardinality = np.mean(np.array(rectified_cardinality), axis=0)
                error_bars = stats.sem(np.array(rectified_cardinality))
                plt.plot(avg_grad_rec_cardinality, label='Rectified BR')
                plt.fill_between([i for i in range(avg_grad_rec_cardinality.shape[0])],
                                 avg_grad_rec_cardinality - error_bars,
                                 avg_grad_rec_cardinality + error_bars, alpha=alpha)

        plt.legend(loc="upper left")

        if yscale == 'both':
            if j == 0:
                plt.yscale('log')
        elif yscale == 'log':
            plt.yscale('log')

        if j == 0:
            plt.savefig(os.path.join('results/' + str(dim) + 'conv' + time_string + '.pdf'))
        else:
            plt.savefig(os.path.join('results/' + str(dim) + 'card' + time_string + '.pdf'))

if __name__ =="__main__":
    run_experiments(num_complete_experiments=4, num_threads=4, iters=250, dim=30, lr=.5, thresh=TH, sequential=False, rectified=False,
                grad_learn=True,
                # pipeline_joint=True,
                yscale='none', train_iters=TRAIN_ITERS)
