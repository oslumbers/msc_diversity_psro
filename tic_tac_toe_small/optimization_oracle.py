# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class of Optimization Oracles generating best response against opponents.
Oracles are as defined in (Lanctot et Al., 2017,
https://arxiv.org/pdf/1711.00832.pdf ), functions generating a best response
against a probabilistic mixture of opponents. This class implements the abstract
class of oracles, and a simple oracle using Evolutionary Strategy as
optimization method.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer
import abstract_meta_trainer
from sklearn import preprocessing
import itertools


class AbstractOracle(object):
  """The abstract class representing oracles, a hidden optimization process."""

  def __init__(self,
               number_policies_sampled=100,
               number_episodes_sampled=10,
               nb_players=2,
               **unused_oracle_specific_kwargs):
    """Initialization method for oracle.
    Args:
      number_policies_sampled: Number of different opponent policies sampled
        during evaluation of policy.
      number_episodes_sampled: Number of episodes sampled to estimate the return
        of different opponent policies.
      **unused_oracle_specific_kwargs: Oracle specific args, compatibility
        purpose. Since oracles can vary so much in their implementation, no
        specific argument constraint is put on this function.
    """
    self._number_policies_sampled = number_policies_sampled
    self._number_episodes_sampled = number_episodes_sampled
    self.l_card = [[] for _ in range(nb_players)]
    self.nb_players = nb_players

  def set_iteration_numbers(self, number_policies_sampled,
                            number_episodes_sampled):
    """Changes the number of iterations used for computing episode returns.
    Args:
      number_policies_sampled: Number of different opponent policies sampled
        during evaluation of policy.
      number_episodes_sampled: Number of episodes sampled to estimate the return
        of different opponent policies.
    """
    self._number_policies_sampled = number_policies_sampled
    self._number_episodes_sampled = number_episodes_sampled

  def __call__(self, game, policy, total_policies, current_player,
               probabilities_of_playing_policies,
               **oracle_specific_execution_kwargs):
    """Call method for oracle, returns best response against a set of policies.
    Args:
      game: The game on which the optimization process takes place.
      policy: The current policy, in policy.Policy, from which we wish to start
        optimizing.
      total_policies: A list of all policy.Policy strategies used for training,
        including the one for the current player.
      current_player: Integer representing the current player.
      probabilities_of_playing_policies: A list of arrays representing, per
        player, the probabilities of playing each policy in total_policies for
        the same player.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.
    """
    raise NotImplementedError("Calling Abstract class method.")

  def evaluate_policy(self, game, pol, total_policies, current_player,
                      probabilities_of_playing_policies,
                      **oracle_specific_execution_kwargs):
    """Evaluates a specific policy against a nash mixture of policies.
    Args:
      game: The game on which the optimization process takes place.
      pol: The current policy, in policy.Policy, from which we wish to start
        optimizing.
      total_policies: A list of all policy.Policy strategies used for training,
        including the one for the current player.
      current_player: Integer representing the current player.
      probabilities_of_playing_policies: A list of arrays representing, per
        player, the probabilities of playing each policy in total_policies for
        the same player.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.
    Returns:
      Average return for policy when played against policies_played_against.
    """
    rectify_training = oracle_specific_execution_kwargs.get("rectify_training")

    totals = 0
    count = 0
    for _ in range(self._number_policies_sampled):
      # For Rectified Nash, it's necessary to make sure that we're only
      # including policies against which the evaluated policy wins on
      # expectation, which forces us to make multiple runs per policy.

      policies_selected = []
      for k in range(len(total_policies)):
        if k == current_player:
          policies_selected.append(pol)
        else:
          selected_opponent = np.random.choice(
              total_policies[k],
              1,
              False,
              p=probabilities_of_playing_policies[k]).reshape(-1)[0]
          policies_selected.append(selected_opponent)

      policy_total = 0
      for _ in range(self._number_episodes_sampled):
        new_return = abstract_meta_trainer.sample_episode(
            game.new_initial_state(),
            policies_selected).reshape(-1)[current_player]
        policy_total += new_return
      policy_total /= self._number_episodes_sampled

      if rectify_training:
        gain_on_average = int(policy_total >= 0)
        policy_total = gain_on_average * policy_total
        add_counter = gain_on_average
      else:
        add_counter = 1

      totals += policy_total
      count += add_counter

    # Avoid the 0 / 0 case.
    return totals / max(1, count)


class EvolutionaryStrategyOracle(AbstractOracle):
  """Oracle using evolutionary strategies to compute BR to policies."""

  def __init__(self, alpha=0.1, beta=10, n_evolution_tests=100, **kwargs):
    self._alpha = alpha
    self._beta = beta
    self._n_evolution_tests = n_evolution_tests
    super(EvolutionaryStrategyOracle, self).__init__(**kwargs)

  def __call__(self, game, pol, total_policies, current_player,
               probabilities_of_playing_policies,
               **oracle_specific_execution_kwargs):
    """Call method for oracle, returns best response against a set of policies.
    Args:
      game: The game on which the optimization process takes place.
      pol: The current policy, in policy.Policy, from which we wish to start
        optimizing.
      total_policies: A list of all policy.Policy strategies used for training,
        including the one for the current player.
      current_player: Integer representing the current player.
      probabilities_of_playing_policies: A list of arrays representing, per
        player, the probabilities of playing each policy in total_policies for
        the same player.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.
    Returns:
      Expected (Epsilon) best response.
    """
    max_perf = -np.infty
    best_policy = None
    # Easy to multithread, but this is python.
    for _ in range(self._n_evolution_tests):
      new_policy = pol.copy_with_noise(alpha=self._alpha, beta=self._beta)
      new_value = self.evaluate_policy(game, new_policy, total_policies,
                                       current_player,
                                       probabilities_of_playing_policies,
                                       **oracle_specific_execution_kwargs)

      lambda_weight = oracle_specific_execution_kwargs.get("lambda_weight")
      solver = oracle_specific_execution_kwargs.get("solver")

      # Update metagame with new policy
      new_policies = [[] for _ in range(2)]
      new_policies[current_player] = [new_policy]
      meta_games = self.update_metagame(solver, new_policies)
      M = meta_games[current_player]

      # Compute cardinality
      M = preprocessing.normalize(M, norm='l2', axis=1)  # Normalise
      L = M @ M.T  # Compute kernel
      L_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))  # Compute cardinality

      new_value = lambda_weight*new_value + (1-lambda_weight)*L_card

      if new_value > max_perf:
        max_perf = new_value
        best_policy = new_policy
        self.l_card[current_player] = L_card

    return best_policy
  
  
  def update_metagame(self, solver, new_policies, seed=None):
    """Given new agents in _new_policies, update meta_games through simulations.
    Args:
      seed: Seed for environment generation.
    Returns:
      Meta game payoff matrix.
    """
    if seed is not None:
      np.random.seed(seed=seed)

    # Concatenate both lists.
    updated_policies = [
        solver._policies[k] + new_policies[k]
        for k in range(solver._num_players)
    ]

    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    total_number_policies = [
        len(updated_policies[k]) for k in range(solver._num_players)
    ]
    number_older_policies = [
        len(solver._policies[k]) for k in range(solver._num_players)
    ]
    number_new_policies = [
        len(new_policies[k]) for k in range(solver._num_players)
    ]

    # Initializing the matrix with nans to recognize unestimated states.
    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(solver._num_players)
    ]

    # Filling the matrix with already-known values.
    older_policies_slice = tuple(
        [slice(len(solver._policies[k])) for k in range(solver._num_players)])
    for k in range(solver._num_players):
      meta_games[k][older_policies_slice] = solver._meta_games[k]

    # Filling the matrix for newly added policies.
    for current_player in range(solver._num_players):
      # Only iterate over new policies for current player ; compute on every
      # policy for the other players.
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, solver._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        if np.isnan(meta_games[current_player][tuple(used_index)]):
          estimated_policies = [
              updated_policies[k][current_index[k]]
              for k in range(current_player)
          ] + [
              new_policies[current_player][current_index[current_player]]
          ] + [
              updated_policies[k][current_index[k]]
              for k in range(current_player + 1, solver._num_players)
          ]
          utility_estimates = solver.sample_episodes(estimated_policies,
                                                   solver._sims_per_entry)
          for k in range(solver._num_players):
            meta_games[k][tuple(used_index)] = utility_estimates[k]

    return meta_games
