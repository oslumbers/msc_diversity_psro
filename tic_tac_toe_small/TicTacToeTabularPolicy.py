import numpy as np
from tictactoe import find_all_boards
import copy

class TicTacToeTabularPolicy:

    def __init__(self, game):
        infostates = game.infostates
        self.hash_infostates, self.infostate_action_prob, self.legal_action_mask, self.player_turn = self.to_array(infostates)

    def to_array(self, infostates):
        hash_infostates = {}
        infostate_action_prob = np.zeros((len(infostates), 9))
        legal_action_mask = np.zeros((len(infostates), 9))
        player_turn = np.zeros((len(infostates),))
        for i, key in enumerate(infostates):
            hash_infostates[key] = i
            legal_action_mask[i, :] = infostates[key]['mask'].astype(int)
            infostate_action_prob[i, :] = infostates[key]['probabilities']
            player_turn[i] = infostates[key]['player_turn']

        return hash_infostates, infostate_action_prob, legal_action_mask, player_turn

    def __copy__(self):
        return copy.deepcopy(self)

    def action(self, key):
        i = self.hash_infostates[key]
        probs = self.infostate_action_prob[i, :]
        return np.random.choice(np.arange(9), p=probs)

    def copy_with_noise(self, alpha=0.0, beta=0.0, random_state=np.random.RandomState()):
        new_policy = self.__copy__()
        noise_mask = random_state.normal(size=self.infostate_action_prob.shape)
        noise_mask = np.exp(beta * noise_mask) * self.legal_action_mask
        noise_mask = noise_mask / (np.sum(noise_mask, axis=1).reshape(-1, 1))
        new_policy.infostate_action_prob = (1-alpha)*new_policy.infostate_action_prob + alpha*noise_mask
        return new_policy
