import numpy as np
import gym
from typing import List, Dict
from gym.spaces import Discrete, Box
import random
import itertools


infostates = ['KPB', 'K', 'QPB', 'Q', 'JPB', 'J',  # turn for player 1
              'KB', 'KP', 'QB', 'QP', 'JB', 'JP']  # turn for player 2
actions = ['P', 'B']
kuhn_cards = ['J', 'Q', 'K']

def infostate2vector(infostate):
    # infosate = card + history just the sum of two strings
    assert infostate in infostates, infostate + " not in infostates"
    idx = infostates.index(infostate)
    vec = np.zeros((len(infostates),), dtype=np.float32)
    vec[idx] = 1.
    return vec

class KuhnPoker:

    def __init__(self):
        self.done = False
        self.current_player = 0
        self.current_cards = None
        self.history = None

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0., high=1., shape=(len(infostates),), dtype=np.float32)

        self.infostates = infostates
        self.actions = actions
        self.cards = kuhn_cards

    @staticmethod
    def is_terminal(history: str) -> bool:
        return history in ['BP', 'BB', 'PP', 'PBB', 'PBP']

    @staticmethod
    def get_payoff(history: str, cards: List[str]) -> int:
        """ATTENTION: this gets payoff for 'active' player in terminal history"""
        if history in ['BP', 'PBP']:
            return +1
        else:  # PP or BB or PBB
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opponent_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opponent_card == 'J':
                return payoff
            else:
                return -payoff

    def step(self, a):
        self.history += self.actions[a]  # Update history
        self.current_player = 1 - self.current_player  # Next player

        if self.is_terminal(self.history):
            self.done = True
            r = ((-1)**self.current_player)*self.get_payoff(self.history, self.current_cards)  # Return reward for player 1
            next_obs = [0., 0.]
        else:
            r = 0.
            next_obs = [infostate2vector(self.current_cards[self.current_player]+self.history),
                        self.current_cards[self.current_player]+self.history]

        info = None

        return next_obs, r, self.done, info

    def reset(self, set_cards=None):
        self.current_player = 0
        self.done = False
        self.current_cards = random.sample(self.cards, 2) if set_cards is None else set_cards
        self.history= ''
        return [infostate2vector(self.current_cards[0]+self.history), self.current_cards[0]+self.history]


def calc_ev(p1_strat, p2_strat, cards, history, active_player):
    """ Returns value for player 2!! (p2_strat) """
    if KuhnPoker.is_terminal(history):
        return -KuhnPoker.get_payoff(history, cards)
    my_card = cards[active_player]
    next_player = (active_player + 1) % 2
    if active_player == 0:
        strat = p1_strat[my_card + history]
    else:
        strat = p2_strat[my_card + history]
    return -np.dot(strat, [calc_ev(p1_strat, p2_strat, cards, history + a, next_player) for a in actions])


def calc_best_response(agg_hagent, br_strat_map, br_player, cards, history, active_player, prob):
    """
    after chance node, so only decision nodes and terminal nodes left in game tree
    """
    if KuhnPoker.is_terminal(history):
        return -KuhnPoker.get_payoff(history, cards)
    key = cards[active_player] + history
    next_player = (active_player + 1) % 2
    if active_player == br_player:
        vals = [calc_best_response(agg_hagent, br_strat_map, br_player, cards, history + action,
                                   next_player, prob) for action in actions]
        best_response_value = max(vals)
        if key not in br_strat_map:
            br_strat_map[key] = np.array([0.0, 0.0])
        br_strat_map[key] = br_strat_map[key] + prob * np.array(vals, dtype=np.float64)
        return -best_response_value
    else:
        strategy = agg_hagent[key]
        action_values = [calc_best_response(agg_hagent, br_strat_map, br_player, cards,
                                            history + action, next_player, prob * strategy[idx])
                         for idx, action in enumerate(actions)]
        return -np.dot(strategy, action_values)

def get_exploitability(agg_hagent):
    exploitability = 0

    br_hagent = {}
    for cards in itertools.permutations(kuhn_cards):
        calc_best_response(agg_hagent, br_hagent, 0, cards, '', 0, 1.0)
        calc_best_response(agg_hagent, br_hagent, 1, cards, '', 0, 1.0)

    for k,v in br_hagent.items():
        v[:] = np.where(v == np.max(v), 1, 0)

    for cards in itertools.permutations(kuhn_cards):
        ev_1 = calc_ev(agg_hagent, br_hagent, cards, '', 0)
        ev_2 = calc_ev(br_hagent, agg_hagent, cards, '', 0)
        exploitability += 1 / 6 * (ev_1 - ev_2)
    return exploitability, br_hagent



if __name__=='__main__':
    env = KuhnPoker()
    a_list = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1] * 6


    for cards in itertools.permutations(kuhn_cards):
        for i in range(5):
            obs = env.reset(set_cards=cards)
            while 1:
                a = a_list.pop(0)
                next_obs, r, d, _ = env.step(a)
                if d:
                    print(cards[0]+cards[1]+env.history, r)
                    break
        print('-----------')


































