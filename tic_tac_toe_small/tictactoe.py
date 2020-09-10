import numpy as np
import pickle


class TicTacToe:
    def __init__(self, symmetry=False):
        self.board = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        self.symmetry = symmetry
        if self.symmetry:
            self.infostates = pickle.load(open('all_boards.p', 'rb'))
        else:
            self.infostates = pickle.load(open('all_boards_no_symmetry.p', 'rb'))

    def num_players(self):
        return 2

    def new_initial_state(self):
        self.board = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        return self

    def is_terminal(self, board=None):
        if board is None:
            return evaluate(self.board) != 0 or np.count_nonzero(self.board) - self.board.size == 0
        else:
            return evaluate(board) != 0 or np.count_nonzero(board) - board.size == 0

    def returns(self):
        winner = evaluate(self.board)
        if winner !=0:
            payoff = np.array([-1., -1.])
            payoff[winner - 1] = 1.
        else:
            payoff = np.array([0., 0.])
        return payoff

    def current_player(self):
        return 0 if np.count_nonzero(self.board) % 2 == 0 else 1

    def board2key(self):
        return str(self.board.flatten().tolist())

    def key2board(self, key):
        return np.array(eval(key)).reshape((3, 3))

    def update(self, a, player):

        if self.symmetry:

            b = self.board.copy()
            idxs = np.unravel_index(a, self.board.shape)
            b[idxs] = 1 if player == 0 else 2
            if self.is_terminal(board=b):
                self.board = b
                return
            key = str(b.flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b, k=1).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b, k=2).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b, k=3).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return


            b1 = np.fliplr(b)
            key = str(b1.flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b1, k=1).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b1, k=2).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b1, k=3).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return

            b2 = np.flipud(b)
            key = str(b2.flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b2, k=1).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b2, k=2).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return
            key = str(np.rot90(b2, k=3).flatten().tolist())
            if key in self.infostates:
                self.board = self.key2board(key)
                return

        else:
            idxs = np.unravel_index(a, self.board.shape)
            self.board[idxs] = player+1


# Check for empty places on board
def possibilities(board):
    l = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == 0:
                l.append(np.ravel_multi_index((i, j), board.shape))
    return l


def evaluate(board):
    for player in [1, 2]:
        # Row win
        for i in range(board.shape[0]):
            if (board[i, :] == player).all():
                return player
        # Col win
        for j in range(board.shape[1]):
            if (board[:, j] == player).all():
                return player
        # Diag win
        if (np.diag(board) == player).all():
            return player
        elif (np.diag(np.fliplr(board)) == player).all():
            return player
    # Draw
    return 0


def is_terminal(board):
    return evaluate(board) != 0 or np.count_nonzero(board) - board.size == 0


def is_winner(board):
    return evaluate(board) != 0


def is_draw(board):
    return np.count_nonzero(board) - board.size == 0


def get_payoff(board):
    payoff = [0, 0]
    winner = evaluate(board)
    payoff[winner - 1] = 1
    return payoff


def apply_move(board, player, a):
    board[np.unravel_index(a, board.shape)] = 1 if player == 0 else 2
    return board


def add_board(all_boards, board, only_decision=False, symmetry=False):
    if only_decision and is_terminal(board):
        return
    a = possibilities(board)
    prob = (1/len(a))*np.ones((9, ))
    mask = np.array([True] * len(prob))
    mask[a] = False
    prob[mask] = 0.

    player_turn = 0 if np.count_nonzero(board)%2==0 else 1

    if symmetry:
        b = board.copy()
        b1 = np.fliplr(b)
        b2 = np.flipud(b)
        b3 = np.rot90(b, k=1)
        b4 = np.rot90(b, k=2)
        b5 = np.rot90(b, k=3)
        b6 = np.rot90(b1, k=1)
        b7 = np.rot90(b1, k=2)
        b8 = np.rot90(b1, k=3)
        b9 = np.rot90(b2, k=1)
        b10 = np.rot90(b2, k=2)
        b11 = np.rot90(b2, k=3)
        key1, key2 = str(b1.flatten().tolist()), str(b2.flatten().tolist())
        key3, key4, key5 = str(b3.flatten().tolist()), str(b4.flatten().tolist()), str(b5.flatten().tolist())
        key6, key7, key8 = str(b6.flatten().tolist()), str(b7.flatten().tolist()), str(b8.flatten().tolist())
        key9, key10, key11 = str(b9.flatten().tolist()), str(b10.flatten().tolist()), str(b11.flatten().tolist())

        if key1 in all_boards or key2 in all_boards \
                or key3 in all_boards or key4 in all_boards or key5 in all_boards \
                or key6 in all_boards or key7 in all_boards or key8 in all_boards \
                or key9 in all_boards or key10 in all_boards or key11 in all_boards:
            return


        key = str(board.flatten().tolist())
        if key not in all_boards:
            all_boards[key] = {'player_turn': player_turn, 'probabilities': prob, 'mask': np.logical_not(mask)}
            print(len(all_boards))
    else:
        key = str(board.flatten().tolist())
        if key not in all_boards:
            all_boards[key] = {'player_turn': player_turn, 'probabilities': prob, 'mask': np.logical_not(mask)}
            print(len(all_boards))


def find_all_boards(board, player, all_boards, only_decision=False, symmetry=False):
    for a in possibilities(board):
        pre_board = board.copy()
        board = apply_move(board, player, a)
        add_board(all_boards, board, only_decision=only_decision, symmetry=symmetry)
        if not is_terminal(board):
            player = 1 - player
            find_all_boards(board, player, all_boards, only_decision=only_decision, symmetry=symmetry)
            board = pre_board
            player = 1 - player
        elif is_winner(board):
            board = pre_board


if __name__ == "__main__":
    board = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
    player = 0
    all_boards = {str(board.flatten().tolist()): {'player_turn': 0, 'probabilities': (1/9)*np.ones((9, )),
                                                  'mask':np.array(9*[True])}}


    import pickle
    find_all_boards(board, player, all_boards, only_decision=True, symmetry=False)
    pickle.dump(all_boards, open('all_boards_no_symmetry.p', 'wb'))
