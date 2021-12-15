import torch
import Board
import BoardFunctions
modelpth = '/Users/alexpaley/Desktop/dfmodel1_real/queried.pth'

model = torch.load(modelpth)
model.eval()

import copy
class NN:
    # This is a connect four that uses a neural net to choose a move


    def _init_(self):
        nextmove = 0


    def choose_a_move(self, player, b):
        """
        This function takes in every possible move and uses a neural network to evaluate it's chance of winning.
        It chooses the move with the highest probability of winning.
        :param player:
        :param b:
        :return:
        """
        legal_moves = BoardFunctions.legal_moves(b)
        move = (legal_moves[0], 0)
        for i in legal_moves:
            bnew = copy.deepcopy(b)
            bnew[(i[0])][i[1]] = 1 ##change to 2 if the opponent is YELLOW
            tensor = torch.tensor(bnew).float()
            tensor = tensor.reshape((1, 42))
            pred = model(tensor)
            temp = pred[0][0].item()
            if temp > move[1]:
                move = (i, temp)
        return move
