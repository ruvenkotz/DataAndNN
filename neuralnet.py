import torch
from torch import nn
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from neuralnetfunc import X_y, ff_func
import models as mm

"""
Some of the baics of this script were gained from "Copy of Pytorch Basics 2021" from NLP
"""


BATCHSIZE = 64


datapath_random = '/Users/alexpaley/Desktop/cfdata/Train1.csv'
datapath_w_o_b = '/Users/alexpaley/Desktop/cfdata/Train2.csv'
datapath_center = '/Users/alexpaley/Desktop/cfdata/CenterPlay1.csv'
datapath_endgame = '/Users/alexpaley/Desktop/cfdata/Endgames2.csv'
datapath_random_a_lot = '/Users/alexpaley/Desktop/cfdata/Random2.csv'
datapath_queried = '/Users/alexpaley/Desktop/cfdata/queried.csv'

modelpth = '/Users/alexpaley/Desktop/dfmodel3/random_a_lot.pth'



X, y = X_y(datapath_random_a_lot) # gets the data from a csv into an X with 42 inputs symbolizing the board and a y with 3 inputs symbolizing win, loss, or draw


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)


training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))


from torch.utils.data import DataLoader

# We want a nn with 42 input neurons, 22 neurons for one hidden layer, 3 nuerons per output layer
model, lr, mom = mm.model3()

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
criterion = nn.CrossEntropyLoss()
model.train()
train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE)
for epoch in range(16):
    for batch_num, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            acc = (torch.sum(torch.argmax(pred, dim=0) == y).item()) / len(y)
            print("epoch: " + str(epoch) + ";    batch: " + str(batch_num))



test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE)

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
test_loss, acc = 0, 0

model.eval()

with torch.no_grad():
    for X, y in test_dataloader:
        pred = model(X)
        pred = ff_func(pred)
        # I have to convert the softmax probabilities into an input that is of the form xxy, xyx, or yxx where y =1 and x =0.
        test_loss += criterion(pred, y).item()

        for ind in range(0, len(pred)):
            if torch.equal(pred[ind], y[ind].float()):
                acc += 1

test_loss /= num_batches
acc /= size

print("Test accuracy: " + str(acc) + "  Test loss: " + str(test_loss))

torch.save(model, modelpth)

### save the neural network weights and then bring it back up in the engine

