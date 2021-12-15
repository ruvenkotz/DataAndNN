import torch
from torch import nn
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from neuralnetfunc import X_y, ff_func
import matplotlib.pyplot as plt
import models as mm

datapath_endgame = '/Users/alexpaley/Desktop/4701/Endgames2.csv'

X, y = X_y(datapath_endgame, head = False) # gets the data from a csv into an X with 42 inputs symbolizing the board and a y with 3 inputs symbolizing win, loss, or draw


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)


training_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))


from torch.utils.data import DataLoader
BATCHSIZE = 64
# We want a nn with 42 input neurons, 22 neurons for one hidden layer, 3 nuerons per output layer
model, lr, mom = mm.model3()

#for i in range(3):
    #mom_list = []

    #for j in range(10):
lr = .2
mom = .96
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

with torch.no_grad(): #grad's keep track of calculation for back propogation
    for X, y in test_dataloader:
        pred = model(X)
        pred = ff_func(pred)
        test_loss += criterion(pred, y).item()
        #I have to convert the softmax probabilities into an input that is of the form xxy, xyx, or yxx where y =1 and x =0.
        #acc += (torch.sum(torch.argmax(pred, dim=0) == y).item())
        #acc += (pred == y).sum()
        for ind in range(0, len(pred)):
            if torch.equal(pred[ind], y[ind].float()):
                acc += 1

test_loss /= num_batches
acc /= size
#mom_list.append(acc)
print("Test accuracy: " + str(acc) + "  Test loss: " + str(test_loss))
#inp = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,0])
#spec = model(inp.float())
#if acc > prev[0]:
#    prev = (acc, mom)
#print(prev)
#y_axis = mom_list
#x_axis = [.90,.91,.92,.93,.94,.95,.96,.97,.98,.99]
#plt.plot(x_axis, y_axis)
#plt.xlabel("momentum")
#plt.ylabel("accuracy")
#plt.title("LR = " + str(lr))
#plt.show()
#plt.clf()