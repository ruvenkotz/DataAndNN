import pandas as pd
import numpy as np
import torch as tf

pathname = '/Users/alexpaley/Desktop/cfdata/WinOrBlock1.csv'


def X_y (pth, head = False):
    """
    This function rads a csv from path "pth"
    :param pth: the path containing the csv
    :param head: if true, calls df.head() to make debugging easier
    :return: Returns two float tensors which contain the board attributes for X and the respective labels y
    """
    df = pd.read_csv (pth) #dataframe from csv
    if head:
        df = df.head() #cuts to 5 rows
    X = df.iloc[:, 0:-3] #get dataframe from the beginning until the last three
    y = df.iloc[:  , -3:]
    tensorX = tf.tensor(X.values).float()
    tensory = tf.tensor(y.values).float()
    return tensorX, tensory



def final_func(x):
    """
    Takes the input tensor and makes it such that the highest value gets turned into 1.0 and every other value
    gets turned into 0.0
    :param x: input tensor
    :return: A new and updated tensor
    """
    one = max(x)
    arr = []
    for i in range(0, len(x)):
        if x[i] == one:
           arr.append(1.0)
        else:
            arr.append(0.0)
    tensor = tf.from_numpy(np.array(arr))
    return tensor

def ff_func(X):
    """
    This is used to evaluate the model.
    This takes a tensor of tensors and calls final_func on all of them.
    :param X: the input tensor of tensors.
    :return: A new and updated tensor which contains all of the updated tensors.
    """
    lst = []
    for i in X:
        tensor = final_func(i)
        lst.append(tensor)

    res = tf.tensor([])
    for i in lst:
        i = tf.tensor(i)
        res = tf.cat((res,i), dim=0)
    res = res.reshape((len(lst), 3))
    return res.float()

def concat_csv(csv1, csv2, csvpthname):
    one = pd.read_csv(csv1)
    print(len(one))
    two = pd.read_csv(csv2)
    print(len(two))
    lst = [one, two]
    three = pd.concat(lst)
    print(len(three))
    three.to_csv(csvpthname, index=False)
