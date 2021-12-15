import torch

def model1():
    """
    Ideal learning rate is .02 and ideal momentum is .96
    :return: model, lr, momentum
    """
    return torch.nn.Sequential(
    torch.nn.Linear(42, 22),
    torch.nn.ReLU(),
    torch.nn.Linear(22, 3),
    torch.nn.Softmax(),
), .02, .96

def model2():
    """
    Ideal learning rate is .02 and ideal momentum is .96
    :return: model, lr, momentum
    """
    return torch.nn.Sequential(
        torch.nn.Linear(42,3),
        torch.nn.ReLU(),
        torch.nn.Softmax(),
    ), .2, .96

def model3():
    """
    Ideal learning rate is .02 and ideal momentum is .96
    :return: model, lr, momenutm
    """
    return torch.nn.Sequential(
        torch.nn.Linear(42, 35),
        torch.nn.Sigmoid(),
        torch.nn.Linear(35, 22),
        torch.nn.Sigmoid(),
        torch.nn.Linear(22, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 3),
        torch.nn.Softmax()
    ), .2, .96



