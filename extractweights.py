import torch
modelpth = '/Users/alexpaley/Desktop/dfmodel3/random.pth'

model = torch.load(modelpth)
model.eval()



weights = {}

for i, (name, para) in enumerate(model.named_parameters()):
    #print('{}: {}'.format(name, para))
    weights[i] = para[:].detach().numpy()

def arr_to_str(arr):
    """
    This function turns an array of numbers into a string of said numbers. e.g. arr_to_str([1,2,3]) = "1 2 3 "
    :param arr: The array of numbers
    :return: string of numbers
    """
    string = ""
    for i in arr:
        string = string + str(i) + " "
    return string

def write_to_file(path, data, lst, model):
    """
    This function takes the list of weights and writes it to a textfile. First it
    :param path: the path to write it to which will have the data name and .txt appended to it.
    :param data: the type of data e.g. random, center, endgames
    :param lst: the list of weights and biases for this model
    :param model: which neural net model we are using
    :return:
    """
    path = path + '/' + data + '.txt'
    with open(path, 'x') as ff:
        if model == 1:
            ff.writelines("42 22 3" + "\n")
            ff.writelines("0" + "\n")
        if model == 2:
            ff.writelines("42 3" + "\n")
            ff.writelines("0" + "\n")
        if model == 3:
            ff.writelines("42 35 22 10 3" + "\n")
            ff.writelines("1" + "\n")
    for i in lst:
        if len(i.shape) > 1:
            for j in i:
                string = arr_to_str(j)
                with open(path, 'a') as f:
                    f.writelines(string + "\n")
        else:
            string = arr_to_str(i)
            with open(path, 'a') as f:
                f.writelines(string + "\n")



#write_to_file('/Users/alexpaley/Desktop/model3_weights_txt', "random", [weights[0], weights[1], weights[2], weights[3], 1)