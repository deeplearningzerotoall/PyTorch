import torch
import torch.optim as optim
import numpy as np

sample = " if you want you"

# make dictionary
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}

# hyper parameters
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

# data setting
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

# transform as torch tensor variable
X = torch.Tensor(x_one_hot).float()
Y = torch.Tensor(y_data).long()

# declare RNN
rnn = torch.nn.RNN(dic_size, hidden_size, batch_first=True)

# loss & optimizer setting
weights = torch.Tensor(np.ones(dic_size)).float()  # weight for each class, not for position in sequence
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(rnn.parameters(), lr=0.1)

# start training
for i in range(50):

    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
