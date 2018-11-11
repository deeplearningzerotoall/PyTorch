import torch
import torch.optim as optim
import numpy as np

char_set = ['h', 'i', 'e', 'l', 'o']

# hyper parameters
input_size = len(char_set)
hidden_size = len(char_set)
learning_rate = 0.1

# data setting
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

# transform as torch tensor variable
X = torch.Tensor(x_one_hot).float()
Y = torch.Tensor(y_data).long()

# declare RNN
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)  # batch_first guarantees the order of output = (B, S, F)

# loss & optimizer setting
weights = torch.Tensor(np.ones(input_size)).float()  # weight for each class, not for position in sequence
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# start training
for i in range(100):

    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
