# Lab 10 MNIST and softmax
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = 0.7

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# MNIST data image of shape 28 * 28 = 784
# nn layers
linear1 = torch.nn.Linear(784, 512, bias=True)
linear2 = torch.nn.Linear(512, 512, bias=True)
linear3 = torch.nn.Linear(512, 512, bias=True)
linear4 = torch.nn.Linear(512, 512, bias=True)
linear5 = torch.nn.Linear(512, 10, bias=True)
selu = torch.nn.SELU()
selu_dropout = torch.nn.AlphaDropout(p=1 - keep_prob)

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)

# model
model = torch.nn.Sequential(linear1, selu, selu_dropout,
                            linear2, selu, selu_dropout,
                            linear3, selu, selu_dropout,
                            linear4, selu, selu_dropout,
                            linear5).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
model.train()    # set the model to train mode (dropout=True)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# Test model and check accuracy
with torch.no_grad():
    model.eval()    # set the model to evaluation mode (dropout=False)

    # Test the model using test sets
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())


'''
Epoch: 0001 cost = 0.551087022
Epoch: 0002 cost = 0.257470459
Epoch: 0003 cost = 0.204459295
Epoch: 0004 cost = 0.176355377
Epoch: 0005 cost = 0.156724215
Epoch: 0006 cost = 0.138451248
Epoch: 0007 cost = 0.129061013
Epoch: 0008 cost = 0.118015580
Epoch: 0009 cost = 0.107403427
Epoch: 0010 cost = 0.101504572
Epoch: 0011 cost = 0.096441686
Epoch: 0012 cost = 0.093530126
Epoch: 0013 cost = 0.085378215
Epoch: 0014 cost = 0.083097845
Epoch: 0015 cost = 0.079341516
Epoch: 0016 cost = 0.077695794
Epoch: 0017 cost = 0.071116112
Epoch: 0018 cost = 0.067317173
Epoch: 0019 cost = 0.067397237
Epoch: 0020 cost = 0.065503702
Epoch: 0021 cost = 0.068812400
Epoch: 0022 cost = 0.070660502
Epoch: 0023 cost = 0.056371965
Epoch: 0024 cost = 0.059767291
Epoch: 0025 cost = 0.052266609
Epoch: 0026 cost = 0.051223129
Epoch: 0027 cost = 0.057534486
Epoch: 0028 cost = 0.050468620
Epoch: 0029 cost = 0.061170042
Epoch: 0030 cost = 0.059151404
Epoch: 0031 cost = 0.049802747
Epoch: 0032 cost = 0.045141894
Epoch: 0033 cost = 0.046983775
Epoch: 0034 cost = 0.040513776
Epoch: 0035 cost = 0.068266168
Epoch: 0036 cost = 0.042746723
Epoch: 0037 cost = 0.039906193
Epoch: 0038 cost = 0.038618356
Epoch: 0039 cost = 0.051107597
Epoch: 0040 cost = 0.045729216
Epoch: 0041 cost = 0.067414165
Epoch: 0042 cost = 0.038910475
Epoch: 0043 cost = 0.038644001
Epoch: 0044 cost = 0.039094109
Epoch: 0045 cost = 0.043331295
Epoch: 0046 cost = 0.037244216
Epoch: 0047 cost = 0.034457162
Epoch: 0048 cost = 0.037011944
Epoch: 0049 cost = 0.044235360
Epoch: 0050 cost = 0.036349982
Learning finished
Accuracy: 0.11349999904632568

Epoch: 0001 cost = 0.551087022
Epoch: 0002 cost = 0.257470459
Epoch: 0003 cost = 0.204459295
Epoch: 0004 cost = 0.176355377
Epoch: 0005 cost = 0.156724215
Epoch: 0006 cost = 0.138451248
Epoch: 0007 cost = 0.129061013
Epoch: 0008 cost = 0.118015580
Epoch: 0009 cost = 0.107403427
Epoch: 0010 cost = 0.101504572
Epoch: 0011 cost = 0.096441686
Epoch: 0012 cost = 0.093530126
Epoch: 0013 cost = 0.085378215
Epoch: 0014 cost = 0.083097845
Epoch: 0015 cost = 0.079341516
Learning finished
Accuracy: 0.4002000093460083
'''