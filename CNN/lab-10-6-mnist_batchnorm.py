# Lab 10 MNIST and softmax
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# for reproducibility
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

# parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 32

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
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

# MNIST data image of shape 28 * 28 = 784
# nn layers
linear1 = torch.nn.Linear(784, 32, bias=True)
linear2 = torch.nn.Linear(32, 32, bias=True)
linear3 = torch.nn.Linear(32, 10, bias=True)
relu = torch.nn.ReLU()
bn1 = torch.nn.BatchNorm1d(32)
bn2 = torch.nn.BatchNorm1d(32)

nn_linear1 = torch.nn.Linear(784, 32, bias=True)
nn_linear2 = torch.nn.Linear(32, 32, bias=True)
nn_linear3 = torch.nn.Linear(32, 10, bias=True)

# model
bn_model = torch.nn.Sequential(linear1, relu, bn1,
                            linear2, relu, bn2,
                            linear3).to(device)
nn_model = torch.nn.Sequential(nn_linear1, relu,
                               nn_linear2, relu,
                               nn_linear3).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

# Save Losses and Accuracies every epoch
# We are going to plot them later
train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

train_total_batch = len(train_loader)
test_total_batch = len(test_loader)
for epoch in range(training_epochs):
    bn_model.train()  # set the model to train mode

    for X, Y in train_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        bn_optimizer.zero_grad()
        bn_prediction = bn_model(X)
        bn_loss = criterion(bn_prediction, Y)
        bn_loss.backward()
        bn_optimizer.step()

        nn_optimizer.zero_grad()
        nn_prediction = nn_model(X)
        nn_loss = criterion(nn_prediction, Y)
        nn_loss.backward()
        nn_optimizer.step()

    with torch.no_grad():
        bn_model.eval()     # set the model to evaluation mode

        # Test the model using train sets
        bn_loss, nn_loss, bn_acc, nn_acc = 0, 0, 0, 0
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            bn_prediction = bn_model(X)
            bn_correct_prediction = torch.argmax(bn_prediction, 1) == Y
            bn_loss += criterion(bn_prediction, Y)
            bn_acc += bn_correct_prediction.float().mean()

            nn_prediction = nn_model(X)
            nn_correct_prediction = torch.argmax(nn_prediction, 1) == Y
            nn_loss += criterion(nn_prediction, Y)
            nn_acc += nn_correct_prediction.float().mean()

        bn_loss, nn_loss, bn_acc, nn_acc = bn_loss / train_total_batch, nn_loss / train_total_batch, bn_acc / train_total_batch, nn_acc / train_total_batch

        # Save train losses/acc
        train_losses.append([bn_loss, nn_loss])
        train_accs.append([bn_acc, nn_acc])
        print(
            '[Epoch %d-TRAIN] Batchnorm Loss(Acc): bn_loss:%.5f(bn_acc:%.2f) vs No Batchnorm Loss(Acc): nn_loss:%.5f(nn_acc:%.2f)' % (
            (epoch + 1), bn_loss.item(), bn_acc.item(), nn_loss.item(), nn_acc.item()))
        # Test the model using test sets
        bn_loss, nn_loss, bn_acc, nn_acc = 0, 0, 0, 0
        for i, (X, Y) in enumerate(test_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            bn_prediction = bn_model(X)
            bn_correct_prediction = torch.argmax(bn_prediction, 1) == Y
            bn_loss += criterion(bn_prediction, Y)
            bn_acc += bn_correct_prediction.float().mean()

            nn_prediction = nn_model(X)
            nn_correct_prediction = torch.argmax(nn_prediction, 1) == Y
            nn_loss += criterion(nn_prediction, Y)
            nn_acc += nn_correct_prediction.float().mean()

        bn_loss, nn_loss, bn_acc, nn_acc = bn_loss / test_total_batch, nn_loss / test_total_batch, bn_acc / test_total_batch, nn_acc / test_total_batch

        # Save valid losses/acc
        valid_losses.append([bn_loss, nn_loss])
        valid_accs.append([bn_acc, nn_acc])
        print(
            '[Epoch %d-VALID] Batchnorm Loss(Acc): bn_loss:%.5f(bn_acc:%.2f) vs No Batchnorm Loss(Acc): nn_loss:%.5f(nn_acc:%.2f)' % (
                (epoch + 1), bn_loss.item(), bn_acc.item(), nn_loss.item(), nn_acc.item()))
        print()

print('Learning finished')

def plot_compare(loss_list: list, ylim=None, title=None) -> None:
    bn = [i[0] for i in loss_list]
    nn = [i[1] for i in loss_list]

    plt.figure(figsize=(15, 10))
    plt.plot(bn, label='With BN')
    plt.plot(nn, label='Without BN')
    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)
    plt.legend()
    plt.grid('on')
    plt.show()

plot_compare(train_losses, title='Training Loss at Epoch')
plot_compare(train_accs, [0, 1.0], title='Training Acc at Epoch')
plot_compare(valid_losses, title='Validation Loss at Epoch')
plot_compare(valid_accs, [0, 1.0], title='Validation Acc at Epoch')


'''
[Epoch 1-TRAIN] Batchnorm Loss(Acc): 0.14542(95.64%) vs No Batchnorm Loss(Acc): 0.20199(94.24%)
[Epoch 1-VALID] Batchnorm Loss(Acc): 0.15192(95.51%) vs No Batchnorm Loss(Acc): 0.19679(94.32%)

[Epoch 2-TRAIN] Batchnorm Loss(Acc): 0.12063(96.29%) vs No Batchnorm Loss(Acc): 0.16021(95.46%)
[Epoch 2-VALID] Batchnorm Loss(Acc): 0.14427(95.68%) vs No Batchnorm Loss(Acc): 0.18920(94.92%)

[Epoch 3-TRAIN] Batchnorm Loss(Acc): 0.09637(97.13%) vs No Batchnorm Loss(Acc): 0.14353(95.82%)
[Epoch 3-VALID] Batchnorm Loss(Acc): 0.12341(96.17%) vs No Batchnorm Loss(Acc): 0.16541(95.25%)

[Epoch 4-TRAIN] Batchnorm Loss(Acc): 0.08518(97.42%) vs No Batchnorm Loss(Acc): 0.13275(96.20%)
[Epoch 4-VALID] Batchnorm Loss(Acc): 0.11416(96.64%) vs No Batchnorm Loss(Acc): 0.16461(95.56%)

[Epoch 5-TRAIN] Batchnorm Loss(Acc): 0.07560(97.66%) vs No Batchnorm Loss(Acc): 0.12823(96.36%)
[Epoch 5-VALID] Batchnorm Loss(Acc): 0.11083(96.84%) vs No Batchnorm Loss(Acc): 0.17232(95.55%)

[Epoch 6-TRAIN] Batchnorm Loss(Acc): 0.07636(97.66%) vs No Batchnorm Loss(Acc): 0.13570(96.22%)
[Epoch 6-VALID] Batchnorm Loss(Acc): 0.10608(96.62%) vs No Batchnorm Loss(Acc): 0.17844(95.37%)

[Epoch 7-TRAIN] Batchnorm Loss(Acc): 0.06865(97.89%) vs No Batchnorm Loss(Acc): 0.12285(96.61%)
[Epoch 7-VALID] Batchnorm Loss(Acc): 0.10543(96.96%) vs No Batchnorm Loss(Acc): 0.16415(96.04%)

[Epoch 8-TRAIN] Batchnorm Loss(Acc): 0.06668(97.93%) vs No Batchnorm Loss(Acc): 0.10810(97.03%)
[Epoch 8-VALID] Batchnorm Loss(Acc): 0.10785(96.78%) vs No Batchnorm Loss(Acc): 0.16704(95.98%)

[Epoch 9-TRAIN] Batchnorm Loss(Acc): 0.06295(97.99%) vs No Batchnorm Loss(Acc): 0.12544(96.61%)
[Epoch 9-VALID] Batchnorm Loss(Acc): 0.10589(96.66%) vs No Batchnorm Loss(Acc): 0.20841(95.65%)

[Epoch 10-TRAIN] Batchnorm Loss(Acc): 0.05624(98.25%) vs No Batchnorm Loss(Acc): 0.10978(97.05%)
[Epoch 10-VALID] Batchnorm Loss(Acc): 0.10229(96.95%) vs No Batchnorm Loss(Acc): 0.18756(95.80%)

Learning finished
'''