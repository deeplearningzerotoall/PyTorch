import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision

from model import model


if __name__ == "__main__":

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,128)),
        torchvision.transforms.ToTensor()
    ])
    test_data = torchvision.datasets.ImageFolder(root='./test_data', transform=trans)

    testloader = DataLoader(dataset=test_data, batch_size=8, shuffle=True, num_workers=4)
    
    if(torch.cuda.is_available()):
        device='cuda'
    else:
        device='cpu'

    pre_train_net = model.NN()
    pre_train_net.load_state_dict(torch.load('./model/model.pth',map_location=device))

    pre_train_net = pre_train_net.to(device)

    # 몇개 맞았는지 저장할 변수
    correct = 0
    # 전체 개수를 저장할 변수
    total = 0

    for num, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(device)

        out = pre_train_net(inputs)
        _, predicted = torch.max(out,1)

        #torch.Tensor.cuda()하고 torch.Tensor()는 비교가 안됩니다.
        #따라서 .cpu() 를 이용해서 바꿔주세요
        predicted = predicted.cpu()
        total += labels.size(0)

        #잘 맞추고 있는지 궁금하면 아래 print를 출력해 보세요
        #print(predicted, labels)

        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 50 test images : %d %%'%(100* correct /total))
