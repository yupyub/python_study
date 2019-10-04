"""
< NN 학습 과정 >
1. 학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의
2. 데이터셋(dataset) 입력을 반복
3. 입력을 신경망에서 전파(process)
4. 손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지)을 계산
5. 변화도(gradient)를 신경망의 매개변수들에 역으로 전파
6. 신경망의 가중치를 갱신 
    <가중치(wiehgt) = 가중치(weight) - 학습율(learning rate) * 변화도(gradient)>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 예상되는 입력 크기는 32x32
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3) # 1-inChannel, 6-outChannels, 3-kernelSize
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
        # forward함수만 정의하면, backward 함수는 autograd를 사용하여 자동으로 정의됨
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
