"""
http://swlock.blogspot.com/2018/07/pytorch-autograd-tutorial.html
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
"""
# torch.nn 은 미니-배치(mini-batch)만 지원함. torch.nn 패키지 전체는 하나의 샘플이 아닌, 샘플들의 미니-배치만을 입력으로 받는다.
# 예를 들어, nnConv2D 는 nSamples x nChannels x Height x Width 의 4차원 Tensor를 입력으로 한다.
# 만약 하나의 샘플만 있다면, input.unsqueeze(0) 을 사용해서 가짜 차원을 추가
"""
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
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension * 16_channels
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
# print(net)
'''
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's weight
'''

input = torch.randn(1,1,32,32)
out = net(input)
#net.zero_grad()
#out.backward(torch.randn(1,10))
output = net(input)
target = torch.randn(10) # dummy target, for example
target = target.view(1,-1) # make it the dame shape as output
criterion = nn.MSELoss();
loss = criterion(output,target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
