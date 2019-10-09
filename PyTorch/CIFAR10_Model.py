##########################################################################################
# 2. 합성곱 신경망(Convolution Neural Network)을 정의한다.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3-inChannel, 6-outChannels, 5-kernelSize
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# 참고 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html#checkpoint
# 전체 모델 저장하기
torch.save(net, "./DATA/CIFAR10_Model.pt")

"""
# 전체 모델 불러오기
# 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
model = torch.load(PATH)
model.eval()
"""



















