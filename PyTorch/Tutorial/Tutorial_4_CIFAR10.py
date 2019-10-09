#########################################################################################
# <데이터 처리 도구 추천>
# 이미지 : Pillow, OpenCV
# 오디오 : SciPy, LibROSA
# 텍스트 : Cython, NLTK, SpaCy
# 일반적으로 Python 패키지를 사용해 NumPy배열로 불러오고, 이후 그 배열을 torch.*Tensor로 변환한다.
#########################################################################################
# 영상 분야의 경우 주로 torchvision 패키지를 이용한다.
# 여기에는 Imagenet이나 CIFAR10, MNIST 등과 같이 일반적으로 
# 사용하는 데이터셋을 위한 데이터 로더(data loader):torchvision.datasets 과 
# 이미지용 데이터변환기 (data transformer):torch.utils.data.DataLoader 가 포함되어 있다.
#########################################################################################
# 이 코드에서는 CIFAR10 데이터셋을 사용한다. 여기에는 
# ‘비행기(airplane)’, ‘자동차(automobile)’, ‘새(bird)’, ‘고양이(cat)’, ‘사슴(deer)’, 
# ‘개(dog)’, ‘개구리(frog)’, ‘말(horse)’, ‘배(ship)’, ‘트럭(truck)’. 으로 구성되어 있다.
# 그리고 CIFAR10에 포함된 이미지의 크기는 3x32x32로, 이는 32x32 픽셀 크기의 이미지가 
# 3개 채널(channel)의 색상로 이뤄져 있다는 것을 뜻한다.
##########################################################################################
# <이미지 분류기 학습과정>
# 1. torchvision 을 사용하여 CIFAR10의 학습용/시험용 데이터셋을 불러오고, 정규화(nomarlizing)한다.
# 2. 합성곱 신경망(Convolution Neural Network)을 정의한다.
# 3. 손실 함수를 정의한다.
# 4. 학습용 데이터를 사용하여 신경망을 학습시킨다.
# 5. 시험용 데이터를 사용하여 신경망을 검사한다.
##########################################################################################
# 1. torchvision 을 사용하여 CIFAR10의 학습용/시험용 데이터셋을 불러오고, 정규화(nomarlizing)한다.
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

##########################################################################################
# 1-1. 학습용 이미지 확인(Visualize)
###########################################################
# Matplotlib는 기본적으로 Xwindows 백엔드를 선택함으로
# Xwindows 백엔드를 사용하지 않도록 matplotlib를 설정
# 참조 : https://cnpnote.tistory.com/entry/PYTHON-tkinterTclError-%ED%91%9C%EC%8B%9C-%EC%9D%B4%EB%A6%84-%EC%97%86%EC%9D%8C-%EB%B0%8F-DISPLAY-%ED%99%98%EA%B2%BD-%EB%B3%80%EC%88%98-%EC%97%86%EC%9D%8C
# linux 환경이라 image display가 정상 작동하지 않는다.
# Xming 이나 VcXsrv등을 사용해 image를 밖으로 export 해야 한다
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
###########################################################
# import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수
def imshow(img):
	img = img/2+0.5	# unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
# 이미지 출력 (linux 환경이라 정상작동하지 않는다)
#imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
#print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

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
# CIFAR10_Model.py 에서 정의한 Model을 불러온다
import sys
sys.path.insert(0, './DATA')
import CIFAR10_Model
net = torch.load("./DATA/CIFAR10_Model.pt")
##########################################################################################
# 3. 손실 함수를 정의한다. + Optimizer정의하기
# 교차 엔트로피 손실(Cross-Entropy loss)과 모멘텀(momentum) 값을 갖는 SGD를 사용한다.
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# 추론 / 학습 재개를 위해 일반 체크포인트(checkpoint) 불러오기
# 참고 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html#checkpoint
import os 
if os.path.isfile("./DATA/CIFAR10_checkpoint.tar") :
	checkpoint = torch.load("./DATA/CIFAR10_checkpoint.tar")
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	net.eval()
	print("Load CheckPoint")
##########################################################################################
# 4. 학습용 데이터를 사용하여 신경망을 학습시킨다.
# checkpoint가 없는 경우만 학습을 시킨다
else :
	net.train()
	for epoch in range(3): # 데이터셋을 수차례 반복한다.
		running_loss = 0.0
		for i, data in enumerate(trainloader,0):
			inputs,labels = data
			# 변화도(Gradient) 매개변수를 0으로 초기화
			optimizer.zero_grad()
			# 순전파/역전파/최적화 실행
			outputs = net(inputs)
			loss = criterion(outputs,labels)
			loss.backward()
			optimizer.step()
			
			# 통계 출력
			running_loss += loss.item()
			if i%2000 == 1999: # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
				running_loss = 0.0
		
	print("Training Finish")
	
	# 추론 / 학습 재개를 위해 일반 체크포인트(checkpoint) 저장하기
	torch.save({
	            'epoch': epoch,
	            'model_state_dict': net.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            'loss': loss,
	            }, "./DATA/CIFAR10_checkpoint.tar")
	print("Save CheckPoint")
##########################################################################################
# 5. 시험용 데이터를 사용하여 신경망을 검사한다.
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지를 출력
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# 정확도 확인
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
