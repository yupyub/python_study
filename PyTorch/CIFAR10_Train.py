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

"""
# 1-1. 학습용 이미지 확인(Visualize)
###########################################################
# 자세한 주석은 Tutorial_4_CIFAR10.py 참조
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
"""

# 추론 / 학습 재개를 위해 모델 불러오기
import sys
sys.path.insert(0, './DATA')
import NN_Model_2         # ___________________SET MODEL NAME________________________ #
MODELNAME = "NN_Model_2"  # ___________________SET MODEL NAME________________________ #
net = torch.load("./DATA/%s.pt" % (MODELNAME))

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
if os.path.isfile("./DATA/%s_cp.tar" % (MODELNAME)) :
	checkpoint = torch.load("./DATA/%s_cp.tar" % (MODELNAME))
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
    for epoch in range(2): # 데이터셋을 수차례 반복한다.
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
            }, "./DATA/%s_cp.tar" % (MODELNAME))
    print("Save CheckPoint")
