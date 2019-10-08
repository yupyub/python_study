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
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
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
# 이미지 출력
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))




