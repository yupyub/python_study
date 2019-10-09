import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
import sys
sys.path.insert(0, './DATA')
import NN_Model_1         # ___________________SET MODEL NAME________________________ #
MODELNAME = "NN_Model_1"  # ___________________SET MODEL NAME________________________ #
net = torch.load("./DATA/%s.pt" % (MODELNAME))

##########################################################################################
# 모델/체크포인트(checkpoint) 불러오기
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
else :
    print("Trained Model doesn't exist")
    print("Try 'CIFAR10_Train.py' First")
    exit()


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