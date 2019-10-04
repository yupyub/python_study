from __future__ import print_function
import torch

#x = torch.empty(5,3) # un initialized
#x = torch.rand(5,3) # random 
#x = torch.zeros(5,3,dtype=torch.long) # zeros & type long

'''
x = torch.tensor([5.5,3])
x = x.new_ones(5,3,dtype=torch.double) # new_* 메소드는 크기를 받는다
print(x)
x = torch.randn_like(x, dtype=torch.float) #dtype 를 Override
print(x)
print(x.size())
'''
"""
x = torch.rand(5,3)
y = torch.rand(5,3)
print(x,"\n",y,"\n",x+y)
print(torch.add(x,y))
torch.add(x,y,out = result)
print(result)
y.add_(x) # 바꿔치기(InPlace) 연산들은 _를 접미사로 갖는다
print(y)
"""
'''
x = torch.rand(5,3)
print(x)
print(x[:,1])
'''
"""
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) # -1은 다른 차원들을 사용해 유추함
print(x)
print(y)
print(z)
print(x.size(), y.size(), z.size())
"""
'''
x = torch.randn(1)
print(x)
print(x.item())
'''
""" 
# tensor -> numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1) # Torch Tensor와 NumPy 배열은 저장 공간을 공유
print(a)
print(b) 
"""
'''
# numpy -> tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
'''

"""
# 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
# ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA 장치 객체(device object)로
    y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
    x = x.to(device)                       # ``.to("cuda")`` 를 사용하면 됩니다.
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!
"""