import torch


'''
a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
print(a.grad_fn) # Not Exist
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)
'''
"""
x = torch.ones(2,2,requires_grad = True)
y = x+2

print(y)
print(x.grad_fn) # 사용자 정의 Tensor 이므로 .grad_fn 은 None 이다.
print(y.grad_fn)
print(y*y)

z = y*y*3 # (1+2)*(1+2)*3=27
out = z.mean()
#print(z,out)

out.backward() # same as out.backward(torch.tensor(1.)) (out 이 1개의 원소만을 갖기 때문)
print(x.grad) # print d(out)/dx
"""

# Gradients
x = torch.randn(3,requires_grad = True)
y=  x*2
while y.data.norm() < 1000:
    y = y*2
print(y)
gradients = torch.tensor([0.1,1.0,0.0001],dtype = torch.float)
y.backward(gradients)
print(x.grad)
print(x.requires_grad)
print((x**2).requires_grad)
# with torch.no_grad(): 로 코드 블럭을 감싸서 autograd가 .requires_grad=True 인 Tensor들의 연산 기록을 추적하는 것을 멈출 수 있다.
with torch.no_grad() :  
    print((x**2).requires_grad)

    