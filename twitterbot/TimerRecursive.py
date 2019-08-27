import threading as th 
def recur(count):
    if count < 10:
        count+=1
        print(count)
        timer = th.Timer(1,recur,args = [count]) #args is list type
        timer.start()
        
print("Starting Timer...")
recur(0)

print("Timer End...")