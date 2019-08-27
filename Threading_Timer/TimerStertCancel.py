import threading as th

count = 0

def on_timer():
    global count
    count+=1
    print(count)
    timer =  th.Timer(0.1,on_timer)
    timer.start()
    if count == 10:
        print("Cancel Timer...")
        timer.cancel()
    
print("Starting Timer...")
on_timer();