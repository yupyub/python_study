import threading as th 
def recur(count):
    if count < 10:
        count+=1
        print(count)
        timer = th.Timer(0.1,recur,args = [count]) #TimerCallback 함수에서, 매개면수는 args를 이용해서 리스트 타입으로 넘긴다
        timer.start()
        
print("Starting Timer...")
recur(0)

print("RUNNNN?") #recur(1) 실행 이전에 "RUNNNN?" 문구가 나온다, TimerCallback 함수의 실행으로 다음 recur(1)이 나오기전, 1초를 기다리면서 밑의 코드를 그대로 실행하기 때문