"""
file = open("test.txt","w")
file.write("hello")
file.close()
"""

with open("test.txt","r") as file : # 코드블록 시작 전, 컨스트럭트매니저.__enter__() 호출
    str = file.read()
    print(str)
    # file.close()  ->  with 구문에서는 블록 종료시 자동으로 close()를 호출해 준다 // 컨스트럭트매니저.__exit__() 호출
    

class open2(object):
    def __init__(self,path):
        print("initialized")
        self.file = open(path)    # open function default : "rt" = read by text mode
        
    def __enter__(self):
        print("entered")
        return self.file
    
    def __exit__(self, ext, exv, trb):
        print("exited")
        self.file.close()
        return True
    
with open2("test.txt") as file:
    str = file.read()
    print(str)

    
    
""" using decorator """

from contextlib import contextmanager as ctx

@ctx
def open3(path):
    print("opening file...")
    file = open(path)
    try:    # same as __enter__
        print("yield file...")
        yield file
    finally:    # same as __exit__
        print("closing file...")
        file.close()
        
with open3("test.txt") as file:
    str2 = file.read()
    print(str2)
