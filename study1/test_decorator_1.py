"""
def aka(func):
    def do():
        func()
@aka
def aa():
    print("it's aa")


aa()

"""

def decorator_function(original_function):
    #def wrapper_function():
    print("{} 함수가 호출되기 전입니다.".format(original_function.__name__))
    #return original_function()
    return original_function
    #return wrapper_function
 
@decorator_function
def display_1():
    print("display_1 함수가 실행됐습니다.")
 
@decorator_function
def display_2():
    print("display_2 함수가 실행됐습니다.")
 
# display_1 = decorator_function(display_1) 
# display_2 = decorator_function(display_2) 
 
display_1() # 변수에 함수형 리턴값 할당 없이 바로 함수 호출이 가능하다. 
display_2() # 변수에 함수형 리턴값 할당 없이 바로 함수 호출이 가능하다.


"""
def decorator_function(original_function):
    def wrapper_function():
        print("{} 함수가 호출되기 전입니다.".format(original_function.__name__))
        return original_function()
    return wrapper_function
 
def display_1():
    print("display_1 함수가 실행됐습니다.")
 
def display_2():
    print("display_2 함수가 실행됐습니다.")
 
display_1 = decorator_function(display_1)
display_2 = decorator_function(display_2)
 
display_1()
print("")
display_2()
##########################################################################
def decorator_function(original_function):
    def wrapper_function():
        print("{} 함수가 호출되기 전입니다.".format(original_function.__name__))
        return original_function()
    return wrapper_function
 
@decorator_function
def display_1():
    print("display_1 함수가 실행됐습니다.")
 
@decorator_function
def display_2():
    print("display_2 함수가 실행됐습니다.")
 
# display_1 = decorator_function(display_1) 
# display_2 = decorator_function(display_2) 
 
display_1() # 변수에 함수형 리턴값 할당 없이 바로 함수 호출이 가능하다. 
display_2() # 변수에 함수형 리턴값 할당 없이 바로 함수 호출이 가능하다.

"""
