class MyException(Exception):
    def __init__(self,arg):
        super().__init__("MyException is occured : {}".format(arg))

def convert_to_integer(text):

    if text.isdigit(): # 부호 (+,-) 처리 못함
        return int(text)
    else:
        raise MyException(text)

if __name__ == "__main__":
    try :
        print("input : ",end = "")
        text = input()
        num = convert_to_integer(text)
    except MyException as err:
        print("Error : {} ".format(err))

    else :
        print("It is converted integer type: {} ({})".format(num,type(num)))

