arr = [1,2,3]

try:
    print("input:",end = "")
    idx = int(input())
    print("in arr[{}]: {}".format(idx,arr[idx]))


except BaseException as err:
    print("It check all exeptions ({})".format(err))

except:
    print("ERROR")
    
else:
    print("Success!")
    
finally:
    print("it should be execute")
