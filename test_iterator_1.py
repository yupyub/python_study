class MyIter:
    def __init__(self,start,end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current<self.end:
            current = self.current
            self.current+=1
            return current
        else:
            raise StopIteration()

def MyGener(start, end):
    current = start
    while current < end:
        yield current
        current += 1

    while current > start:
        yield current
        current -= 1
    return 

for i in MyIter(1,5):
    print(i)

for i in MyGener(1,5):
    print(i)


    
