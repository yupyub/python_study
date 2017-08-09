class aka:
    tot = 0
    __tot__ = 0
    def __init__(self):
        self.one = 0
    def add(self,num):
        #print("self : ",self)
        self.one += num
    @classmethod
    def add_tot(cls,num):
        #print("cls : ",cls)
        cls.tot += num
    @classmethod
    def p_cls(cls):
        print(cls.tot)
    @staticmethod
    def p_whole():
        print(aka.tot)
        print(a.one)
    
if __name__ == "__main__":
    a = aka()
    a.add(10)
    a.add_tot(100)
    b = aka()
    b.add(2)
    b.add_tot(2)
    #print(b.tot)
    #a.p_cls()
    #aka.p_whole()
    print(a.__init__)

