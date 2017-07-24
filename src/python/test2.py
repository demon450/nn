import test

test.f1()

class test2:
    def __init__(self,x):
        self.x = x
        
    def getX(self):
        print(self.x)
        
t2 = test2("test2")
t2.getX()

a = 3;
a |= 10;
print(a)