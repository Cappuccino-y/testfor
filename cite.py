import time

print (time.ctime())


def test(name,age=18,args='sad'):
    print(name)
    print(age)
    print(args)


test('lyy',**{'age':12,'args':13})

def f(x,title):
    print(title)
    return pow(x,2)
f(2,title='asd')

class Person(object):

    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def talk(self,w):
        print("person is talking {0:.2f}".format(w))



class Chinese(Person):

    def __init__(self,age,name,weight,language):
        super(Chinese, self).__init__(name,age,weight)
        self.language=language

    def walk(self):
        print('is walking...')

    def __call__(self, s):
        print(s)


class FYI():
    def __init__(self):
        self.x=Chinese(32,'LYY',12,'c')
    def __call__(self,m):
        return self.x(m)

Q=FYI()