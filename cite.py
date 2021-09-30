import time
from hhh.asdad import hello
from Card import FrenchDeck

print (time.ctime())

a,b = hello.sayhello()

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

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):
        print("person is talking....")


class Chinese(Person):

    def __init__(self,name,age,language):
        super().__init__(name,age)
        self.name='asd'
        self.age=age

        self.language=language

    def walk(self):
        print('is walking...')

    def __call__(self, s):
        print(s)

m=Chinese('Chinese',12,'GSD')
n=123

