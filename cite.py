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
        self.weight = 'weight'

    def talk(self):
        print("person is talking....")



class Chinese(Person):

    def __init__(self,name,age,language,weight):


        self.language=language

    def walk(self):
        print('is walking...')

    def __call__(self, s):
        print(s)


m=Chinese('Chinese',12,'GSD',0)
n=123

class FYI():
    def __init__(self):
        self.x=Chinese('Chinese',12,'GSD',12)
    def trial(self):
        n=self.x(2)




Q=FYI()