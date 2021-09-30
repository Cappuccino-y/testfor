import matplotlib.pyplot as plt
from numpy import *
x=linspace(0,2*pi,50)
plt.plot(x,sin(x),'bh')

x=random.rand(200)
y=random.rand(200)
size=random.rand(200)*30
color=random.rand(200,3)
plt.scatter(x,y,size,color)