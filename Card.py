import collections
import random
import math
import os
import numpy as np
Card = collections.namedtuple('Card', 'rank suit')

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()
c=random.choice(deck)
suit_value=dict(spades=3,hearts=2,diamonds=1,clubs=0)

def spades_high(card):
    rank_value=FrenchDeck.ranks.index(card.rank)
    return rank_value*len(suit_value)+suit_value[card.suit]

# for card in sorted(deck,key=lambda x:spades_high(x)):
#     print(card)

class Vector:
    def __init__(self,x=0,y=-0):
        self.x=x
        self.y=y
    def __repr__(self):
        return 'Vector({},{})'.format(self.x,self.y)
    def __abs__(self):
        return math.hypot(self.x,self.y)
    def __bool__(self):
        return bool(abs(self))
    def __add__(self, other):
        return Vector(self.x+other.x,self.y+other.y)
    def __mul__(self, scalar):
        return Vector(self.x*scalar,self.y*scalar)

colors=['b','w']
sizes=['S','M','L']
a=tuple('%s %s\n' %(c,s) for c in colors for s in sizes)
_,filename= os.path.split('/asd/asdas/asd.cs')

def asd(*state):
        print(state)
        return state
p=asd(1,2,3)
a,b,*rest=range(5)
fmt='{:15}|{:9.4f}|{:9.4f}'
print(fmt.format('Sao Paul',-23.54,-74.02))

example1=dict(location='beijing',university='beihang')
cd=example1.items()

zarten = np.array([[1,2,3,4], [4,5,6,7], [7,8,9,10]])
print(zarten[[0,2],[0,1]])
