import numpy as np
import random

n=20
r=1
points=[]
s=100
steps=range(s)

for i in range(n):
    phi=random.random()*2*np.pi
    theta= random.random()*np.pi
    points.append([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])

