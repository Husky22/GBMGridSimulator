import math
import random


def fibonacci_sphere(samples=1,randomize=True):
    rnd=1.
    if randomize:
        rnd=random.random()*samples
    points=[]
    offset=2./samples
    increment=math.pi*(3.-math.sqrt(5.))

    for i in range(samples):
        y=((i*offset)-1)+(offset/2)
        r=math.sqrt(1-y**2)
        phi=((i+rnd)%samples)*increment
        x=math.cos(phi)*r
        z=math.sin(phi)*r
        points.append([x,y,z])

    return points

print fibonacci_sphere(20)
