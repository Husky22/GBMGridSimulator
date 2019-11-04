import numpy as np
import random
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
randomize=True
samples=20
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

particles=np.array(points)
fig=plt.figure()
fig2=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax2=fig2.add_subplot(111,projection='3d')
ax2.scatter3D(particles[::,0],particles[::,1],particles[::,2])
velocities=np.zeros([3,len(particles)])
steps=np.linspace(0,1000,4000)
dt=0.05

def force_law(pos1,pos2):
    dist=np.linalg.norm(pos1-pos2,2)
    return (pos1-pos2)/dist * math.exp(-dist**2)
oldparticles=particles
oldparticles_temp=particles
it=0
histlist=[]
for step in steps:
    i=0
    minlist=[]
    for particle in particles:
        distlist=[]
        otherparticles=np.delete(particles,i,axis=0)
        force=np.zeros(3)
        for otherparticle in otherparticles:

            distance=np.linalg.norm(particle-otherparticle,2)
            distlist.append(distance)

            force=force+force_law(particle,otherparticle)
            minlist.append(np.amin(distlist))
        normalcomponent=particle/np.linalg.norm(particle,2)
        force=force-np.dot(normalcomponent,force)*normalcomponent
        if it==0:
            newp=particle+0.5*force*dt**2
            particles[i]=newp/np.linalg.norm(newp,2)
        else:
            oldparticles_temp[i]=particles[i]
            newp=2*particle-oldparticles[i]+force*dt**2
            particles[i]=newp/np.linalg.norm(newp,2)
            oldparticles[i]=oldparticles_temp[i]
        i+=1
    it+=1
    histlist.append(minlist)

nbins=50
# print(histlist[1])
# hist,bins=np.histogram(histlist[1],bins=50)
# print(hist)
# xs=(bins[:-1]+bins[1:])/2
# plt.bar(xs,hist,width=1/60)
k=0
for z in steps[0::100]:
    hist,bins=np.histogram(histlist[k],bins=nbins)
    xs=(bins[:-1]+bins[1:])/2

    ax.bar(xs,hist,zs=z,zdir='y',alpha=0.8,width=0.01)
    k+=1

ax2.scatter3D(particles[::,0],particles[::,1],particles[::,2])
plt.show()



