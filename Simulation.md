
# Table of Contents

1.  [Setting up the Simulation](#orgbd37c64)
    1.  [Random Distribution of Points on a sphere](#org8c5137c)
    2.  [Evenly Distribution by Fibonacci Sphere Algorithm](#org49c150b)
    3.  [Spacecraft coordinates](#org59de673)
2.  [The Coordinate Systems](#org043f479)
    1.  [ICRS](#org6b075de)
    2.  [GBM](#orgb01518e)



<a id="orgbd37c64"></a>

# Setting up the Simulation


<a id="org8c5137c"></a>

## Random Distribution of Points on a sphere

Choose <img src="ltximg/Simulation_be26b572b10e75f2aee8d8b352ffdb630ea2b8bd.png" alt="Simulation_be26b572b10e75f2aee8d8b352ffdb630ea2b8bd.png" /> and <img src="ltximg/Simulation_50a877075ef7068017c4d7b3e16bc8dad25a0099.png" alt="Simulation_50a877075ef7068017c4d7b3e16bc8dad25a0099.png" /> from uniform distribution in <img src="ltximg/Simulation_8a401eaba6a0fc3840464c4cb32143f89a965491.png" alt="Simulation_8a401eaba6a0fc3840464c4cb32143f89a965491.png" />
To obtain the angles use formulas:
<img src="ltximg/Simulation_d0714118c873597063f40e6db3cb93184e6dbcb0.png" alt="Simulation_d0714118c873597063f40e6db3cb93184e6dbcb0.png" />


<div class="equation-container">
<span class="equation">
<img src="ltximg/Simulation_7b63972420d28a0f726b9faf5b234249547d6874.png" alt="Simulation_7b63972420d28a0f726b9faf5b234249547d6874.png" />
</span>
<span class="equation-label">
1
</span>
</div>

Source: [MathWorld](http://mathworld.wolfram.com/SpherePointPicking.html)


<a id="org49c150b"></a>

## Evenly Distribution by Fibonacci Sphere Algorithm

[StackOverflow Source](https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere)

[Modified Fibonacci Source](http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/) (Modified Fibonacci Broken)


<a id="org59de673"></a>

## Spacecraft coordinates

From Trigdata we can get

    RA_SCX  =             198.1504 / [deg] Pointing of spacecraft x-axis: RA        
    DEC_SCX =               2.8766 / [deg] Pointing of spacecraft x-axis: Dec       
    RA_SCZ  =             279.3483 / [deg] Pointing of spacecraft z-axis: RA        
    DEC_SCZ =             -71.8215 / [deg] Pointing of spacecraft z-axis: Dec  

[SkyCoord Introduction of Spacecraft Coordinates](file:///home/niklas/venv/lib/python2.7/site-packages/gbmgeometry-0.1.2-py2.7.egg/gbmgeometry/gbm.py)
Usage of icrs coordinate system

[J2000 and GBM transformation](file:///home/niklas/venv/lib/python2.7/site-packages/gbmgeometry-0.1.2-py2.7.egg/gbmgeometry/gbm_frame.py)


<a id="org043f479"></a>

# The Coordinate Systems


<a id="org6b075de"></a>

## ICRS


<a id="orgb01518e"></a>

## GBM

We need quaternions and spacecraft position from trigdat

