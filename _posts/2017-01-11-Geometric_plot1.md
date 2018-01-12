---
layout: post
title: Nomad Transparent Conducting Oxide Kaggle Part 1
subtitle: Vornoi Tesselation and Formation Energy Prediction
---

Recently, a high quality dataset of formation energies and bandgaps for a family of sesquioxide type materials was posted on Kaggle by the Fritz Haber Institute of the Max Planck Society with calculated material parameters and unit cell structures ([link](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/)).  For those not in a Materials Science or Chemistry related field, transparent conducting oxides (TCOs) are a crucial class of functional materials which combine optical transparency with electrical conductivity.  These materials are used to enable every from capacity sensing in the touchscreens of smartphones, to the top electrode of solar cell devices.  Given that such devices are so widely employed, and there exists considerable demand for even more, there is a pressing need to develop new classes of materials which have favorable properties and are relatively inexpensive.  Suitable TCOs today rely on rare earth elements, whose price and supply can fluctuate wildly. 

In this competition, we are challenged to produce a computionally efficient method for predicting the optical performance (parameterized by the band gap) and the stability (represented by the formation energy) of candidate sesquioxide materials based on their stoichiometric ratios and unit cell structures.  Analytical methods for solving for these parameters do not exist, due to the need to solve a many body Schrödinger equation.  Approximations for these methdos based on clever atomic potentials are also inordinately computationally expensive.  Thus, computationally tractable and efficient methods for ranking candidate materials by their properties would go a long way to accelerating research in this field.

In this blog, I'll give some hints about working with geometries files and extracting meaningful material feature based on these three-dimensional structures.  I'll mostly be following along with a recent publication by a colleague of mind, which you can find [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.024104).

Let's start by importing some helper libraries and the main .csv file, which contains a structure index and the stoichiometry of the structure. Then we'll load a geometry file for one particular structure, which contains the basis vectors which describe that structure and the atomic positions (in x,y,z).


```python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
import plotly as plyt



train=pd.read_csv('./train.csv')

def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

idx = train.id.values[100]
fn = "./train/{}/geometry.xyz".format(idx)
train_xyz, train_lat = get_xyz_data(fn)

Indium = go.Scatter3d(
    x=[i[0][0] for i in train_xyz if i[1]=='In'],
    y=[i[0][1] for i in train_xyz if i[1]=='In'],
    z=[i[0][2] for i in train_xyz if i[1]=='In'],
    mode='markers',
    name='Indium',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(90,180,172, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

Oxygen = go.Scatter3d(
    x=[i[0][0] for i in train_xyz if i[1]=='O'],
    y=[i[0][1] for i in train_xyz if i[1]=='O'],
    z=[i[0][2] for i in train_xyz if i[1]=='O'],
    mode='markers',
    name='Oxygen',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(216,179,101, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

Aluminum = go.Scatter3d(
    x=[i[0][0] for i in train_xyz if i[1]=='Al'],
    y=[i[0][1] for i in train_xyz if i[1]=='Al'],
    z=[i[0][2] for i in train_xyz if i[1]=='Al'],
    mode='markers',
    name='Aluminum',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(166,97,26, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

Gallium = go.Scatter3d(
    x=[i[0][0] for i in train_xyz if i[1]=='Ga'],
    y=[i[0][1] for i in train_xyz if i[1]=='Ga'],
    z=[i[0][2] for i in train_xyz if i[1]=='Ga'],
    mode='markers',
    name='Gallium',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(128,205,193, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)



data = [Indium, Oxygen, Aluminum, Gallium]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='unitcell_orig')

```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~khujsak/31.embed" height="525px" width="100%"></iframe>



Above, I have plotted the position of each atom in the unit cell according to its physical position.  Feel free to rotate the plot around and get a feeling for the structure and the symmetry.  Individual elements are color coded and can be activated/deactivated as desired.

The unit cells themselves may not occupy a perfect square, in fact this is very uncommon!  To deal with oblique shapes and standardize the tesselation for many different sized unit cells in real space, we need to convert them to a common range.  Fortunately, the lattice vectors supply a basis for these structures in which the atoms extend from zero to one along each of the three basis vectors.  Transforming our structures to this space will allow us to create periodic examples of our unit cells.  Since real materials are composed of very large sets of this repeated unit cell structure, to accurately model the geometry we need to accomodate this periodicity.

As mentioned above, we are given the three crystal lattice vectors, which together can form a basis for the crystal :$(a_1, a_2, a_3)^T$.  We can describe the position of an atom in the crystal $R(x,y,z)$, where x,y,z are the real space coordinates, in terms of the basis vectors themselves:

$$R(x,y,z)=a_1+a_2y+a_3z$$

Taking the inverse of the basis vector matrix, we can solve for the reduced form coordinates:

$$r(x,y,z)=A^{-1}R(x,y,z)$$


```python
from numpy.linalg import inv

A=np.transpose(train_lat)
B = inv(A)

xyz=[]

for i in enumerate(train_xyz):
    
    r = np.matmul(B, i[1][0])
    xyz.append(r)
    

xyz=np.array(xyz)
```


```python

trace1 = go.Scatter3d(
    x=xyz[:,0],
    y=xyz[:,1],
    z=xyz[:,2],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)


data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~khujsak/21.embed" height="525px" width="100%"></iframe>




```python
min_x=np.min(xyz[:,0])
max_x=np.max(xyz[:,0])
min_y=np.min(xyz[:,1])
max_y=np.max(xyz[:,1])
min_z=np.min(xyz[:,2])
max_z=np.max(xyz[:,2])

print(min_x, max_x, min_y, max_y, min_z, max_z)


```

    -7.11507675694e-20 0.966 -5.42101086243e-20 0.966 -1.08420217249e-19 0.966
    

As promised, the reduced coordinates fill a unit square parameterized by the lattice vectors of the crystal.  This will allow us to build periodic arrays of the unit cell simply by stacking unit squares.  Frequently, when considering the formation energy of a crystal, we will be interested in the relative energies and stabilities of all the bonds represented within the crystal.  As such, we will need some way to graphically describe the relative neighborhood each atom has available within the larger cell, and how that neighborhood compares to the local environment of other face-sharing atoms.  The method we will use to construct an estimate of this local environment is voronoi tesselation.  Basically, we will construct a set of vectors from each atom to its neighboring cells, and take the perpindicular bisectors.  The cell traced out by the perpindicular bisectors will represent the relative available volume.  This technique has a long history in materials science, consider Wigner-Seitz cells!

We will use a python package tess, which is a pythonic interface to the voro++ package.  This package has built in support for periodic cells.


```python
import tess as ts

cntr=ts.Container(xyz, limits=((0,0,0),(1,1,1)), periodic=True)
```


```python
test=[v.centroid() for v in cntr]
test=np.array(test)
np.shape(test)
```




    (80, 3)






As expected, for an 80 atom structure we have produced a tesselation with 80 unique cells.

We will now pick three random cells, [0,1,15] and plot their volume in 3D.


```python
test=[v.vertices() for v in cntr]

test=np.array(test)

np.shape(test)


test0=np.array(test[0])
test1=np.array(test[1])
test2=np.array(test[15])

```


```python
import random
r = lambda: random.randint(0,255)
```


```python

trace0 = go.Mesh3d(x=test0[:,0],
                   y=test0[:,1],
                   z=test0[:,2],
                   alphahull=0,
                   opacity=0.4,
                   color='#%02X%02X%02X' % (r(),r(),r()))


trace1 = go.Mesh3d(x=test1[:,0],
                   y=test1[:,1],
                   z=test1[:,2],
                   alphahull=0,
                   opacity=0.4,
                   color='#%02X%02X%02X' % (r(),r(),r()))


trace2 = go.Mesh3d(x=test2[:,0],
                   y=test2[:,1],
                   z=test2[:,2],
                   alphahull=0,
                   opacity=0.4,
                   color='#%02X%02X%02X' % (r(),r(),r()))


py.iplot([trace0, trace1, trace2], filename = 'voronoi surface')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~khujsak/25.embed" height="525px" width="100%"></iframe>



And now we'll plot all of the cells!


```python
trace=[]
i=0
for cell in test:
    tmp=np.array(cell)
    trace.append(go.Mesh3d(x=tmp[:,0],
                   y=tmp[:,1],
                   z=tmp[:,2],
                   alphahull=0,
                   opacity=0.4,
                   color='#%02X%02X%02X' % (r(),r(),r())))
    

py.iplot(trace, filename = 'voronoi surface2')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~khujsak/27.embed" height="525px" width="100%"></iframe>



Note that the space is still in reduced coordinates!  In order to interpret the information in our vornoi cell and compare it to tesselations of other structures (with different lattice vectors), we will need to convert to a common set of coordinates: real space.  To do this, we'll simply multiply by the original crystal lattice vectors, as described above.


```python
trace=[]
i=0
for cell in test:
    tmp=np.array(cell)
    
    for i in range(np.shape(tmp)[0]):
        tmp[i,:]=np.matmul(A, tmp[i,:])
    
    trace.append(go.Mesh3d(x=tmp[:,0],
                   y=tmp[:,1],
                   z=tmp[:,2],
                   alphahull=0,
                   opacity=0.4,
                   color='#%02X%02X%02X' % (r(),r(),r())))
   

    

py.iplot(trace, filename = 'voronoi surface2_realspace')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~khujsak/29.embed" height="525px" width="100%"></iframe>



Finally, we get our vornoi tesselated cells back in real space coordinates and plot the results.  As you can see, this method neatly handles oblique and oddly shaped unit cells while accounting for periodicity.  In the next following posts I will demonstrate how one can engineer features from these cells!
