# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 22:22:16 2018
Transductive/Unsupervised SVM for classification (simple concept implementation)
@author: afsar
"""

import autograd.numpy as np
from autograd import grad
import itertools
import matplotlib.pyplot as plt
def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    if clf is not None:
        npts = 100
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z,(npts,npts))
        
        extent = [minx,maxx,miny,maxy]
        plt.imshow(z,vmin = -2, vmax = +2)    
        plt.contour(z,[-1,0,1],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])   
    if Y is not None:
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        plt.show()   
		
def primal(weights):
    """
    This is the primal objective function
    """
    w = weights[:-1]
    b = weights[-1]
    score = np.dot(inputs,w)+b#    
    obj = 0.5*lambdaa*np.linalg.norm(w)**2
    loss = np.mean(np.exp(-3*(score**2))) #the Symmetric Sigmoid loss from "LARGE SCALE TRANSDUCTIVE SVMS"  by Collobert 2006
    obj+=loss
    return obj
def decision_function(x):
	return np.dot(x,w)+b

if __name__=='__main__':
    # Setup some training data
    Xp = 1.5+np.random.randn(100,2); Xn = -1.5-np.random.randn(100,2);
    inputs = np.vstack((Xp,Xn))	
    targets = np.hstack((np.ones(len(Xp)),-np.ones(len(Xn))))
    
    lambdaa = 0.01 #regularization parameter
    lrate = 0.01 #learning rate
    T = 100 #number of epochs
    # Define a function that returns gradients of training loss using Autograd.
    training_gradient_fun = grad(primal)
    L = []
    # Optimize weights using gradient descent.
    weights = np.random.rand(inputs.shape[1]+1)
    print("Initial loss:", primal(weights))    
    for i in range(T):
        weights -= training_gradient_fun(weights) * lrate
        L.append( primal(weights))
    
    w = weights[:-1]
    b = weights[-1]
    score = np.dot(inputs,w)+b    
    print "Classification Accuracy",np.mean((2*(score>0)-1)==targets)
    
    #plt.plot(L); 
    #plt.xlabel('iterations'); plt.ylabel('Loss'); plt.grid()
    #plt.show(); 
    plotit(inputs,targets,clf=decision_function)