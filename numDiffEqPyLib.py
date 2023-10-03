import numpy as np
import matplotlib.pyplot as plt
import math

#testFunction
def helloWorld():
    print("Hello World!")

#Generalized Function that takes in the stepsize, initial conditions and said function from previous cell and returns
#a 2d array where the first entry is the array of x-values and the second entry is the array of approximated y-values

def eulersMethod(stepSize, initialPair, function):
    y0=initialPair[1]
    x0=initialPair[0]
    deltat=stepSize

    lasty = y0
    lastx = x0

    solutionx = np.array([x0])
    solutiony = np.array([y0])
    iterations = math.floor(6/deltat)
    for i in range(iterations):
        nexty = lasty+(deltat*(function(lastx,lasty)))
        nextx = lastx+deltat
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        lasty=nexty
        lastx+=deltat
        
    solution = np.array([solutionx, solutiony])
    return(solution)
