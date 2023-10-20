import numpy as np
import matplotlib.pyplot as plt
import math

#testFunction
def helloWorld():
    print("Hello World!")

#eulersMethod(): Generalized Function that takes in the stepsize, initial conditions and said function from previous cell and returns
#a 2d array where the first entry is the array of x-values and the second entry is the array of approximated y-values

def eulersMethod(stepSize, initialPair, function, intervalLength):
    y0=initialPair[1]
    x0=initialPair[0]
    deltat=stepSize

    lasty = y0
    lastx = x0

    solutionx = np.array([x0])
    solutiony = np.array([y0])
    iterations = math.floor(intervalLength/deltat)
    for i in range(iterations):
        nexty = lasty+(deltat*(function(lastx,lasty)))
        nextx = lastx+deltat
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        
        lasty = nexty
        lastx += deltat
        
    solution = np.array([solutionx, solutiony])
    return(solution)

#rungeKutta(): Function set up exactly the same as eulersMethod(), just runs the Runge-Kutta algorithm instead
def rungeKutta(stepSize, initialPair, function, intervalLength):
    y0=initialPair[1]
    x0=initialPair[0]
    
    lasty = y0
    lastx = x0
    
    solutionx = np.array([x0])
    solutiony = np.array([y0])
    iterations = math.floor(intervalLength/stepSize)
    for i in range(iterations):
        k1 = function(lastx,lasty)
        k2 = function(lastx+(stepSize/2), lasty+(stepSize*(k1/2)))
        k3 = function(lastx+(stepSize/2), lasty+(stepSize*(k2/2)))
        k4 = function(lastx+stepSize, lasty+(stepSize*k3))
        
        nexty = lasty+((stepSize/6)*(k1+2*k2+2*k3+k4))
        nextx = lastx+stepSize
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        
        lasty = nexty
        lastx += stepSize
        
    solution = np.array([solutionx, solutiony])
    return(solution)
