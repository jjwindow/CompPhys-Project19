"""
Numerical methods for integrating the Newton-Coates function for a quantum particle.
This document contains all the functions written to compare methods of computing
the integral. These methods are utilised and compared in Newtoncoates_calcs.py.

Jack J Window - Imperial College London - Autumn 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt # Order of magnitude faster than np.sqrt(). Tested using -mtimeit in CMD.
import copy

def trapezoid(func, a, b, epsilon, itNum = None):
    """
    Performs trapezoid rule numerical integration on a generic function between limits a and b.
    performs integration until convergence at accuracy epsilon is reached. Returns the value of
    the integration and the number of iterations taken to reach convergence.
    itNum (default = None) is an integer number of iterations. If itNum = j iterations are performed before 
    epsilon has been reached, T_j will be returned.
    """
    if itNum is not None:
        itNum = int(itNum)
    
    z = [a,b] # First two x-values are the limits of integration
    f = [func(a), func(b)] # Function values at limits
    h = abs(a-b) # Interval size
    I = 0.5 * h * (f[0]+f[1])
    n = len(z)
    check = epsilon + 1 # Meets while loop condition without needing explicit I_2 calculation
    count = 1 # Initialise iteration counter
    if itNum == 1:
        return [I, 2**count + 1]

    while check >= epsilon:
        # Iterates through Extended Trapezoidal Rule until accuracy condition is met
        I_old = I # I_(j+1) -> I_j
        i = 0
        trapSum = [] # The values of the function at the newly added x points will fill this.
        while i < n:
            n = len(z)
            midpoint = (z[i]+z[i+1])/2
            # Insert intermediate x and f(x) values
            z.insert(i+1, midpoint) 
            # f.insert(i+1, func(midpoint)) # function values at each x. Useful for plotting but not necessary.
            trapSum.append(func(midpoint))
            i += 2 # Skip new entry into x, f(x) or loop will be infinite.

        h = h/2 # Number of x_n doubles, so h halves
        # Iterative formula for Extended Trapezoidal Method:
        I = 0.5 * I_old + h*np.array(trapSum).sum()
        count += 1
        check = abs((I - I_old)/I) # Update convergence check

        if type(itNum) is not None:
            if count == itNum:
                return [I, 2**count + 1]
        
    return [I, 2**count + 1]

# print(trapezoid(psi_sq, 0, 2, 1e-9))

def simpson(func, a, b, epsilon):
    S = (4/3) * trapezoid(func, a, b, epsilon, 2)[0] - (1/3) * trapezoid(func, a, b, epsilon, 2)[0]
    numEvals = trapezoid(func, a, b, epsilon, 2)[1] + trapezoid(func, a, b, epsilon, 2)[1]
    check = epsilon + 1
    i = 2
    while check >= epsilon:
        S_old = S.copy()
        T_array1 = trapezoid(func, a, b, epsilon, i+1)
        T_array2 = trapezoid(func, a, b, epsilon, i)
        _T = T_array1[0]
        T = T_array2[0]
        _count = T_array1[1]
        count = T_array2[1]

        if (type(_T) is np.float64) and (type(T) is np.float64):
            S = (4/3) * _T - (1/3) * T
            numEvals += (count + _count)
        check = abs((S-S_old)/S_old)
        i += 1

    return [S, numEvals]

### Q3: Monte Carlo Methods ###
import random

def linear_pdf(z):
    return 0.98 - 0.48 * z

def test_pdf(z):
    return 0.7 - 0.3 * z

def randNum(pdf, z_lo, z_hi, pdf_lo=None, pdf_hi=None):
        """
        Sub-function to generate a random number according to the
        PDF supplied to the function. If None, then a uniform deviate
        between a and b will be returned. If a PDF is to be used then the
        rejection method is implemented to return random numbers that
        follow a supplied PDF.
        """

        if pdf is None:
            # Use uniform random deviate in interval
            return np.random.uniform(z_lo, z_hi)
        else:
            # Rejection method:
            valueReturned = False
            # Keep generating numbers until criteria are met.
            while valueReturned == False:
                z_rand = np.random.uniform(z_lo, z_hi)
                f_rand = np.random.uniform(pdf_lo, pdf_hi)
                if pdf(z_rand) > f_rand:
                    valueReturned = True
            return z_rand

def updater(n, f_avg, diff_sq, f_new):
    """
    Function to update the average function value whilst also updating the 
    variance estimate for each new data point, using Welford's online algorithm.
    """
    n += 1
    # Update function average estimate
    f_avg_old = f_avg
    f_avg += (f_new-f_avg)/n
    # Update sum of squared differences
    diff_sq += (f_new-f_avg)* (f_new-f_avg_old)
    return n, f_avg, diff_sq

def mc_integrate(func, a, b, epsilon, pdf=None):
    """
    2D Monte Carlo integration. Takes the integrand as argument
    'func'. Integrates between supplied values a and b with N samples, and refines
    to an accuracy 'epsilon'. If supplied with an importance sampling
    PDF for the random number generation, it will use that to improve 
    the result. Returns number of iterations and the result of 
    the integration.
    """
    w = abs(a-b) # Calculate width of integral limits
    z_limits = [a, b]
    z_limits.sort() # Make sure limit values are in order
    z_lo = z_limits[0]
    z_hi = z_limits[1]
    z = np.linspace(z_limits[0], z_limits[1], w*500) # Number density constant irrespective of width size
    f = [func(_z) for _z in z]

    if pdf is not None:
        # Calculate PDF min and max for later use
        pdf_vals = [pdf(_z) for _z in z]
        pdf_lo = min(pdf_vals)
        pdf_hi = max(pdf_vals)
    else:
        pdf_lo = None
        pdf_hi = None
    
    N = 1e4 # Starting N value
    I = -1 # Dummy I value so first check value can be calcualated
    count = 0
    checkVals = []

    check = epsilon + 1 # Dummy check value to enter loop
    n = 0
    f_avg = 0
    diff_sq = 0
    z_rand_arr = []

    while check > epsilon:
        # This loop continues until user-supplied accuracy has been
        # reached.
        I_old = I # Reassign I_old for convergence calculation
        while N > n:
            # This loop iterates until the number of desired samples
            # have been calculated.
            z_rand = randNum(pdf, z_lo, z_hi, pdf_lo, pdf_hi) # Random z val
            f_rand = func(z_rand) # Value of function at z
            if pdf is not None:
                f_rand = f_rand/pdf(z_rand) 
            z_rand_arr.append(z_rand)
            n, f_avg, diff_sq = updater(n, f_avg, diff_sq, f_rand)
            
        if pdf is None:
            I = w * f_avg # Formula for estimation of area
        else:
            I = f_avg
        check = abs((I - I_old)/I_old) # Convergence check
        checkVals.append(check) # For plotting
        N += 1e4 # Increase N (law of large numbers)
        count += 1
    # Calculate variance of function average estimate using squared differences sum
    var = diff_sq/N
    # Formula for uncertainty on I
    sigma_I = sqrt(var)
    
    return [I, sigma_I], count, checkVals, z_rand_arr

def metropolis(func, a, b, epsilon, pdf):
    """
    Algorithm to use adaptive importance sampling according to a given PDF.
    """
    def acceptProb(pdf, z_val, z_val_new):
        """
        Returns the probability of accepting the new z-value as the next step
        according to the sampling strategy defined in the Metropolis algorithm.
        pdfVal represents the evaluation of the PDF at the starting point, and 
        pdfVal_ represetns the evaluation at the new z value, an incremental 
        step away from the starting point.
        """
        pdfVal = pdf(z_val)
        pdfVal_ = pdf(z_val_new)

        if pdfVal_ >= pdfVal:
            return 1
        else:
            return pdfVal_/pdfVal
    
    def acceptor(p):
        """
        Function that takes a probability of acceptance and returns whether a new 
        step is accepted as a Boolean value.
        """
        rand = random.random()
        if rand <= p:
            return True
        else:
            return False
    
    def stepValue(z, s):
        """
        Randomly generate an increment using a normal distribution with a given width,
        centred at the current z-value.
        """
        return np.random.normal(z, s) - z

    w = abs(a-b)    
    z_limits = [a, b]
    z_limits.sort() # Make sure limit values are in order
    z_lo = z_limits[0]
    z_hi = z_limits[1]
    N = 1e4
    check = epsilon + 1
    n = 0
    f_avg = 0
    diff_sq = 0
    I = -1
    checkVals = []
    count = 0

    while check > epsilon:
        z_start = randNum(None, z_lo, z_hi) # Uniform random number in z range
        s = w/50 # Appropriately small Gaussian width, changes with function limits.
        I_old = I
        while N > n:
            z_new = z_start + stepValue(z_start, s) # Take an incremental step from start point
            p = acceptProb(pdf, z_start, z_new) # Determine probability of acceptance
            # Check if the new z value is accepted and within integration limits
            if (acceptor(p) is True) and (z_lo <= z_new ) and (z_new <= z_hi):
                f_rand = func(z_new)/pdf(z_new)
                n, f_avg, diff_sq = updater(n, f_avg, diff_sq, f_rand) # Update average and variance
                z_start = z_new # reassign z_new as the new starting point.

        I = f_avg
        check = abs((I - I_old)/I_old) # Convergence check
        checkVals.append(check) # For plotting
        N += 1e4 # Increase N (law of large numbers)
        count += 1
    # Calculate variance of function average estimate using squared differences sum
    var = diff_sq/N
    # Formula for uncertainty on I
    sigma_I = sqrt(var)
    return [I, sigma_I], count, checkVals
                
def metropolisPDF(z):
    """
    PDF used to guide the metropolis algorithm's random walk through the function
    by weighting samples in regions where the function is larger.
    """
    return np.cos(z/1.5)/1.45991

# # I, count, checkVals, z_rand_arr = mc_integrate(psi_sq, 0, 2, 1e-6, linear_pdf)
# I, count, checkVals = metropolis(psi_sq, 0, 2, 1e-6, metropolisPDF)
# checkVals = checkVals[1:] # First value is dummy value.
# print(I[0], "+- ", I[1], ", ", count)

# # Plot convergence check values
# plt.plot(np.linspace(0, count, len(checkVals)), checkVals)
# plt.show()
# # Graph shows oscillatory behaviour instead of smooth descent expected.

# # Plot psi_sq function
# z = np.linspace(0, 2, 1000)
# f = [psi_sq(_z) for _z in z]
# pdf = [metropolisPDF(_z) for _z in z]
# plt.plot(z, pdf)
# # plt.hist(z_rand_arr, 75, density=True)
# plt.show()


