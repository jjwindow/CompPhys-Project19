"""
Numerical methods for integrating the Newton-Coates function for a quantum particle.
All functions utilised are from the module NewtonCoates_funcs. Full details of the
code are annotated in the file.

Jack J Window - Imperial College London - Autumn 2019
"""
from NewtonCoates_funcs import *

### Q2: Trapezoidal and Simpson's Rules ###

# Define the integrand

def psi_sq(z):
    """
    The 1-D probability density function for the quantum system specified.
    The phase factor a(z) disappears when the wavefunction becomes the probability density.
    Takes a 1D spatial parameter z and returns the value of the probability density function
    at that z-value.
    """
    return (1/np.pi)**0.5 * np.exp(-z**2)

# Integration boundaries:
low = 0
high = 2

# Trapezoidal Method:
epsilon = 1e-6
I_trap = trapezoid(psi_sq, low, high, epsilon)
print("Trapezoidal Integration: ", I_trap[0], "\nNo. Function Evals: ", I_trap[1])

# Simpson's Method:
I_simp = simpson(psi_sq, low, high, 1e-6)
print("Simpson Integration: ", I_simp[0], "\nNo. Function Evals: ", I_simp[1])