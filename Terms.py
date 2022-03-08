import numpy as np
import celerite2
from celerite2 import driver

class NSTerm(celerite2.terms.Term):
    
    def __init__(self, term, f):
        self.term = term
        self.f = f
        
    def get_coefficients(self):
        return self.term.get_all_coefficients()
    
    def get_celerite_matrices(self, x, diag, a=None, c=None, U=None, V=None):
        
        c, a, U, V = self.term.get_celerite_matrices(x, np.zeros(np.shape(x)))
        farr = np.tile(self.f(x), (np.shape(U)[1], 1)).T
        return c, farr[:, 0] * farr[:, 0] * a + diag, farr * U, farr * V
    
    def __add__(self, b):
        return NSTermSum(self, b)
    
    def __radd__(self, b):
        return NSTermSum(b, self)
    
class NSTermSum(celerite2.terms.Term):
    
    def __init__(self, *terms):
        self._terms = terms
        
    def __add__(self, b):
        return NSTermSum(b, *self._terms)
    
    def __radd__(self, b):
        return NSTermSum(*self._terms, b)
        
    def get_coefficients(self):
        coeffs = (t.get_coefficients() for t in self.terms)
        return tuple(np.concatenate(c) for c in zip(*coeffs))
        
    def get_celerite_matrices(self, x, diag, a=None, c=None, U=None, V=None):
        
        c, a, U, V = self._terms[0].get_celerite_matrices(x, np.zeros(np.shape(x)))
        farr = np.tile(self._terms[0].f(x), (np.shape(U)[1], 1)).T
        U = farr * U
        V = farr * V
        a = farr[:, 0] * farr[:, 0] * a
        
        for term in self._terms[1:]:
            
            ct, at, Ut, Vt = term.get_celerite_matrices(x, np.zeros(np.shape(x)))
            farr = np.tile(term.f(x), (np.shape(Ut)[1], 1)).T
            a = a + farr[:, 0] * farr[:, 0] * at
            c = np.concatenate((c, ct))
            U = np.concatenate((U, farr * Ut), axis=1)
            V = np.concatenate((V, farr * Vt), axis=1)
                        
        return c, a + diag, U, V