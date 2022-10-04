# Create a gym env that simulates 3-lane highway traffic
import numpy as np
import random


def supply(Q,w,kj,k): # downstream supply
    return  min(Q, w*(kj-k))

def demand(Q,u,k): # upstream demand
    return min(Q, k * u)

def Flux(u, w, kj, k, kd, cap_u, cap_d):
    return min(supply(cap_d, w, kj, kd), demand(cap_u, u, k)) # the flow from current cell to downstream cell

def den_to_v(den, u, w, kj): # unit is mi/hr
    kc = w/(u+w)*kj
    if den<=kc:
        return u
    else:
        q = w*max((kj-den),0.0)
        return q/den