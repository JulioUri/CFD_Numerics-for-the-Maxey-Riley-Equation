#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065
"""

import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Analytical(velocity_field):

  def __init__(self, omega):
    self.omega    = omega
    self.limits   = False
    
  def get_velocity(self, x, y, t):
    u = -y * self.omega
    v =  x * self.omega
    return u, v
  
  def get_gradient(self, x, y, t):
    ux =  0.0
    uy = -1.0 * self.omega
    vx =  1.0 * self.omega
    vy =  0.0
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    ut = 0.0
    vt = 0.0
    return ut, vt