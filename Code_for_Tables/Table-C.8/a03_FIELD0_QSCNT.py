#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065
"""

import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Quiescent(velocity_field):

  def __init__(self):
    self.limits     = False
    
  def get_velocity(self, x, y, t):
    u = 0.0
    v = 0.0
    return u, v
  
  def get_gradient(self, x, y, t):
    ux =  0.0
    uy =  0.0
    vx =  0.0
    vy =  0.0
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    ut = 0.0
    vt = 0.0
    return ut, vt