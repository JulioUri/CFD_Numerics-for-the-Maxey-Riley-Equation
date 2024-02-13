import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Oscillatory(velocity_field):
    
    def __init__(self):
        self.limits  = False
        self.Lambda  = 6 # / (2.0 * np.pi)
    
    def get_velocity(self, x, y, t):
        u = 0.05
        v = np.sin(self.Lambda * t)
        return u, v
    
    def get_gradient(self, x, y, t):
        ux, uy = 0.0, 0.0
        vx, vy = 0.0, 0.0
        return ux, uy, vx, vy

    def get_dudt(self, x, y, t):
        ut = 0.0
        vt = self.Lambda * np.cos(self.Lambda * t)
        return ut, vt