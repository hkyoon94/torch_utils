######################################################################################
######################### AUXILIARY FUNCTIONS FOR TORCHTOOLS #########################
######################################################################################

import os
cwd = os.getcwd()
import plotly.io as pyio
from math import sqrt


def near_sqrt_divisor(n):
    m = int(sqrt(n))
    while n % m != 0: 
        m = m+1
    return m

def printhead(header):
    print(f"\n##### NOW {header}... #####\n")