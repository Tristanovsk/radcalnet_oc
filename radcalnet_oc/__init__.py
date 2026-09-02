'''

Version history
==================

0.0.1:
    - initial version (2025/03/17)
0.0.2:
    - add multiangular option for TOA simulation (2026/03/31)
0.0.3:
    - add sunglint module (2016/09/02)
'''

__package__ = 'radcalnet_oc'
__version__ = '0.0.3'

from .lut import LUT, AuxData, SolarIrradiance, Spectral
from .kernel import Aerosol, Misc, GaseousTransmittance, Radiometry
from .process import Process
from .aeronet_oc import Aeronet
from .coxmunk import Sunglint


import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)