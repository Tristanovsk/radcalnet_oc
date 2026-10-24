'''
radcalnet_oc: simulation of the top-of-atmosphere signal above AERONET-OC sites, for the calibration
and validation of satellite sensors over ocean and coastal waters.

The version history is given in CHANGELOG.md (History page of the documentation).
'''

__package__ = 'radcalnet_oc'
__version__ = '1.0.1'

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