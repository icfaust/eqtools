# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This file is part of EqTools.
#
# EqTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EqTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides classes for working with NSTX EFIT data.
"""

import scipy

from .EFIT import EFITTree
from .core import PropertyAccessMixin, ModuleWarning

import warnings

try:
    import MDSplus
    from MDSplus._treeshr import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work. Exception raised "
                      "was of type %s, message was '%s'."
                      % (_e_MDS.__class__, _e_MDS.message),
                      ModuleWarning)
    _has_MDS = False

class NSTXEFITTree(EFITTree):
    """Inherits :py:class:`~gptools.EFIT.EFITTree` class. Machine-specific data
    handling class for the National Spherical Torus Experiment (NSTX). Pulls EFIT
    data from selected MDS tree and shot, stores as object attributes. Each EFIT
    variable or set of variables is recovered with a corresponding getter method.
    Essential data for EFIT mapping are pulled on initialization (e.g. psirz grid).
    Additional data are pulled at the first request and stored for subsequent usage.
    
    Intializes NSTX version of EFITTree object.  Pulls data from MDS tree for storage
    in instance attributes.  Core attributes are populated from the MDS tree on initialization.
    Additional attributes are initialized as None, filled on the first request to the object.

    Args:
        shot: (long) int
            NSTX shot index (long)
    
    Kwargs:
        tree: str
            Optional input for EFIT tree, defaults to 'EFIT01'
            (i.e., EFIT data are under \\EFIT01::top.results).
        length_unit: str
            Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
                
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
                
            Default is 'm' (all units taken and returned in meters).
        gfile: str
            Optional input for EFIT geqdsk location name, defaults to 'g_eqdsk'
            (i.e., EFIT data are under \\tree::top.results.G_EQDSK)
        afile: str
            Optional input for EFIT aeqdsk location name, defaults to 'a_eqdsk'
            (i.e., EFIT data are under \\tree::top.results.A_EQDSK)
        tspline: Boolean
            Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic: Boolean
            Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
    """
    def __init__(self, shot, tree='EFIT01', length_unit='m', gfile='geqdsk', afile='aeqdsk', tspline=False, monotonic=False):

        root = '\\'+tree+'::top.results.'
        super(NSTXEFITTree, self).__init__(shot, tree, root, length_unit=length_unit, gfile=gfile, afile=afile, tspline=tspline, monotonic=monotonic)
  
    def getFluxVol(self): 
        """
        Not implemented in NSTXEFIT tree.
        
        Returns volume within flux surface [psi,t]
        """
        raise NotImplementedError()
        
    def getRmidPsi(self, length_unit=1):
        """ returns maximum major radius of each flux surface [t,psi]
        """
        
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.getNode(self._root+'derived:rpres')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = RmidPsiNode.units
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()
        
        
    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS [t]
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.getNode(self._root+self_afile+':volume')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = volLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

class NSTXEFITTreeProp(NSTXEFITTree, PropertyAccessMixin):
    """NSTXEFITTree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass