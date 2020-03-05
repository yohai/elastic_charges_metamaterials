import os
import pickle as pickle
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy import io as spio

import pycharge as pc
from matplotlib.path import Path
from matplotlib.patches import PathPatch
lats = ['square', 'triangle', 'rtriangle']

if not os.path.isdir('pkls'):
    os.mkdir('pkls')


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    source: https://stackoverflow.com/questions/7008608/
    """

    def _check_keys(dct):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dct:
            if isinstance(dct[key], spio.matlab.mio5_params.mat_struct):
                dct[key] = _todict(dct[key])
        return AttrDict(dct)

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dct = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dct[strg] = _todict(elem)
            else:
                dct[strg] = elem
        return AttrDict(dct)

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class AttrDict(dict):
    """
    Dictionarry with attribute access.
    ref: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def lstsq(A, B):
    return np.linalg.lstsq(A, B, rcond=None)[0]


def loop_and_merge(ds, func, loop_dims, *args, merge_into=True, **kwargs):
    """
    applies func to ds (DataArray/Dataset), looping over loop_dims and concatenating all outputs
    to a single DataArray/Dataset
    """
    if isinstance(loop_dims, str):
        current_dim = loop_dims
        other_dims = []
    else:
        current_dim = loop_dims[0]
        other_dims = loop_dims[1:]
    assert current_dim in ds.dims, f'{current_dim} not in dataset dimensions'
    if len(other_dims) > 0:
        res = [
            loop_and_merge(ds.isel(**{current_dim: i}),
                           func,
                           other_dims,
                           *args,
                           merge_into=False,
                           **kwargs) for i in range(len(ds[current_dim]))
        ]
        res = xr.concat(res, ds[current_dim])
    else:
        res = [
            func(ds.isel(**{current_dim: i}),
                 *args,
                 **kwargs) for i in range(len(ds[current_dim]))
        ]
        res = xr.concat(res, ds[current_dim])
    if merge_into:
        res = xr.merge([ds, res])
    return res


def get_Y():
    """Get the Young modulus for the charges and the FE calculation."""
    dss = {lat: pc.process_lattice(lat, layers_to_keep=1) for lat in lats}
    lrs = {lat: loop_and_merge(ds, pc.linear_response, 'porosity', merge_into=True)
           for lat, ds in dss.items()}

    Y_c = xr.concat([l.Y for l in lrs.values()], 'lattice')
    Y_FE = FE_Y()
    Y_FE.porosity.values = Y_c.porosity
    Y = xr.concat([Y_c, Y_FE], pd.Index(['FE', 'charges'], name='kind'))
    return Y


def get_ce():
    """Get the critical strain for the charges and the FE calculation."""
    nl = pickle.load(open('pkls/non_linear.pkl', 'rb'))
    ce = xr.concat([v.delta_c for v in nl.values()], 'lattice')
    height = xr.concat([v.height for v in nl.values()], 'lattice')
    gce = pc.FE_ce()
    gce.coords['porosity'] = ce.porosity
    ce = xr.concat([ce, 0.5 * gce * height], pd.Index(['charges', 'FE'], name='kind'))
    return ce


def FE_Y(lat=None):
    """Returns the Young modulus obtained from fully nonlinear FE calculations."""
    Ygl = xr.Dataset()
    for f in glob('data/FE/Y_*'):
        mat = loadmat(f)
        for k, v in mat.items():
            if k.startswith('Y_'):
                Ygl[os.path.os.path.basename(f)[:-18]] = ('porosity', v)
                break
    Ygl.coords['porosity'] = np.arange(.3, .71, .05)
    Ygl = Ygl.rename(Y_TRIA='triangle', Y_TRIA_Xshape='rtriangle', Y_SQUARE='square')
    if lat is not None:
        return Ygl[lat]
    else:
        return Ygl.to_array('lattice')


def FE_ce(lat=None):
    """Returns the critical strain at instability from fully nonlinear FE calculations."""
    lats = ['square_strain', 'triangle_strain', 'triangle_Xshape_strain']
    da = xr.DataArray([loadmat(f'data/FE/{q}.mat')[q] for q in lats],
                      dims=['lattice', 'porosity'],
                      coords={'porosity': np.arange(.3, .71, .05),
                              'lattice': ['square', 'triangle', 'rtriangle']})
    if lat is not None:
        da = da.sel(lattice=lat)
    return da


class Cyclifier:
    """Class module for ordering the points in outline plots."""
    def __init__(self, nodes):
        self.cycles = Cyclifier.find_cycles(nodes)

    def __call__(self, field, as_patches=False, skip=1):
        return Cyclifier.cyclify(field, cycles=self.cycles, as_patches=as_patches, skip=skip)

    @staticmethod
    def walk_one_step(left, left_ix, ix_in_left):
        p = left[:, ix_in_left]
        ix = left_ix[ix_in_left]
        left = np.delete(left, ix_in_left, axis=1)
        left_ix = np.delete(left_ix, ix_in_left)
        return p, ix, left, left_ix

    @staticmethod
    def get_one_cycle(left, left_ix):
        p0, ix0, left, left_ix = Cyclifier.walk_one_step(left, left_ix, 0)
        cycle_ix = [ix0]
        cycle = [p0]

        d = np.linalg.norm(left - p0[:, np.newaxis], axis=0)
        ix_in_left = np.argmin(d)
        p, ix, left, left_ix = Cyclifier.walk_one_step(left, left_ix, ix_in_left)
        cycle.append(p)
        cycle_ix.append(ix)

        while ix != ix0:
            d = np.linalg.norm(left - p[:, np.newaxis], axis=0)
            ix_in_left = np.argmin(d)
            p, ix, left, left_ix = Cyclifier.walk_one_step(left, left_ix, ix_in_left)
            cycle.append(p)
            cycle_ix.append(ix)
            if len(cycle) == 4:  # add back the first point after a few steps
                left = np.c_[p0, left]
                left_ix = np.r_[ix0, left_ix]
        return np.array(cycle_ix), left, left_ix

    @staticmethod
    def find_cycles(ref):
        """returns a list of lists. each list iterates through a cycle."""
        if not isinstance(ref, np.ndarray):
            ref = ref.values
        cycles = []
        left = ref.copy()
        left_ix = list(range(left.shape[1]))

        while left.size:
            cycle_ix, left, left_ix = Cyclifier.get_one_cycle(left, left_ix)
            cycles.append(cycle_ix)
        ix = np.argsort([len(c) for c in cycles])
        cycles = [cycles[i] for i in ix[::-1]]
        return cycles

    @staticmethod
    def cyclify(field, cycles, as_patches=False, skip=1):
        if not isinstance(field, np.ndarray):
            field = field.values
        if as_patches:
            return [field[:, c[::skip]] for c in cycles]
        else:
            f = [np.c_[field[:, c], [np.nan, np.nan]] for c in cycles]
            return np.concatenate(f, axis=1)

def patchify(paths):
    """Converts a list of paths to matplotlib patches. Assumes the first path is the exterior
        and the rest are holes."""

    def ring_coding(path):
        """The codes will be all "LINETO" commands, except for MOVETO at the
        beginning of each subpath and CLOSEPOLY at the end"""
        n = path.shape[1]
        codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return codes

    def winding_direction_is_visible(hole):
        return ((hole[0] - np.roll(hole[0], 1)) *
                (hole[1] + np.roll(hole[1], 1))).sum() > 0

    if not winding_direction_is_visible(paths[0]):
        paths[0] = paths[0][:, ::-1]
    for i in range(1, len(paths)):
        if winding_direction_is_visible(paths[i]):
            paths[i] = paths[i][:, ::-1]

    vertices = np.concatenate(paths,axis=1)
    codes = np.concatenate([ring_coding(p) for p in paths])
    return PathPatch(Path(vertices.T, codes))

