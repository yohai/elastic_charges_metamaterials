import xarray as xr
import numpy as np
import seaborn as sns
import _pickle as pickle
import pycharge as pc
import matplotlib as mpl
import os
from utils import get_Y, FE_ce, Cyclifier, patchify

if __name__ == '__main__':
    mpl.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
from matplotlib import rc
sns.set(style="whitegrid", font_scale=1.3)
rc('font', **{'family': 'serif'})
rc('text', usetex=False)

lats = ['square', 'triangle', 'rtriangle']


class BaseTrig:
    """" Base class to be inherited by CTrig (charges) and GTrig (FE calculation).  The class handles loading
         the triangulation, calculating relevant fields and plotting. Has two plotting functions: heatmap to 
         plot spatially varying fields and boundary_plot to plot just the boundary.
    """
    def __init__(self, lattice, p, method):
        self.method = method
        self.lattice = lattice
        self.p = p         # porosity
        self.holes = None  # list of hole positions
        self.boundary = None
        self.N_holes = None
        self.tri = None    # A matplotlib Tri object.
        self.real_nodes = None
        self.mask = None   # mask for Tri object, specifies which nodes to plot.
        self.length_factor = None

    def calc_tri(self):
        self.N_holes = self.holes.shape[1]
        self.tri = Triangulation(*self.lagrange_nodes(only_real=False))
        self._set_mask()

    def _set_mask(self, crop=np.inf):
        # mask tris which involve a hole
        c1 = self.tri.triangles.min(axis=1) <= self.N_holes

        # mask tris with points outside crop
        c2 = np.abs(self.tri.x[self.tri.triangles]).max(axis=1) > crop
        c3 = np.abs(self.tri.y[self.tri.triangles]).max(axis=1) > crop
        self.mask = c1 | c2 | c3
        self.tri.set_mask(self.mask)

    def lagrange_nodes(self, only_real=True):
        if only_real:
            return self.real_nodes
        else:
            return np.c_[self.holes, self.real_nodes]

    def euler_nodes(self, only_real=True):
        nodes = np.array([self.tri.x, self.tri.y])
        if only_real:
            nodes = nodes[:, self.N_holes:]
        return nodes

    def deform(self, strain, kind='lin'):
        """
        Deforms the mesh. Kind specifiec by which field to deform the mesh. It can be either 'lin'
        for linear response or 'mode' for most unstable mode. strain is the amplitude.
        """
        if 'linear'.startswith(kind):
            H = self.lagrange_nodes()[1].max() - self.lagrange_nodes()[1].min()
            scale = H * strain/2
            sol = scale*self.d_lin
            self.stress *= scale
        elif 'mode'.startswith(kind):
            sol = strain * self.d_mode
        else:
            raise ValueError(f'didnt understand kind={kind}')

        all_nodes = self.lagrange_nodes(only_real=False)
        self.tri.x = all_nodes[0] + self.append_shit(sol[0])
        self.tri.y = all_nodes[1] + self.append_shit(sol[1])

    def heatmap(self, c, ax=None, cbar=True, cmap=None, clim=None, cax=None, crop=None):
        if ax is None:
            ax = plt.gca()
        single_color = c is None
        if single_color:
            c = np.full_like(self.lagrange_nodes()[0], 0)
            clim = [-1, 1]
            color = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
            cmap = mpl.colors.ListedColormap((color, color))

        if clim is None:
            levels = np.linspace(c.min(), c.max(), 20)
        else:
            levels = np.linspace(*clim, 20)
        if crop:
            self._set_mask(crop)
        if cmap is None and not single_color:
            cmap = sns.color_palette("RdBu_r", 20)
            cmap = mpl.colors.ListedColormap(cmap)

        hf = ax.tricontourf(self.tri, self.append_shit(c), levels=levels,
                            cmap=cmap)
        if not single_color:
            hc = ax.tricontour(self.tri, self.append_shit(c), levels=levels,
                               colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                               linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
        ax.set_aspect(1)
        ax.axis('off')
        if cbar:
            if cax is None:
                cbar = plt.colorbar(hf, ax=ax)
            else:
                cbar = plt.colorbar(hf, cax=cax)
        return ax, cbar

    def boundary_plot(self, deformation='mode', fill=True, scale=1, ax=None,
                      facecolor='b', edgecolor='None', alpha=1, skip=1):
        if deformation == 'mode':
            d = self.d_mode[:, self.boundary]
        elif deformation == 'lin':
            d = self.d_lin[:, self.boundary]
        if ax is None:
            ax = plt.gca()

        reference = self.lagrange_nodes()[:, self.boundary]
        if hasattr(self, 'cyc'):
            cyc = self.cyc
        else:
            print('no cyclifier, calculating')
            cyc = Cyclifier(reference)
            self.cyc = cyc
            print('done')

        pos = reference + scale * d
        patch = patchify(cyc(pos, as_patches=True, skip=skip))
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_alpha(alpha)
        patch = ax.add_patch(patch)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('equal')
        return patch

    def append_shit(self, x, fill_value=None):
        """Prepend zeros to x on hole nodes."""
        assert x.ndim == 1
        if hasattr(x, 'values'):
            x = x.values
        if fill_value is None:
            fill = x.mean().astype(x.dtype)
        else:
            fill = fill_value
        return np.r_[fill + np.zeros(self.N_holes, dtype=x.dtype), x]

    @property
    def sxy(self):
        return self.stress[2]

    @property
    def syy(self):
        return self.stress[1]

    @property
    def sxx(self):
        return self.stress[0]


class GTrig(BaseTrig):
    """A triangulation based on ABAQUS FE mesh (for FE calculation)."""
    def __init__(self, lattice='square', p=0.5):
        super().__init__(lattice, p, method='FE')
        self.holes = (pc.process_lattice(lattice, layers_to_keep=1, porosity=p)
                        .holes.values.T[:, 1:])
        self._boundaries = None

        file_prefix = f'data/FE/fullfe/{lattice}_{int(1000 * p)}.'
        self.file_prefix = file_prefix
        gl_nodes = np.load(file_prefix+'nodes', allow_pickle=True).T
        if lattice is 'rtriangle':
            gl_nodes = gl_nodes[[1, 0]]

        self.d_lin = np.load(file_prefix+'lin_displacement', allow_pickle=True)
        self.calculate_scale_factor(gl_nodes)
        self.real_nodes = (gl_nodes - self.shift) / self.length_factor
        self.stress = 2 * self.length_factor * np.load(file_prefix+'lin_stress', allow_pickle=True)
        self.d_mode = np.load(file_prefix+'mode_displacement', allow_pickle=True) / self.length_factor
        if lattice is 'rtriangle':
            self.stress[:2] = self.stress[:2][::-1]
            self.d_lin = self.d_lin[::-1]
            self.d_mode = self.d_mode[::-1]

        self.d_mode += 2 * (self.d_mode - self.d_mode.mean(axis=1, keepdims=True))
        self.calc_tri()
        self.calc_boundary()
        self.normalize_linear_solution()

    def calculate_scale_factor(self, gl_nodes):
        # First, scale and translate nodes
        # translation
        pc_nodes = pc.get_geom(lat=self.lattice, porosity=self.p).p
        mg = gl_nodes.min(axis=1)
        Mg = gl_nodes.max(axis=1)
        m = pc_nodes.min(axis=1)
        M = pc_nodes.max(axis=1)
        self.shift = np.expand_dims((mg + Mg) / 2, axis=1)

        # expansion
        mg = gl_nodes.min(axis=1)
        Mg = gl_nodes.max(axis=1)
        factor = (Mg - mg) / (M - m)
        assert np.allclose(*factor, atol=1e-3)
        self.length_factor = factor.flat[0]

    def calc_boundary(self):
        h = self.holes[:, self.holes.shape[1] // 2]
        R = np.linalg.norm(self.real_nodes - h[:, np.newaxis], axis=0).min()
        boundaries = np.zeros(self.real_nodes.shape[1], dtype='bool')
        for i in [0, 1]:        # when x or y are maximal
            boundaries = boundaries | np.isclose(self.real_nodes[i], self.real_nodes[i].max(), atol=1e-5)
            boundaries = boundaries | np.isclose(self.real_nodes[i], self.real_nodes[i].min(), atol=1e-5)
        for h in self.holes.T:  # when you're closest to a hole
            d = np.linalg.norm(self.real_nodes - h[:, np.newaxis], axis=0)
            boundaries = boundaries | np.isclose(d, R, atol=1e-3)
        self.boundary = boundaries
        return boundaries

    def normalize_linear_solution(self):
        """ Scales and translates linear solution and stress so that vertical displacement of 
            top/bottom boundaries is \pm 1
        """
        self.d_lin = self.d_lin - self.d_lin.mean(axis=1, keepdims=True)
        self.d_lin /= self.d_lin[1].max()


class CTrig(BaseTrig):
    """A triangulation based on MATLAB FE mesh (for charges calculation)."""
    def __init__(self, lattice='square', p=0.5, calc_fields=True):
        super().__init__(lattice, p, method='charges')
        self.g = pc.get_geom(lat=lattice, porosity=p)
        self.length_factor = 1
        cs = None
        if calc_fields:
            ds = (pc.process_lattice(lattice, layers_to_keep=1, porosity=p)
                  .pipe(pc.linear_response)
                  .pipe(pc.nl_response)
                  )
            cs = pc.calc_charge_fields(ds, reference=True)
            self.ds = ds
            self.stress = xr.dot(cs.s, ds.Q0, dims='charge').values[[0, 2, 1]]
            self.d_lin = xr.dot(cs.d.sel(order=1), ds.Q0, dims='charge').values
            self.d_mode = xr.dot(cs.d.sel(order=1), ds.cmodes.isel(mode=0), dims='charge')

            # calculate boundary
            edges, nodes = pc.all_edges(self.g)
            inds = np.unique(np.concatenate([edges.i1, edges.i2]))
            self.boundary = inds

        self.holes = ds.holes.values.T[:, 1:]  # shape = (2, N_holes)
        if cs is not None:
            self.real_nodes = cs.reference.values
        else:
            self.real_nodes = self.g.p
        self.calc_tri()


def calculate_and_save_tris(save=True, return_result=True, use_saved=True):
    res = {}
    for p in [.3, .5, .7]:
        fname = f'pkls/tris_{int(1000 * p)}.pkl'
        if os.path.isfile(fname) and use_saved:
            continue
        print(p, ':')
        cs = [CTrig(lattice=k, p=p) for k in lats]
        gs = [GTrig(lattice=k, p=p) for k in lats]
        tris = xr.DataArray(
            np.array([cs, gs]),
            coords=[('kind', ['charges', 'FE']), ('lattice', lats)]
        ).T
        for t in tris.values.flat:
            t.cyc = Cyclifier(t.lagrange_nodes()[:, t.boundary])
        if save:
            pickle.dump(tris, open(fname, 'wb'))
        if return_result:
            res[p] = tris
    if return_result:
        return res


def make_axes_for_linear_figure():
    fig = plt.figure(figsize=(10, 10))
    gridspec = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, .05])
    axs = np.zeros([3, 4], dtype='object')
    for j in range(4):
        for i in range(3):
            if j in [1, 2] and not (i==0 and j==1):
                axs[i, j] = fig.add_subplot(gridspec[i, j], sharex=axs[0, 1])
            else:
                axs[i, j] = fig.add_subplot(gridspec[i, j])
    return fig, axs, gridspec


def linear_figure(p=0.5, crop=1, strain=0.25, cmap=None):
    tris = pickle.load(open(f'pkls/tris_{int(1000*p)}.pkl', 'rb'))
    if np.isscalar(strain):
        for t in tris.values.flat:
            t.deform(strain, kind='linear')
    else:
        for t, s in zip(tris.values.flat, strain.flat):
            t.deform(s, kind='linear')

    line_data = get_Y()

    fig, axs, gridspec = make_axes_for_linear_figure()
    for ax in axs[:, [1, 2]].flat:
        ax.set_xlim([-crop, crop])
        ax.set_ylim([-crop, crop])

    # Line plots
    for ax, lat in zip(axs[:, 0], lats):
        h1 = ax.plot(line_data.porosity, line_data.sel(lattice=lat, kind='FE'), 'o', label='Finite Elements')
        h2 = ax.plot(line_data.porosity, line_data.sel(lattice=lat, kind='charges'), 'o', label='Charges')
        ax.set_aspect(1)

    for ax in axs[:, 0]:
        ax.set_xlim([.25, .75])
        ax.set_ylim([0, 0.55])
        ax.set_xticks(np.arange(.3, .73, .1))
        # ylabel commented because it only works if you set up latex support in matplotlib.
        # ax.set_ylabel(r'$Y_{\\mbox{eff}}/Y$')

    axs[-1, 0].set_xlabel('Porosity')
    for ax in axs[[0, 1], 0]:
        ax.set_xticklabels([])

    # HEAT MAPS
    for i, row in enumerate(axs[:, 1:]):
        trow = tris[i].values

        vs = [t.sxy for t in trow]
        m = max(np.abs(v).max() for v in vs)
        m = np.around(m, 2)
        if i == 2:
            m = 0.12

        _ = trow[0].heatmap(ax=axs[i, 1], c=vs[0], clim=[-m, m], cbar=False, crop=crop*1.2, cmap=cmap)
        _ = trow[1].heatmap(ax=axs[i, 2], c=vs[1], clim=[-m, m], cbar=True, crop=crop*1.2, cmap=cmap,
                            cax=row[-1])
        cbar = _[1]
        cbar.set_ticks([-m, 0, m])
        cbar.set_ticklabels([-m, '', m])
        cbar.outline.set_edgecolor('k')
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.set_ylabel(r'$\sigma_{xy}/Y$')

    # tweak axes position
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for ax in axs[:, -1]:  # colorbars
        ap = ax.get_position()
        H = ap.height
        center = ap.min[1] + H / 2
        fac = 0.7
        ax.set_position([ap.min[0] - 0.01, center - fac * H / 2, ap.width, fac * H])

    axs[0, 1].set_title('Charges')
    axs[0, 2].set_title('Finite Elements')

    ps = np.array([ax.get_position() for ax in axs[:, 1]], dtype='object')
    centers = np.array([[(p.min + p.max) / 2] for p in ps]).squeeze()
    lines = [Line2D((0.35, axs[0, 2].get_position().max[1]), (y, y),
                    transform=fig.transFigure, figure=fig,
                    color=[.6, .6, .6], zorder=-1, ls='--')
             for y in (centers[1:, 1] + centers[:-1, 1]) / 2]
    c = (0.5 + 0.75) / 2
    lines.append(Line2D((c, c), (0.1, .88),
                        transform=fig.transFigure, figure=fig,
                        color=[.6, .6, .6], zorder=-1, ls='--'))
    fig.lines.extend(lines)
    leg = plt.figlegend([h1[0], h2[0]], ['Finite Elements', 'Charges'],
                         loc='center left', bbox_to_anchor=(0.11, 0.58))
    fig.canvas.draw()
    return fig, axs, leg


def instability_figure(p=0.7, scale=1):
    fig, axs = plt.subplots(3,4,figsize=(14, 10))
    FE_critical_strain = FE_ce()
    tris = pickle.load(open(f'pkls/tris_{int(1000*p)}.pkl', 'rb'))
    nl = pickle.load(open('pkls/non_linear.pkl', 'rb'))
    scales = [.2, -.1, -.08]
    colors = sns.color_palette('Set2')[1:3]
    for row, lat, scale in zip(axs, lats, scales):
        ### Critical strain
        h1 = row[0].plot(nl[lat].porosity, nl[lat].delta_c, 'o', label='Charges', color=colors[0], alpha=.9)
        h2 = row[0].plot(FE_critical_strain.porosity,
                         FE_critical_strain.sel(lattice=lat) / (2 / nl[lat].height[0]), 'o',
                         label='Finite Elements', color=colors[1], alpha=.9)

        ### Unstable mode plots
        t_c = tris.sel(lattice=lat, kind='charges').values[()]
        t_FE = tris.sel(lattice=lat, kind='FE').values[()]
        for ax in row[1:-1]:
            t_c.boundary_plot(ax=ax, facecolor=colors[0], edgecolor='k', alpha=0.65, scale=scale)
            t_FE.boundary_plot(ax=ax, facecolor=colors[1], edgecolor='k', alpha=0.65, scale=1, skip=5)
            ax.set_aspect(1)
        W = nl[lat].width.values[0]
        H = nl[lat].height.values[0]
        if W > H:
            row[1].set_xlim([-W/2, W/2])
        else:
            row[1].set_ylim([-H/2, H/2])
            
        ### Eigenvalues plot
        row[3].set_prop_cycle(color=mpl.cm.Purples_r(np.linspace(0, 1, 80)))
        evs = nl[lat].eigenvalues.sel(porosity=p)
        evs=(evs
             .where((evs > -3.1) & (evs < 4.3) & (evs.delta < 0.31))
             .dropna('delta', how='all')
             .dropna('mode', how='all')
            )
        row[3].plot(evs.delta, evs)
        dc = nl[lat].delta_c.sel(porosity=p)
        row[3].plot([dc, dc], [0, -5], '--k')
        row[3].plot(dc, 0, 'ok')
        row[3].plot(dc, 0, 'ok')
        row[3].text(dc, -3, '$\\epsilon_c$', ha='center', va='top', fontsize=14)

    axs[0, 3].set_title('Eigenvalues')
    for ax in axs[:, 0]:
        ax.set_ylabel('Critical strain')
    for ax in axs[:, -1].flat:
        ax.set_xlim([0, 0.3])
        ax.set_ylim([-3, 4.2])
    axs[-1, 0].set_xlabel('Porosity')
    axs[-1, -1].set_xlabel('Strain')
    for ax in axs[:, 2].flat:
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_aspect(1)
    for i, ax in enumerate(axs[:, 0]):
        ax.set_xticks(np.arange(.3, .71, .1))
        ax.set_yticks(np.arange(0, .5, .1))
        ax.set_xlim([0.25, 0.75])
        ax.set_ylim([0.0, 0.45])
    for ax in axs[:2,  [0, -1]].flat:
        ax.set_xticklabels(['']*len(ax.get_xticks()))

    plt.subplots_adjust(hspace=0.1, wspace=0.13)
    leg = fig.legend([h1[0], h2[0]], ['Finite Elements', 'Charges'],
                     loc='center left', bbox_to_anchor=(0.1, .32))
    return fig, axs, leg
