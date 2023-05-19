import numpy as np
import matplotlib.pyplot as plt

def plot_sts_map(cgeom, evec, neighbor_list, filename, plot_bond=False, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):
    def get_local_grid(x_arr, y_arr, p, cutoff=10.0):
        """Method that selects a local grid around an atom

        Args:
            x_arr: global x array
            y_arr: global y array
            p: atomic position
            cutoff (float, optional): extent of local grid in all directions. Defaults to 5.0.
        """

        x_min_i = np.abs(x_arr - p[0] + cutoff).argmin()
        x_max_i = np.abs(x_arr - p[0] - cutoff).argmin()
        y_min_i = np.abs(y_arr - p[1] + cutoff).argmin()
        y_max_i = np.abs(y_arr - p[1] - cutoff).argmin()

        local_x, local_y = np.meshgrid(x_arr[x_min_i:x_max_i], y_arr[y_min_i:y_max_i], indexing='ij')

        return [x_min_i, x_max_i, y_min_i, y_max_i], [local_x, local_y]

    def carbon_2pz_slater(x, y, z, z_eff=3.25):
        """Carbon 2pz slater orbital

        z_eff determines the effective nuclear charge interacting with the pz orbital
        Potential options:

        z_eff = 1
            This corresponds to a hydrogen-like 2pz orbital and in
            some cases matches well with DFT reference

        z_eff = 3.136
            Value shown in https://en.wikipedia.org/wiki/Effective_nuclear_charge

        z_eff = 3.25
            This is the value calculated by Slater's rules (https://en.wikipedia.org/wiki/Slater%27s_rules)
            This value is also used in https://doi.org/10.1038/s41557-019-0316-8
            This is the default.
        
        """
        r_grid = np.sqrt(x**2 + y**2 + z**2)  # angstrom
        a0 = 0.529177  # Bohr radius in angstrom
        return z * np.exp(-z_eff * r_grid / (2 * a0))

    def _get_atoms_extent( atoms, edge_space):
        xmin = np.min(atoms[:, 0]) - edge_space
        xmax = np.max(atoms[:, 0]) + edge_space
        ymin = np.min(atoms[:, 1]) - edge_space
        ymax = np.max(atoms[:, 1]) + edge_space
        return [xmin, xmax, ymin, ymax]

    def calc_orb_map( cgeom, evec, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        extent = _get_atoms_extent(cgeom, edge_space)

        # define grid
        x_arr = np.arange(extent[0], extent[1], dx)
        y_arr = np.arange(extent[2], extent[3], dx)

        # update extent so that it matches with grid size
        extent[1] = x_arr[-1] + dx
        extent[3] = y_arr[-1] + dx

        orb_map = np.zeros((len(x_arr), len(y_arr)))

        for at, coef in zip(cgeom, evec):
            p = at
            local_i, local_grid = get_local_grid(x_arr, y_arr, p, cutoff=1.2 * h + 4.0)
            pz_orb = carbon_2pz_slater(local_grid[0] - p[0], local_grid[1] - p[1], h, z_eff)
            orb_map[local_i[0]:local_i[1], local_i[2]:local_i[3]] += coef * pz_orb

        return orb_map, extent

    def calc_sts_map(cgeom, evecs, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        evec = evecs
        orb_map, extent = calc_orb_map(cgeom, evec, h, edge_space, dx, z_eff)
        final_map =  np.abs(orb_map)**2
        return final_map, extent
    
    def visualize_backbone( atoms, neighbor_list):
        i_arr, j_arr = neighbor_list
        for i, j in zip(i_arr, j_arr):
            if i < j:
                p1 = atoms[i]
                p2 = atoms[j]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.0, solid_capstyle='round')

    final_map, extent = calc_sts_map(cgeom, evec, h=h, edge_space=edge_space, dx=dx, z_eff=z_eff)
    plt.imshow(final_map.T, origin='lower', cmap="gist_heat", extent=extent)
    if plot_bond:
        visualize_backbone(cgeom, neighbor_list)
    plt.axis('off')
    plt.savefig('%s.png' % filename, dpi=300, bbox_inches='tight')
    plt.close()







from  HH_DSPMM import HDSPMM_CODE

U = 4.3
t = -2.8
M = 6

my= HDSPMM_CODE('inp.xyz', U, t, M)
my.Run_FCI()
# my.plot_Huckel_orb()

plot_sts_map(my.geom, my.cas_mo[:,3], my.near_neigh, filename='homo')


print("done")
