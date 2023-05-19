import numpy as np
import scipy as sp
import re
import os
import time
from pyscf import tools, fci
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = "1"



r'''Extra tools of no Use'''
def Print_input_parameters(Nst,Nel,Sz,max_oc,Processors):
    if type(Sz) == int:
        Sz = Sz % 2
    print('\n*******************INPUT  PARAMETERS*************************')
    print(f'Number of processor used is                     {Processors}')
    print(f'Maximum occupation of each site is              {max_oc}')
    print(f'Number of sites are                             {Nst}')
    print(f'Total number of electrons are                   {Nel}')
    print(f'Sz value is                                     {Sz}')
    print('*************************************************************')

def Print_time_elepsed(st,et):
    hours, rem = divmod(et-st, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTime Taken: {:0>2}H:{:0>2}M:{:05.2f}S\n".format(int(hours),int(minutes),seconds))

def Print_Title():
    title = '''
+---------------------------------------------------------------------------------+
|                              D S P M M                                          |
|                           ================                                      |
|                        A      CASCI      CODE                                   |
+---------------------------------------------------------------------------------+'''
    print(title)
r'''end extra tools'''


def read_geom(g_file):
    "Readig the geometry form xyz file"
    cgeom = []        # array to store geometry coordinates
    with open(g_file,'r') as gfile:
        gfile.readline()
        gfile.readline()
        for line in gfile:
            if re.search("C", line):
                word = line.split()
                cgeom.append(word[1:])
    cgeom = np.array(cgeom)
    cgeom = cgeom.astype(np.float_)
    return cgeom

def nearest_neighbout(cgeom):
    near_neigh = [[],[]]
    for i in range(len(cgeom)):
        for j in range(len(cgeom)):
            if i!=j:
                dist = np.linalg.norm(cgeom[i, :] - cgeom[j, :])
                if dist < 1.7:
                    near_neigh[0].append(i),near_neigh[1].append(j)
    return near_neigh

def _unpack_el(Nel):
    u_el = np.ceil(Nel/2)
    d_el = np.floor(Nel/2)
    return (int(u_el), int(d_el))

def Huckel_en_vec(Nat, cgeom, t):
    Hamil = np.zeros((Nat, Nat))         # Hamiltonian of huckel terms
    for j in range(Nat):
        for k in range(j+1, Nat):
            distance = np.linalg.norm(cgeom[j, :] - cgeom[k, :])
            if distance < 1.7 and j != k :
                Hamil[j, k] = Hamil[k, j] = t
    E, C =  sp.linalg.eigh(Hamil)
    return E, C, Hamil


def cas_space_vec(Cvec, up_el, norb):
    AE = np.zeros((len(Cvec), norb))
    for ii in range(norb):
        if norb % 2 == 0:
            AE[:, ii] = Cvec[:, up_el - int(0.5 * norb) + ii]
        else:
            AE[:, ii] = Cvec[:, up_el - int(0.5 * (norb+1)) + ii ]
    return AE


def construct_Hfile(Hamil, Natom, AE, norb, U, up_el, C):
    Hfile = []
    for j in range(1, norb+1):
        for k in range(j, norb+1):
            W = 0
            for a in range(1, Natom+1):
                for b in range(1, Natom+1):
                    W += Hamil[a-1,b-1]*(AE[a-1,j-1]*AE[b-1,k-1])
            Hfile.append([W, int(j), int(0), int(0), int(k)])

    # Define Ncore
    if norb % 2 == 0:
        Ncore = up_el - 0.5 * norb
    else:
        Ncore = up_el - 0.5 * (norb + 1) 

    # Add effective field of the "deep" orbitals
    for j in range(1, norb+1):
        for k in range(j, norb+1):
            W = 0
            for n in range(1, int(Ncore)+1):
                W += U * np.sum(AE[:, j-1] * AE[:, k-1] * C[:, n-1] * C[:, n-1])
            Hfile.append([W, j, int(0), int(0), k])
 

    for j in range(1, norb+1):
        for jp in range(1, norb+1):
            for k in range(1, norb+1):
                for kp in range(1, norb+1):
                    # many-body parameter for this interaction
                    V = U*np.sum(AE[:,j-1]*AE[:,jp-1]*AE[:,k-1]*AE[:,kp-1])
                    Hfile.append([V, int(j), int(-k), int(-kp), int(jp)])
                    #Hfile = np.vstack([Hfile, [V, -j, k, kp, -jp]])
                    #Hfile = np.vstack([Hfile, [V, j, k, kp, jp]])
                    #Hfile = np.vstack([Hfile, [V, -j, -k, -kp, -jp]])
    Hfile = np.array(Hfile)
    return Hfile

def hfile_2_fcidump(Hfile):
    def Hfile_2_one_two_body_Hfile(Hfile):
        Hfile1body = Hfile[Hfile[:,2]==0]
        Hfile2body = Hfile[Hfile[:,2]!=0]
        return Hfile1body, Hfile2body
    Hfile1body, Hfile2body = Hfile_2_one_two_body_Hfile(Hfile)
    orb = int(np.max(Hfile[:,1:]))
    two_e = np.zeros((orb,orb,orb,orb))
    one_e = np.zeros((orb,orb))
    for i in range(len(Hfile2body)):
        a,b,c,d=np.abs(Hfile2body[i,1:]).astype(int) 
        two_e[a-1, d-1, b-1, c-1] += Hfile2body[i,0]
    for i in range(len(Hfile1body)):
        a,b = (np.abs(Hfile1body[i,(1,4)])).astype(int) -1
        one_e[b,a] += Hfile1body[i,0]
    # create FCIDUMP file
    fout = open("FCIDUMP", "w")
    # header
    fout.write("&FCI NORB={:d},NELEC={:d},MS2=0 \n".format(orb, orb))
    aux = ""
    for i in range(0, orb):
        aux = aux + "1,"
    fout.write(" ORBSYM={:s} \n".format(aux))
    fout.write(" ISYM=1, \n")
    fout.write("&END  \n ")

    ij = 0
    for i in range(len(two_e)):
        for j in range(0, i+1):
            kl = 0
            for k in range(0, i+1):
                for l in range(0, k+1):
                    if ij >= kl:
                        coef = '{:.12e}' .format(two_e[i,j,k,l])
                        fout.write('{:<30} {:>5} {:>5} {:>5} {:>5} \n'.format(coef, i+1, j+1, k+1, l+1 ))
                    kl += 1
            ij += 1
    for i in range(len(one_e)):
        for j in range(0, i+1):
            coef = '{:.12e}' .format(one_e[i,j])
            fout.write('{:<30} {:>5} {:>5} {:>5} {:>5} \n'.format(coef, i+1, j+1, 0, 0))
    fout.write('{:<30} {:>5} {:>5} {:>5} {:>5}' .format(0, 0, 0, 0, 0))
    fout.close()

def save_orb(cgeom, near_neigh, vect, vec_f_name, title=None):
    def visualize_backbone(ax, atoms, neighbor_list):
        i_arr, j_arr = neighbor_list
        for i, j in zip(i_arr, j_arr):
            if i < j:
                p1 = atoms[i]
                p2 = atoms[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.0, solid_capstyle='round')

    def visualize_evec(ax, cord, evec):
        #area = 0.0
        #vol = 0.0
        for at, e in zip(cord, evec):
            p = at
            # mod = 1.2 * np.abs(e)  # normalized area
            mod = np.abs(e)**(2/3) # normalized vol
            phase = np.abs(np.angle(e) / np.pi)
            col = (1.0 - phase, 0.0, phase)
            circ = plt.Circle(p[:2], radius=mod, color=col, zorder=2)
            ax.add_artist(circ)

    def make_evec_plot(ax, atoms, neighbor_list, data, title=None, filename=None):
        ax.set_aspect('equal')
        visualize_backbone(ax, atoms, neighbor_list)
        visualize_evec(ax, atoms, data)
        ax.axis('off')
        xmin = np.min(cgeom[:, 0]) - 2.0
        xmax = np.max(cgeom[:, 0]) + 2.0
        ymin = np.min(cgeom[:, 1]) - 2.0
        ymax = np.max(cgeom[:, 1]) + 2.0
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(title)
        if filename is not None:
            plt.savefig('%s.png' % filename, dpi=300, bbox_inches='tight')
            # plt.savefig('%s.pdf' % filename, bbox_inches='tight')

    ax = plt.gca()
    make_evec_plot(ax,cgeom, near_neigh, vect, filename=vec_f_name, title=title)
    plt.close()


def plot_sts_map(cgeom, evec, neighbor_list, filename=None, plot_bond=False, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):
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

    def save_sts_map(cgeom, evec, neighbor_list, filename=filename, plot_bond=plot_bond, h=h, edge_space=edge_space, dx=dx, z_eff=z_eff):
        final_map, extent = calc_sts_map(cgeom, evec, h=h, edge_space=edge_space, dx=dx, z_eff=z_eff)
        if filename:
            plt.imshow(final_map.T, origin='lower', cmap="gist_heat", extent=extent)
            if plot_bond:
                visualize_backbone(cgeom, neighbor_list)
            plt.axis('off')
            plt.savefig('%s.png' % filename, dpi=300, bbox_inches='tight')
            plt.close()
        return final_map, extent
    final_map, extent= save_sts_map(cgeom, evec, neighbor_list, filename=filename, plot_bond=plot_bond, h=h, edge_space=edge_space, dx=dx, z_eff=z_eff)
    if not filename:
        return final_map, extent


def Int_z_FCIDUMP(int_file="FCIDUMP"):
    result = tools.fcidump.read(int_file)
    h1e = result['H1']
    h2e = result['H2']
    norb = result['NORB']
    nelec = result['NELEC']
    ecore = result['ECORE']
    Sz = result['MS2']
    return h1e, h2e, norb, nelec, ecore, Sz

def CI_large(C0, norb, nelec, weight=0.08, state =0):
    fcivec = fci.addons.large_ci(C0,norb, nelec, tol=weight ,return_strs=False)
    coff_arr, alpha, beta = [], [], []
    for i in range(len(fcivec)):
        coff_arr.append(fcivec[i][0])
        alpha.append(fcivec[i][1])
        beta.append(fcivec[i][2])
    abasis = np.zeros((len(alpha),norb))
    bbasis = np.zeros((len(alpha),norb))
    for i in range(len(alpha)):
        abasis[i][alpha[i]] = 3
        bbasis[i][beta[i]] = -1
    basis = abasis + bbasis
    basis[basis ==3 ] = 1
    coff_arr = np.array(coff_arr)
    idx = np.argsort(np.abs(coff_arr))[::-1]
    basis = basis[idx];coff_arr = coff_arr[idx]; basis = basis.astype(int)
    print(f'\nWeight of slater determinant of Ψ = {state} \nWEIGHT \t\t COEFF \t\t\t CI')
    for i in range(len(basis)):
        print(f'{str(round( coff_arr[i]**2,3)):<5}{str(round(coff_arr[i],3)):>15}{str(basis[i]):>30}')

def Run_FCI(h1e, h2e, norb, nelec, nroots=8, ecore=0 ,max_space=30, max_cycle=100, verbos = True):
    st = time.time()
    Print_input_parameters(norb,nelec,nelec,2,'YES')
    Energies, fcivec = fci.direct_spin1.kernel(h1e, h2e, norb, nelec, nroots=nroots, ecore=ecore ,max_space=max_space, max_cycle=max_cycle)
    if verbos:
        print('\nSTATE \t\t ENERGY(eV) \t\t Spin \t\t S^2')
        for i, c in enumerate(fcivec):
            S2 = fci.spin_op.spin_square(c, norb, nelec)[0]
            Szz = 0.5*(-1+np.sqrt(1+4*S2))
            print('Ψ = %d \t\t  %.4f \t\t  %.2f \t\t %.2f' % (i, Energies[i], Szz, S2))
        print(f'Energy gap of ground state and excited state is  {np.round((Energies[1]-Energies[0]),5)}')
        CI_large(fcivec[0], norb, nelec, weight=0.08, state=0)
        CI_large(fcivec[1], norb, nelec, weight=0.08, state=1)
        CI_large(fcivec[2], norb, nelec, weight=0.08, state=2)
    et = time.time();Print_time_elepsed(st,et)
    return Energies, fcivec

def Run_FCI_S0(h1e, h2e, norb, nelec, nroots=8, ecore=0 ,max_space=30, max_cycle=100, verbos = True):
    st = time.time()
    Print_input_parameters(norb,nelec,nelec,2,'YES')
    Energies, fcivec = fci.direct_spin0.kernel(h1e, h2e, norb, nelec, nroots=nroots, ecore=ecore ,max_space=max_space, max_cycle=max_cycle)
    if verbos:
        print('\nSTATE \t\t ENERGY(eV) \t\t Spin  \t\t S^2')
        for i, c in enumerate(fcivec):
            S2 = fci.spin_op.spin_square(c, norb, nelec)[0]
            Szz = 0.5*(-1+np.sqrt(1+4*S2))
            print('Ψ = %d \t\t  %.4f \t\t  %.2f \t\t %.2f' % (i, Energies[i], Szz, S2))
        CI_large(fcivec[0], norb, nelec, weight=0.08, state=0)
        CI_large(fcivec[1], norb, nelec, weight=0.08, state=1)
        #CI_large(fcivec[2], norb, nelec, weight=0.08, state=2)
        #CI_large(fcivec[3], norb, nelec, weight=0.08, state=3)
        #CI_large(fcivec[4], norb, nelec, weight=0.08, state=4)
    et = time.time();Print_time_elepsed(st,et)
    return Energies, fcivec

def py_basis(C0, norb, nelec):
    fcivec = fci.addons.large_ci(C0,norb, nelec, tol= 0.0 ,return_strs=False)
    coff_arr, alpha, beta = [], [], []
    for i in range(len(fcivec)):
        coff_arr.append(fcivec[i][0])
        alpha.append(fcivec[i][1])
        beta.append(fcivec[i][2])
    abasis = np.zeros((len(alpha),norb))
    bbasis = np.zeros((len(alpha),norb))
    for i in range(len(alpha)):
        abasis[i][alpha[i]] = 3
        bbasis[i][beta[i]] = -1
    basis = abasis + bbasis
    basis[basis ==3 ] = 1
    coff_arr = np.array(coff_arr)
    return basis.astype(int), coff_arr

def natural_orb(C0, norb, nelec ):
    p1 = fci.direct_spin1.make_rdm1(C0,norb, nelec)
    n_nat,c_nat = np.linalg.eigh(p1)
    idx = np.argsort(n_nat)[::-1]
    n_nat = n_nat[idx];n_nat = np.round(n_nat,2)
    c_nat = c_nat[:,idx]
    print("\nOccupation of Natural Orbitals are:"); print(n_nat)
    return c_nat

def py_trdm1(my, C0, n, m, norb, nelec, atomic_basis):
    trdm = fci.direct_spin1.trans_rdm1(C0[n],C0[m], norb, nelec)
    n_rho = (atomic_basis@trdm@atomic_basis.T)
    n_rho = np.diagonal(n_rho);print(f'Norm Of tdm {n} to {m} is {np.linalg.norm(n_rho)}')
    n_rho = n_rho / np.linalg.norm(n_rho)
    save_orb(my.geom, my.near_neigh, n_rho, 'trans_n_m_')

def _unpack_NTO_el( nel):
    if (nel % 2) == 0:
        a_el = int((nel // 2)+1)
        b_el = int((nel // 2)-1)
    else:
        a_el = int((nel // 2)+2)
        b_el = int((nel // 2)-1)
    return (a_el, b_el )

def make_rdm1_s2t( bra, ket, norb, nelec_ket):
    '''Inefficient version. A check for make_rdm1_t2s'''
    neleca, nelecb = nelec = nelec_ket
    ades_index = fci.cistring.gen_des_str_index(range(norb), neleca)
    bcre_index = fci.cistring.gen_cre_str_index(range(norb), nelecb)
    na_bra = fci.cistring.num_strings(norb, neleca-1)
    nb_bra = fci.cistring.num_strings(norb, nelecb+1)
    na_ket = fci.cistring.num_strings(norb, neleca)
    nb_ket = fci.cistring.num_strings(norb, nelecb)
    assert bra.shape == (na_bra, nb_bra)
    assert ket.shape == (na_ket, nb_ket)
    t1ket = np.zeros((na_bra,nb_ket,norb))
    for str0, tab in enumerate(ades_index):
        for _, i, str1, sign in tab:
            t1ket[str1,:,i] += sign * ket[str0]

    t1bra = np.zeros((na_bra,nb_bra,norb,norb))
    for str0, tab in enumerate(bcre_index):
        for a, _, str1, sign in tab:
            t1bra[:,str1,a] += sign * t1ket[:,str0]
    dm1 = np.einsum('ab,abpq->pq', bra, t1bra)
    return dm1

def make_rdm1_t2s( bra, ket, norb, nelec_ket):
    neleca, nelecb = nelec = nelec_ket
    ades_index = fci.cistring.gen_des_str_index(range(norb), neleca+1)
    bdes_index = fci.cistring.gen_des_str_index(range(norb), nelecb)
    na_bra = fci.cistring.num_strings(norb, neleca+1)
    nb_bra = fci.cistring.num_strings(norb, nelecb-1)
    na_ket = fci.cistring.num_strings(norb, neleca)
    nb_ket = fci.cistring.num_strings(norb, nelecb)
    assert bra.shape == (na_bra, nb_bra)
    assert ket.shape == (na_ket, nb_ket)
    t1bra = np.zeros((na_ket,nb_bra,norb))
    t1ket = np.zeros((na_ket,nb_bra,norb))
    for str0, tab in enumerate(bdes_index):
        for _, i, str1, sign in tab:
            t1ket[:,str1,i] += sign * ket[:,str0]
    for str0, tab in enumerate(ades_index):
        for _, i, str1, sign in tab:
            t1bra[str1,:,i] += sign * bra[str0,:]
    dm1 = np.einsum('abp,abq->pq', t1bra, t1ket)
    return dm1

def trans_t2s(bra,ket, norb, nelec_ket,):
    dm1 = make_rdm1_t2s(bra,ket, norb, nelec_ket)
    c,d = np.linalg.eig(dm1@dm1.T)
    print(f'NTO coefficients are:\n {np.round(c,3)}')
    return np.round(c,3), d

def py_NTO( C0, h1e, h2e, norb, nelec, ecore = 0):
    nto_elec = _unpack_NTO_el(nelec)
    E1,C1 = Run_FCI( h1e, h2e, norb, nto_elec, ecore = ecore , verbos= True)
    nelec_k = fci.addons._unpack_nelec(nelec=nelec)
    eigen_valu, NTOs = trans_t2s(bra=C1[0],ket=C0[0], norb=norb, nelec_ket=nelec_k)
    return eigen_valu, NTOs

def make_tr_1e_rm(bra, ket, norb, nelec_ket):
    neleca, nelecb = nelec = nelec_ket
    ades_index = fci.cistring.gen_des_str_index(range(norb), neleca)
    bra = np.array(bra); ket = np.array(ket)
    Fm = np.zeros((norb,1))
    if neleca==nelecb:
        for str0, tab in enumerate(ades_index):
            for _, i, str1,sign in tab:
                Fm[i] += sign * np.einsum('i,i->',ket[:,str0],bra[:,str1])
    else:
        for str0, tab in enumerate(ades_index):
            for _, i, str1,sign in tab:
                Fm[i] += sign * np.einsum('i,i->',ket[str0],bra[str1])
    return Fm

def make_tr_1e_add(bra, ket, norb, nelec_ket):
    neleca, nelecb = nelec = nelec_ket
    ades_index = fci.cistring.gen_cre_str_index(range(norb), nelecb)
    bra = np.array(bra); ket = np.array(ket)
    Fm = np.zeros((norb,1))
    if neleca==nelecb:
        for str0, tab in enumerate(ades_index):
            for i, _, str1,sign in tab:
                Fm[i] += sign * np.einsum('i,i->',ket[str0],bra[str1])
    else:
        for str0, tab in enumerate(ades_index):
            for i, _, str1,sign in tab:
                Fm[i] += sign * np.einsum('i,i->',ket[:,str0],bra[:,str1])
    return Fm

def py_Dysons(C0, h1e, h2e, norb, nelec, geom, near_neigh, cas_mo,  ecore = 0, nstat =1 ):
    Em,Cm = Run_FCI( h1e, h2e, norb, int(nelec-1), ecore = ecore )
    EM,CM = Run_FCI( h1e, h2e, norb, int(nelec+1), ecore = ecore )
    nelec_k = fci.addons._unpack_nelec(nelec=nelec)
    for i in range(nstat):
        Fm = make_tr_1e_rm(bra=Cm[i], ket=C0[0], norb=norb, nelec_ket=nelec_k)
        FM = make_tr_1e_add(bra=CM[i], ket=C0[0], norb=norb, nelec_ket=nelec_k)
        Fm = cas_mo@Fm; FM = cas_mo@FM
        print(f"\nNorm of -1e and +1e For n = {i}")
        n_Fm, n_FM = np.round(np.linalg.norm(Fm),3), np.round(np.linalg.norm(FM),3)
        print('\t',n_Fm,'\t',n_FM)
        Fm = Fm / np.linalg.norm(Fm)
        FM = FM / np.linalg.norm(FM)
        Fm = np.concatenate(Fm)
        FM = np.concatenate(FM)
        save_orb(geom, near_neigh, Fm, f'Dyson_rm_e_{i}_', n_Fm)
        save_orb(geom, near_neigh, FM, f'Dyson_add_e_{i}_', n_FM )
        plot_sts_map(geom, Fm, near_neigh, f'sts_dyson_rm_e_{i}_')
        plot_sts_map(geom, FM, near_neigh, f'sts_dyson_add_e_{i}_')


def cal_charge(C0, norb, nelec):
    C0 = np.array(C0)
    neleca, nelecb = fci.addons._unpack_nelec(nelec=nelec)
    inda = fci.cistring._gen_occslst(range(norb), neleca)
    indb = fci.cistring._gen_occslst(range(norb), nelecb)
    charges_up = np.zeros(norb)
    charges_down = np.zeros(norb)
    for i in range(norb):
        posn_a = np.argwhere(inda==i)
        posn_b = np.argwhere(indb==i)
        posn_a = posn_a[:,0]
        posn_b = posn_b[:,0]
        cmat_a = np.einsum("ij, ij -> ij", C0[posn_a], C0[posn_a])
        charges_up[i] += np.sum(cmat_a)
        cmat_b = np.einsum("ij, ij -> ij", C0[:,posn_b], C0[:,posn_b])
        charges_down[i] += np.sum(cmat_b)
    return np.round(charges_up,2), np.round(charges_down,2)   

class HDSPMM_CODE:
    def __init__(self,xyz_file, U, t, norb):
        Print_Title()
        self.geom = read_geom(xyz_file)
        self.near_neigh =nearest_neighbout(cgeom=self.geom)
        self.U = U
        self.t = t
        self.norb = norb
        self.Huckel_En, self.Huckel_Cvec, self.Huckel_Hamil = Huckel_en_vec(len(self.geom), cgeom=self.geom, t= self.t)
        self.up_el, self.dn_el = _unpack_el(len(self.geom))
        self.cas_mo = cas_space_vec(self.Huckel_Cvec, self.up_el, self.norb)
        self.Hfile = construct_Hfile(self.Huckel_Hamil, len(self.geom), self.cas_mo, self.norb, self.U, self.up_el, self.Huckel_Cvec)
        hfile_2_fcidump(self.Hfile)
        result = tools.fcidump.read("FCIDUMP")
        self.h1e = result['H1']
        self.h2e = result['H2']
        self.norb = result['NORB']
        self.nelec = result['NELEC']
        self.ecore = result['ECORE']
        self.Sz = result['MS2']

    def Int_z_FCIDUMP(self, FCIDUMP=None):
        if not FCIDUMP:
            FCIDUMP='FCIDUMP'
        return Int_z_FCIDUMP(int_file=FCIDUMP)

    def Run_FCI(self):
        h1e, h2e, norb, nelec, ecore = self.h1e, self.h2e, self.norb, self.nelec, self.ecore
        self.En_gr, self.fvec_gr = Run_FCI(h1e, h2e, norb, nelec, nroots=8, ecore=ecore ,max_space=30, max_cycle=100, verbos = True)
        return self.En_gr, self.fvec_gr

    def Run_FCI_S0(self):
        h1e, h2e, norb, nelec, ecore = self.h1e, self.h2e, self.norb, self.nelec, self.ecore
        self.En_gr, self.fvec_gr = Run_FCI_S0(h1e, h2e, norb, nelec, nroots=8, ecore=ecore ,max_space=30, max_cycle=100, verbos = True)
        return self.En_gr, self.fvec_gr

    def py_basis(self):
        C0, norb, nelec = self.fvec_gr[0], self.norb, self.nelec
        self.basis, self.coeff_arr = py_basis(C0, norb, nelec)
        return self.basis, self.coeff_arr

    def CI_large(self, state=1):
        fcivec = self.fvec_gr
        for i in range(state):
            CI_large(fcivec[i], self.norb, self.nelec, weight=0.08, state =i)
    
    def plot_Huckel_orb(self):
        orbs = self.cas_mo
        for i in range(self.norb):
            save_orb(self.geom, self.near_neigh, orbs[:,i], f'Huck_orb_{i}_')

    def natural_orb(self):
        nat_orb = natural_orb(self.fvec_gr[0], self.norb, self.nelec)
        nat_vec_ao = self.cas_mo @ nat_orb
        for i in range(len(nat_orb)):
            save_orb(self.geom, self.near_neigh, nat_vec_ao[:,i], f"natural_orb_{i}_")

    def py_trdm1(self, n, m):
        py_trdm1(self, self.fvec_gr, n, m, self.norb, self.nelec)

    def NTO(self):
        eigen_valu, NTOs = py_NTO(self.fvec_gr, self.h1e, self.h2e, self.norb, self.nelec, self.ecore)
        ntos_ao = self.cas_mo @ NTOs
        final_map = None
        for i in range(len(NTOs)):
            save_orb(self.geom, self.near_neigh, ntos_ao[:,i], f"NTOs_{i}_", eigen_valu[i])
            final_map_i, extent = plot_sts_map(self.geom, ntos_ao[:,i], self.near_neigh)
            if final_map is None:
                final_map = final_map_i * eigen_valu[i]
            else:
                final_map += final_map_i * eigen_valu[i]
        plt.imshow(final_map.T, origin='lower', cmap="gist_heat", extent=extent)
        plt.axis('off')
        plt.savefig('%s.png' % "sts_NTO", dpi=300, bbox_inches='tight')
        plt.close()        

    def dysons(self, nstat=1):
        py_Dysons(self.fvec_gr, self.h1e, self.h2e, self.norb, self.nelec, self.geom,  self.near_neigh, self.cas_mo,  self.ecore, nstat=nstat )

    def cal_charge(self):
        up, down = cal_charge(self.fvec_gr[0], self.norb, self.nelec)
        return up, down

    def __del__(self):
        print("\n\t\t\tHAPPY LANDING!")
        print("\nWhat I cannot create I cannot understand - Richard Feynman\n")

