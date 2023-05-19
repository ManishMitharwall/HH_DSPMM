#!/auto/vestec1-elixir/home/manishkumar/.conda/envs/kpython310/bin/python3.1
import numpy as np
from pyscf import fci, tools
import time, os, re
os.environ['OMP_NUM_THREADS'] = "4"

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
            print('Ψ = %d \t\t  %.4f \t\t  %.2f \t\t %.2f' % (i, Energies[i]*27.2114, Szz, S2))
        print(f'Energy gap of ground state and excited state is  {np.round((Energies[1]-Energies[0])*27.2114,5)}')
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
            print('Ψ = %d \t\t  %.4f \t\t  %.2f \t\t %.2f' % (i, Energies[i]*27.2114, Szz, S2))
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
    write_molden("cas_orbital.molden.input",c_nat,'natural_orbitals.molden.input')

def py_trdm1( C0, n, m, norb, nelec):
    trdm = fci.direct_spin1.trans_rdm1(C0[n],C0[m], norb, nelec)
    atomic_basis, Energies = getMOLDENcoefs("cas_orbital.molden.input")
    f_out_file = open(f'tdm_{n}_{m}.molden.input','w')
    a_len=len(atomic_basis)
    n_rho = (atomic_basis@trdm@atomic_basis.T)
    n_rho = np.diagonal(n_rho);print(f'Norm Of tdm {n} to {m} is {np.linalg.norm(n_rho)}')
    n_rho = n_rho / np.linalg.norm(n_rho)
    with open("cas_orbital.molden.input",'r') as mfile:
        for line in mfile:
            f_out_file.write(line)
            if r'MO' in line: break
    counter= -0.1500
    for p in range(1):
        f_out_file.write(' Sym=      1a\n Ene= '+str(counter)+'\n'+ ' Spin= Alpha\n Occup= 1.000000 \n')
        counter+=0.010
        for t in range(int(a_len)):
            f_out_file.write(f' {t+1} \t {n_rho[t]}\n')
    f_out_file.close()

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

def trans_t2s(bra,ket, norb, nelec_ket):
    dm1 = make_rdm1_t2s(bra,ket, norb, nelec_ket)
    c,d = np.linalg.eig(dm1@dm1.T)
    print(f'NTO coefficients are:\n {np.round(c,3)}')
    write_molden("cas_orbital.molden.input",d,'NTO_orbitals.molden.input')

def py_NTO( C0, h1e, h2e, norb, nelec, ecore = 0):
    nto_elec = _unpack_NTO_el(nelec)
    E1,C1 = Run_FCI( h1e, h2e, norb, nto_elec, ecore = ecore , verbos= True)
    nelec_k = fci.addons._unpack_nelec(nelec=nelec)
    trans_t2s(bra=C1[0],ket=C0[0], norb=norb, nelec_ket=nelec_k)

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

def py_Dysons(C0, h1e, h2e, norb, nelec, ecore = 0, nstat = 1 ):
    Em,Cm = Run_FCI( h1e, h2e, norb, int(nelec-1), ecore = ecore )
    EM,CM = Run_FCI( h1e, h2e, norb, int(nelec+1), ecore = ecore )
    nelec_k = fci.addons._unpack_nelec(nelec=nelec)
    for i in range(nstat):
        Fm = make_tr_1e_rm(bra=Cm[i], ket=C0[0], norb=norb, nelec_ket=nelec_k)
        FM = make_tr_1e_add(bra=CM[i], ket=C0[0], norb=norb, nelec_ket=nelec_k)
        _dyson_2_molden(Fm=Fm,FM=FM,nstat=i)
    
def _dyson_2_molden(Fm,FM,nstat):
    atomic_basis, Energies = getMOLDENcoefs("cas_orbital.molden.input")
    Fm = atomic_basis@Fm; FM = atomic_basis@FM
    f_out_file = open(f'dysons_{nstat}_orbitals.molden.input','w')
    print(f"\nNorm of -1e and +1e For n = {nstat}")
    print('\t',np.round(np.linalg.norm(Fm),3),'\t',np.round(np.linalg.norm(FM),3))
    Fm = (Fm / np.linalg.norm(Fm)).T
    FM = (FM / np.linalg.norm(FM)).T
    F = np.concatenate((Fm,FM))
    with open("cas_orbital.molden.input",'r') as mfile:
        for line in mfile:
            f_out_file.write(line)
            if r'MO' in line: break   
    counter= -0.1500
    for p in range(len(F)):
        f_out_file.write(' Sym=      1a\n Ene= '+str(counter)+'\n'+ ' Spin= Alpha\n Occup= 1.000000 \n')
        counter+=0.010
        for t in range(int(len(atomic_basis))):
            f_out_file.write(f' {t+1}        {F[p,t]}\n')
    f_out_file.close()

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

def write_molden(m_file,T_mat,wfile):
    atomic_orb , energies = getMOLDENcoefs(m_file)
    new_basis = atomic_orb @ T_mat
    f_out_file = open(wfile,'w')
    with open(m_file,'r') as mfile:
        for line in mfile:
            f_out_file.write(line)
            if r'MO' in line: break  
    counter= -0.1500
    for p in range(len(T_mat)):
        f_out_file.write(' Sym=      1a\n Ene= '+str(counter)+'\n'+ ' Spin= Alpha\n Occup= 1.000000 \n')
        counter+=0.010
        for t in range(len(new_basis)):
            f_out_file.write(f' {t+1}     {new_basis[t,p]}\n')
    f_out_file.close()

def getMOLDENcoefs(fname):
    coeff = []
    with open(fname,'r') as mfile:
        while r"MO" not in mfile.readline():continue
        Energies =[];co_line = []
        for line in mfile:
            if re.search(r'Ene',line):
                split = line.split('=');
                if co_line: coeff.append(co_line); co_line = []
                Energies.append(float(split[-1]))
            if '=' not in line:
                    line = ' '.join(line.split()); split = line.split(' ')
                    co_line.append(float(split[-1]))
            if r'[' in line: break
        coeff.append(co_line)
        coeff = np.array(coeff).T
    return coeff, Energies

def plot_molden_cube(molden_file, i_start, i_end, out_file):
    mol, _, mo_coeff, _, _, _ = tools.molden.load(molden_file)
    for i in range(i_start, i_end+1):
        tools.cubegen.orbital(mol, f'{out_file}_{i}_.cube', mo_coeff[:,i])


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



class DSPMM_CODE:
    def __init__(self,int_file='FCIDUMP'):
        Print_Title()
        result = tools.fcidump.read(int_file)
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

    def natural_orb(self):
        natural_orb(self.fvec_gr[0], self.norb, self.nelec)
   
    def py_trdm1(self, n, m):
        py_trdm1(self.fvec_gr, n, m, self.norb, self.nelec)

    def NTO(self):
        py_NTO(self.fvec_gr, self.h1e, self.h2e, self.norb, self.nelec, self.ecore)

    def dysons(self, nstat=1):
        py_Dysons(self.fvec_gr, self.h1e, self.h2e, self.norb, self.nelec, self.ecore, nstat=nstat )

    def cal_charge(self):
        up, down = cal_charge(self.fvec_gr[0], self.norb, self.nelec)
        return up, down

    def __del__(self):
        print("\n\t\t\tHAPPY LANDING!")
        print("\nWhat I cannot create I cannot understand - Richard Feynman\n")
    
    
    

            
    
