import sys
sys.path.append('/')
from  HH_DSPMM import HDSPMM_CODE


U = 4.3                              # On-site repulsion
t = -2.8                             # Hopping in eV
M = 6                                # Active space orbitals

my= HDSPMM_CODE('inp.xyz', U, t, M)  # READ GEOMETRY  
my.plot_Huckel_orb()                 # To plot the Huckel orbitals
my.Run_FCI()                         # To run Many-body calculations
my.natural_orb()                     # To calculatet the natural orbitals
my.NTO()                             # To calculate NTOs
my.dysons(3)                         # To calculate Dysons






