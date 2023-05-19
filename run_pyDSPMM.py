#!/auto/vestec1-elixir/home/manishkumar/.conda/envs/kpython310/bin/python3.1
import sys
sys.path.append('/storage/praha1/home/manishkumar/mybin/module/DSPMM')
from DSPMM import DSPMM_CODE

my = DSPMM_CODE()
my.Run_FCI()
my.natural_orb()
my.NTO()
my.dysons(4)

#a=my.cal_charge()



