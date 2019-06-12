import sys, os
from subprocess import check_output
import argparse

'''used to agglomerate awsm_daily run outputs into one .nc file'''
#ZRU 6/12/2019

fp = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(prog = 'ncr_cl.py')
parser.add_argument('--dir-in', default = os.getcwd(), help = 'directory where awsm_daily run folders reside')
parser.add_argument('--dir-out', default = fp, help = 'directory location to save output .nc file')
parser.add_argument('--base-file', default = 'snow.nc', help = 'output nc file from awsm isnobal, generally snow.nc or em.nc')
parser.add_argument('--file-name-out', required = True, help = 'filename of new file')
args = parser.parse_args()

subD = os.listdir(args.dir_in)
subD.sort()
lst = []
for sub in subD:
    lst.append(sub + '/' + args.base_file)
s = ' '.join(lst)

cmd = 'ncrcat {} {}'.format(s, args.dir_out + '/' + args.file_name_out +'.nc')
print(cmd)
check_output(cmd, shell=True)  # t = text output
