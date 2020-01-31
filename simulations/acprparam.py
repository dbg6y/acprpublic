#!/miniconda3/envs/py3_env/bin/python


### Create json parameter files for simulations


## Import Modules
import numpy as np
import itertools
import json


## Create template parameter dictionary
para_temp = {'RUAS': {'NCOM_RUAS': 10, 'UPRT_RUAS': 0.1, 'FCIN_RUAS': 'NA', 
                      'SANC_RUAS': 'NA'}, 
             'COMM': {'NMEM_COMM': 50, 'UPRT_COMM': 0.02, 'FCIN_COMM': 'NA', 
                      'ALPH_COMM': 0.6, 'BETA_COMM': 0.2, 'GAMM_COMM': 10,
                      'HPAR_COMM': 0.33, 'TPAR_COMM': -150, 'GPAR_COMM': -10, 
                      'RINT_COMM': 50, 'RMAX_COMM': 200, 'RUMX_COMM': '2 * RMAX_COMM / 3', 
                      'KPAR_COMM': 2, 'QPAR_COMM': 1}, 
             'MEMB': {'MPAR_MEMB': 'NA', 'EPAR_MEMB': '0.483 / NMEM_COMM', 'WPAR_MEMB': 15}, 
             'RSCE': {'FLMN_RSCE': 'NA', 'FLSD_RSCE': 'NA'}, 
             'SIMU': {'NMBR_SIMU': 'NA', 'NSTP_SIMU': 10000, 'NRUN_SIMU': 10000},
             'DBSE_FILE': 'results/acprresult.db'}


## Define simulation-specific parameters
flmn_rsce = ['30 * NCOM_RUAS', '50 * NCOM_RUAS', '70 * NCOM_RUAS']
flsd_rsce = ['10 * NCOM_RUAS']
sanc_ruas = [5, 10]
mpar_memb = [3]
fcin_ruas = [0.2, 0.5, 0.8]
fcin_comm = [0.5, 0.7, 0.9]
para_simu = list(itertools.product(flmn_rsce, flsd_rsce, sanc_ruas, mpar_memb, fcin_ruas, fcin_comm))


## Loop through simulations
for indx, para in enumerate(para_simu):
    
    # Define new dictionary
    para_data = para_temp
    
    # Fill in simulation-specific parameters
    para_data['SIMU']['NMBR_SIMU'] = indx
    para_data['RSCE']['FLMN_RSCE'] = para[0]
    para_data['RSCE']['FLSD_RSCE'] = para[1]
    para_data['RUAS']['SANC_RUAS'] = para[2]
    para_data['MEMB']['MPAR_MEMB'] = para[3]
    para_data['RUAS']['FCIN_RUAS'] = para[4]
    para_data['COMM']['FCIN_COMM'] = para[5]
       
    # Save as json file
    with open('acprparam_' + ('%02d' % indx) + '.json', 'w') as para_file:
        json.dump(para_data, para_file, indent = 2)
