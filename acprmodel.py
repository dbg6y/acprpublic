#!/miniconda3/envs/py3_env/bin/python


#### Assymetrical Common Pool Resource Model



## Import Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial
import sqlite3
import multiprocessing
import json
import sys

## define objects
class Resource(object):
    '''The flow of resource through the RUAS communities
    
    Attributes: flow_rsce, abfl_rsce
    '''
    
    def __init__(self):
        
        # Set initial values
        flow_rsce = np.random.gamma(FLMN_RSCE**2 / FLSD_RSCE**2, 
                                    FLSD_RSCE**2 / FLMN_RSCE, NSTP_SIMU)
        self.flow_rsce = np.array(flow_rsce, dtype = float)
        self.abfl_rsce = np.array(flow_rsce, dtype = float)


class ResourceUsersAssociation(object):
    '''The association of communities harvesting an asymmetric resource
    
    Attributes: stin_ruas, fcrs_ruas, uprs_ruas, comm_ruas
    '''
    def __init__(self, rsce):
        
        # Set initial values
        self.stin_ruas = np.full([NCOM_RUAS], -99999, dtype = int)
        self.uprs_ruas = rsce.flow_rsce / NCOM_RUAS
        self.sanc_ruas = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.fcrs_ruas = np.full([NSTP_SIMU], np.nan, dtype = float)
        
        # Assign initial statuses
        nres_ruas = int(round(FCIN_RUAS * NCOM_RUAS))
        nunr_ruas = int(round(NCOM_RUAS - nres_ruas))
        stin_ruas = np.append(np.ones(nres_ruas), np.zeros(nunr_ruas))
        np.random.shuffle(stin_ruas)
        self.stin_ruas = stin_ruas
        
        # Create communities
        self.comm_ruas = {}
        for cmid_comm in range(NCOM_RUAS):
            comm = Community(self)
            comm.cmid_comm = cmid_comm
            self.comm_ruas.update({cmid_comm:comm})
             
    def rsce_step(self, rsce, step_simu):
        '''Simulates community resource use and status changes'''
        
        # Determine each community's uptake and adjust flow
        for comm in self.comm_ruas.values():
            comm.rsce_step(rsce, self, step_simu)
            rsce.abfl_rsce[step_simu] = rsce.abfl_rsce[step_simu] \
                                        - comm.uptk_comm[step_simu]
            if rsce.abfl_rsce[step_simu] < 0:
                rsce.abfl_rsce[step_simu] = 0
    
    def util_step(self, step_simu):
        '''Calculates the utility for each community'''
        
        # Set step values
        if step_simu < 1:
            fcrs_step = FCIN_RUAS
        else:
            fcrs_step = self.fcrs_ruas[step_simu - 1]
        
        # Determine if sanctioning occurs
        if np.random.random() < fcrs_step:
            self.sanc_ruas[step_simu] = SANC_RUAS
        else:
            self.sanc_ruas[step_simu] = 0
        
        # Calculate each community's utility
        for comm in self.comm_ruas.values():
            comm.util_step(self, step_simu)
            
    def swap_step(self, step_simu):
        '''Swaps community and member statuses based on utility'''
        
        # Assign new statuses
        for comm in self.comm_ruas.values():
            if step_simu < 1: 
                comm.stus_comm[step_simu] = self.stin_ruas[comm.cmid_comm]
            else:
                comm.stus_comm[step_simu] = comm.stus_comm[step_simu - 1]
        
        # Choose communities to update
        def swap_chck(cmid_ruas, swln_ruas):
            cmid_swap = cmid_ruas[np.random.choice(cmid_ruas.shape[0], 
                                                   swln_ruas, replace = False), :]
            if len(np.unique(cmid_swap)) == cmid_swap.size:
                return cmid_swap
            else:
                return swap_chck(cmid_ruas, swln_ruas)
        
        if UPRT_RUAS > 0.5:
            cmid_swap = np.arange(NCOM_RUAS).reshape((int(NCOM_RUAS / 2), 2))
        else:
            cmid_ruas = np.array([(elmt_ruas, elmt_ruas + 1) 
                                      for elmt_ruas in np.arange(NCOM_RUAS)][:-1])
            if NCOM_RUAS == 1:  
                cmid_swap = cmid_ruas
            else:
                cmid_swap = swap_chck(cmid_ruas, int(UPRT_RUAS * NCOM_RUAS))
        
        # Conduct community swaps
        for swid_swap in cmid_swap:
            comm_swap = [self.comm_ruas[swid_ruas] for swid_ruas in swid_swap]
            stus_swap = np.array([comm.stus_comm[step_simu] for comm in comm_swap])
            tlut_swap = np.array([comm.tlut_comm[step_simu] for comm in comm_swap])
            if stus_swap[0] != stus_swap[1]:
                pswp_swap = (tlut_swap[0] - tlut_swap[1]) \
                            / (abs(tlut_swap[0]) + abs(tlut_swap[0]))
                if pswp_swap >= 0 and np.random.random() < pswp_swap:
                    comm_swap[1].stus_comm[step_simu] = stus_swap[0]
                elif pswp_swap < 0 and np.random.random() < abs(pswp_swap):
                    comm_swap[0].stus_comm[step_simu] = stus_swap[1]
         
        # Swap members in each community
        for comm in self.comm_ruas.values():
            comm.swap_step(step_simu)
        
    def updt_step(self, step_simu):
        '''Updates community fractions, effort'''       
        
        # Update restricted fraction
        nres_step = len([comm for comm in self.comm_ruas.values()
                             if comm.stus_comm[step_simu] == 1])
        fcrs_step = nres_step / NCOM_RUAS
        self.fcrs_ruas[step_simu] = fcrs_step
        
        # Update cooperator fraction and total effort
        for comm in self.comm_ruas.values():
            comm.updt_step(step_simu)
             

class Community(object):
    '''Parameters specific to each community
    
    Attributes: cmid_comm, stin_comm, uptk_comm, rlev_comm, util_comm, 
    tlut_comm, stus_comm, fccp_comm, tlef_comm
    '''
    def __init__(self, ruas):
        
        # Set initial values
        self.cmid_comm = None
        self.stin_comm = np.full([NMEM_COMM], -99999, dtype = int)
        self.uptk_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.rlev_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.sanc_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.util_comm = np.full([NSTP_SIMU, 2], np.nan, dtype = float)
        self.tlut_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.stus_comm = np.full([NSTP_SIMU], -99999, dtype = int)
        self.fccp_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        self.tlef_comm = np.full([NSTP_SIMU], np.nan, dtype = float)
        
        # Assign initial statuses
        ncop_comm = int(round(FCIN_COMM * NMEM_COMM))
        ndef_comm = int(round(NMEM_COMM - ncop_comm))
        stin_comm = np.append(np.ones(ncop_comm), np.zeros(ndef_comm))
        np.random.shuffle(stin_comm)
        self.stin_comm = stin_comm
        
        # Create members
        self.memb_comm = {}
        for mbid_memb in range(NMEM_COMM):
            memb = Member()
            memb.mbid_memb = mbid_memb
            memb.cmid_memb = self.cmid_comm
            self.memb_comm.update({mbid_memb:memb})
    
    def rsce_step(self, rsce, ruas, step_simu):
        '''Updates resource for one step'''
        
        # Set previous values
        if step_simu < 1:
            stus_prev = ruas.stin_ruas[self.cmid_comm]
            rlev_prev = RINT_COMM
            tlef_prev = NMEM_COMM * (FCIN_COMM * EPAR_MEMB 
                                     + (1 - FCIN_COMM) * EPAR_MEMB * MPAR_MEMB)
        else:
            stus_prev = self.stus_comm[step_simu - 1]
            rlev_prev = self.rlev_comm[step_simu - 1]
            tlef_prev = self.tlef_comm[step_simu - 1]
        
        # Determine the amount taken up
        if stus_prev == 1:
            uptk_step = min(RUMX_COMM * (1 - (rlev_prev / RMAX_COMM)**KPAR_COMM), 
                            rsce.abfl_rsce[step_simu] * (1 - (rlev_prev / RMAX_COMM)**KPAR_COMM), 
                            ruas.uprs_ruas[step_simu])
        else:
            uptk_step = min(RUMX_COMM * (1 - (rlev_prev / RMAX_COMM)**KPAR_COMM), 
                            rsce.abfl_rsce[step_simu] * (1 - (rlev_prev / RMAX_COMM)**KPAR_COMM))
                
        # Update resource volume
        self.uptk_comm[step_simu] = uptk_step
        self.rlev_comm[step_simu] = rlev_prev + uptk_step - QPAR_COMM \
                                    * tlef_prev * rlev_prev
        if self.rlev_comm[step_simu] < 0:
            self.rlev_comm[step_simu] = 0
    
    def util_step(self, ruas, step_simu):
        
        # Set initial values
        if step_simu < 1:
            fccp_prev = FCIN_COMM
            rlev_prev = RINT_COMM
            tlef_prev = NMEM_COMM * (FCIN_COMM * EPAR_MEMB 
                                     + (1 - FCIN_COMM) * EPAR_MEMB * MPAR_MEMB)
        else:
            fccp_prev = self.fccp_comm[step_simu - 1]
            rlev_prev = self.rlev_comm[step_simu - 1]
            tlef_prev = self.tlef_comm[step_simu - 1]
        
        # Determine whether community is sanctioned
        if self.uptk_comm[step_simu] > ruas.uprs_ruas[step_simu]:
            sanc_step = ruas.sanc_ruas[step_simu]
        else:
            sanc_step = 0
            
        # Calculate production
        cbdb_step = GAMM_COMM * tlef_prev**ALPH_COMM \
                    * rlev_prev**BETA_COMM - sanc_step
        pycp_step = EPAR_MEMB * (cbdb_step / tlef_prev - WPAR_MEMB)
        pydf_step = EPAR_MEMB * MPAR_MEMB * (cbdb_step / tlef_prev - WPAR_MEMB)
        
        # Calculate utility and save attributes
        ostr_step = HPAR_COMM * np.exp(TPAR_COMM * np.exp(GPAR_COMM * fccp_prev))
        utcp_step = pycp_step
        utdf_step = pydf_step - ostr_step * (pydf_step - pycp_step) / pydf_step
        self.sanc_comm[step_simu] = sanc_step
        self.util_comm[step_simu, :] = np.append(utcp_step, utdf_step)
        self.tlut_comm[step_simu] = NMEM_COMM * (fccp_prev * utcp_step 
                                                 + (1 - fccp_prev) * utdf_step)
    
    def swap_step(self, step_simu):
        '''Swaps member statuses based on utility'''
        
        # Assign new statuses
        for memb in self.memb_comm.values():
            if step_simu < 1: 
                memb.stus_memb[step_simu] = self.stin_comm[memb.mbid_memb]
            else:
                memb.stus_memb[step_simu] = memb.stus_memb[step_simu - 1]
        
        # Identify member pairs
        mbid_comm = np.array(list(self.memb_comm.keys()))
        if UPRT_COMM > 0.5:
            mbid_swap = mbid_comm
        else:
            mbid_swap = np.random.choice(mbid_comm, 2 * int(UPRT_COMM * NMEM_COMM), 
                                         replace = False)
        np.random.shuffle(mbid_swap)
        mbid_swap = mbid_swap.reshape((int(len(mbid_swap) / 2), 2))
        
        # Conduct member swaps
        for swid_swap in mbid_swap:
            memb_swap = [self.memb_comm[mbid_comm] for mbid_comm in swid_swap]
            stus_swap = np.array([memb.stus_memb[step_simu] for memb in memb_swap])
            util_swap = np.array([self.util_comm[step_simu, 0] if stus_comm == 1 
                                  else self.util_comm[step_simu, 1] for stus_comm in stus_swap])
            if stus_swap[0] != stus_swap[1]:
                pswp_swap = (util_swap[0] - util_swap[1]) \
                       / (abs(util_swap[0]) + abs(util_swap[0]))
                if pswp_swap >= 0 and np.random.random() < pswp_swap:
                    memb_swap[1].stus_memb[step_simu] = stus_swap[0]
                elif pswp_swap < 0 and np.random.random() < abs(pswp_swap):
                    memb_swap[0].stus_memb[step_simu] = stus_swap[1]
        
    def updt_step(self, step_simu):
        '''Updates community fractions, effort'''
        
        ncop_step = len([memb for memb in self.memb_comm.values() 
                         if memb.stus_memb[step_simu] == 1])
        fccp_step = ncop_step / NMEM_COMM
        self.fccp_comm[step_simu] = fccp_step
        self.tlef_comm[step_simu] = NMEM_COMM * (fccp_step * EPAR_MEMB 
                                      + (1 - fccp_step) * EPAR_MEMB * MPAR_MEMB)
        

class Member(object):
    '''Parameters specific to each member
    
    Attributes: mbid_memb, cmid_memb, stus_comm
    '''
    def __init__(self):
        
        # Set initial values
        self.mbid_memb = None
        self.cmid_memb = None
        self.stus_memb = np.full([NSTP_SIMU], -99999, dtype = int)


class Results(object):
    '''The simulation results to be saved for analysis
    
    Attributes: ruas_rslt
    '''
    
    def __init__(self):
        
        # Define Attributes
        self.ruas_rslt = None
   
    def rslt_clct(self, rsce, ruas):
        '''Collects results from all objects'''
        
        # Collect RUAS data
        ruas_rslt = []
        for comm in ruas.comm_ruas.values():
            cmtl_rslt = np.array([comm.cmid_comm, comm.fccp_comm[-1], comm.stus_comm[-1]])
            ruas_rslt.append(cmtl_rslt)
        ruas_rslt = np.append(np.array(ruas_rslt), ruas.stin_ruas.reshape(NCOM_RUAS, 1), 1)
        ruas_rslt = pd.DataFrame(ruas_rslt, 
                                 columns = ['cmid', 'fccp', 'stfn', 'stin'], dtype = float)
        ruas_rslt['cmid'] = ruas_rslt['cmid'].astype(int)
        ruas_rslt['stfn'] = ruas_rslt['stfn'].astype(int)
        ruas_rslt['stin'] = ruas_rslt['stin'].astype(int)
        ruas_rslt = ruas_rslt[['cmid', 'fccp', 'stin', 'stfn']]
        
        # Save attributes
        self.ruas_rslt = ruas_rslt


def modl_simu(irun_simu):
    '''Run model and save results'''
    
    # Set new seed for each run
    np.random.seed(irun_simu)
    
    # Initialize objects
    rsce = Resource()
    ruas = ResourceUsersAssociation(rsce)
    rslt = Results()
    
    # Run model
    for step_simu in range(NSTP_SIMU):
        ruas.rsce_step(rsce, step_simu)
        ruas.util_step(step_simu)
        ruas.util_step(step_simu)
        ruas.swap_step(step_simu)
        ruas.updt_step(step_simu)
    
    # Collect Results
    rslt.rslt_clct(rsce, ruas)
        
    # Add new columns with run and simulation number
    rslt.ruas_rslt.insert(0, "irun", np.full(rslt.ruas_rslt.shape[0], irun_simu), False)
    rslt.ruas_rslt.insert(0, "simu", np.full(rslt.ruas_rslt.shape[0], NMBR_SIMU), False)
    
    # Upload tables to Sqlite database
    try:
        lock.acquire(True)
        rslt.ruas_rslt.to_sql('ruas_rslt', con = conn, if_exists = 'append', index = False)
    finally:
        lock.release()


## Import parameters and run model
json_file = sys.argv[1]

with open(json_file, "r") as para_file:
    para_data = json.load(para_file)
    
cats = ['RUAS', 'COMM', 'MEMB', 'RSCE', 'SIMU']
for catg in cats:
    for vble, valu in para_data[catg].items():
        try:
            exec(vble + ' = eval(valu)')
        except:
            exec(vble + ' = valu')
DBSE_FILE = para_data['DBSE_FILE']

# Create database and pool connections
conn = sqlite3.connect(DBSE_FILE)
lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

# Collect simulation data and upload to database
simu_data = pd.DataFrame({'simu': pd.Series(NMBR_SIMU, dtype = int),
                          'flmn': pd.Series(FLMN_RSCE, dtype = float),
                          'flsd': pd.Series(FLSD_RSCE, dtype = float),
                          'sanc': pd.Series(SANC_RUAS, dtype = float),
                          'mpar': pd.Series(MPAR_MEMB, dtype = float),
                          'frin': pd.Series(FCIN_RUAS, dtype = float),
                          'fcin': pd.Series(FCIN_COMM, dtype = float)})
simu_data.to_sql('simu_data', con = conn, if_exists = 'append', index = False)

# Print simulation number
print('Simulation:' + str(NMBR_SIMU))

# Run model
pool.map(modl_simu, range(0, NRUN_SIMU))

# Close connections
pool.close()
pool.join()
conn.close()