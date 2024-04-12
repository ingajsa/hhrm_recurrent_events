#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:53:44 2022

@author: insauer
"""
import sys
import os
import inspect
import pandas as pd


run_type='test'

# run_time (in 4-week steps)
run_time=10

#eta parameter in well-being
eta=1.5

#level of subsistence line
subsistence_line=15926

# time horizon of optimization
t_rng= 15

# productivity of capital stock
pi=0.33

# r minimum recovery_rate
R = 3339

# r minimum recovery_rate
k_pub = 0.25


"""save configuration of the run"""

dirc=os.getcwd()

params=pd.DataFrame(data={'PI':pi,
                 'ETA':eta,
                 'SUBS_SAV_RATE':R,
                 'T_RNG':t_rng,
                 'K_PUB':k_pub,
                 'SUBSISTENCE_LINE':subsistence_line}, index=[0])

params.to_csv('params.csv')

from hhwb.agents.government import Government
from hhwb.agents.hh_register import HHRegister

from hhwb.agents.shock import Shock
from hhwb.application.climate_life import ClimateLife
from hhwb.application.data_analysis import DataAnalysis


""" Shock definition. This script coordinates a run of the household resilience model by setting 
    data pathes according to the configuration of the run. The routine basically encompasses the
    following steps:
       - creating household agents
       - create government agent 
       - create shock agent
       - set-up of the dynamic model
       - running the dynamic model
       - short analysis of the data
"""



work_path=os.path.abspath(os.path.join(os.pardir))
cores=4
    
print('Number threads = ' + str(cores))
hh_reg = HHRegister()


hh_path ='/data/test_hh/test_hh.zip'
    
    
    
shock_path = '/data/test_shocks/test_shocks.zip'

output_data_path=''


""" generates the household agents from a csv, the parameter correspond to the relevant column names"""

hh_reg.set_from_csv(work_path=work_path, path=hh_path, id_col='fhhid', n_ind = 'n_individuals', weight_col='weight',
                      vul_col='vul', income_col='income', income_sp='income_sp', region='region',
                      decile='decile', savings='savings', subsistence_line=subsistence_line,
                      ispoor='ispoor', isurban='isurban')

# # print('Households registered')
# # ## get number of registered households

all_hhs = hh_reg.hh_list

""" set up of the government agent """

gov = Government()
gov.set_tax_rate(all_hhs)

""" set up of the shock agent """

fld = Shock()
fld.read_shock(work_path=work_path, path=shock_path,
                event_identifier='-', run=run_type)

# fld.generate_single_shocks(work_path=work_path,
#                         path_haz='/data/output/shocks/shocks_99.csv',
#                         path_hh='/data/survey_data/PHL/region_hh_full_pack_PHL_pop.csv',
#                         path_hh_orig='/data/survey_data/PHL/survey_PHL.csv',
#                         hh_reg=None, k_eff=0, seed=args.seed)

# print('Shocks prepared')
# print(fld.aff_ids)

""" set up dynamic modeling """

cl = ClimateLife(all_hhs, fld, gov)
# cl.start(work_path=work_path, result_path='/data/output_'+args.run_name+'/',
#           cores=cores, reco_period=args.run_time)

""" call of the dynamic modeling """
cl.start(work_path='', result_path='',
          cores=cores, reco_period=run_time)

""" generate short data analysis"""



da=DataAnalysis(hh_path, shock_path, output_data_path='', column_id='', run_name=run_type)

da.analyse_time(step=10000)
da.analyse_wb(step=20000)

#da.analyse_time_steps(step=10000)
