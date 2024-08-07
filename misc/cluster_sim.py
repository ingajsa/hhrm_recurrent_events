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
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse


import psutil

parser = argparse.ArgumentParser(
    description='run hhwb for different shock series')
parser.add_argument(
    '--run_name', type=str, default='shocks',
    help='string for run name')

parser.add_argument(
    '--type', type=str, default='shocks',
    help='run type')

parser.add_argument(
    '--run_time', type=int, default=160,
    help='run time in years')

parser.add_argument(
    '--seed', type=int, default=0,
    help='random state of household distribution')

parser.add_argument(
    '--eta', type=float, default=1.5,
    help='eta parameter in well-being')

parser.add_argument(
    '--subsistence_line', type=float, default=15926.,
    help='parameter indicating level of subsistence')

parser.add_argument(
    '--pi', type=float, default=0.33,
    help='productivity of capital stock')

parser.add_argument(
    '--T_RNG', type=int, default=15,
    help='time horizon of optimization')

parser.add_argument(
    '--R', type=int, default=3339,
    help='minimum recovery rate')

args = parser.parse_args()

"""save configuration of the run"""

params=pd.DataFrame(data={'PI':args.pi,
                 'ETA':args.eta,
                 'SUBS_SAV_RATE':args.R,
                 'T_RNG':args.T_RNG,
                 'SUBSISTENCE_LINE':args.subsistence_line}, index=[0])

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


CLUSTER=True

if CLUSTER==True:
    work_path='/p/projects/ebm/inga/hhwb'
    cores=psutil.cpu_count(logical = True)
else:
    work_path='/home/insauer/projects/WB_model/hhwb'
    cores=7
    
print('Number threads = ' + str(cores))
hh_reg = HHRegister()


if args.run_name.find('syn') != -1:
    print('syn-path')
    hh_path = '/data/survey_data/PHL/survey_seed/region_hh_full_pack_PHL_pop_syn_{}.csv'.format(str(args.seed))
    shock_path = '/data/shock_data/shocks_syn_seed/'+args.run_name+'.csv'
    output_data_path=''
    
elif args.type == 'single':
    hh_path = '/data/survey_data/PHL/region_hh_full_pack_PHL_pop.csv'
    shock_path = '/data/shock_data/shocks_seed/single_shock_seed_{}/{}_{}.csv'.format(str(args.seed), args.run_name, str(args.seed))
    
else:
    hh_path = '/data/survey_data/PHL/region_hh_full_pack_PHL_pop.csv'
    
    
    
    shock_path = '/data/shock_data/shocks_seed/shocks_{}.csv'.format(str(args.seed))
    output_data_path=''


""" generates the household agents from a csv, the parameter correspond to the relevant column names"""

hh_reg.set_from_csv(work_path=work_path, path=hh_path, id_col='fhhid', n_ind = 'n_individuals', weight_col='weight',
                      vul_col='vul', income_col='income', income_sp='income_sp', region='region',
                      decile='decile', savings='savings', subsistence_line=args.subsistence_line,
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
                event_identifier='-', run=args.run_name)

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
          cores=cores, reco_period=args.run_time)

""" generate short data analysis"""

survey_data_path=work_path+ hh_path
shock_data_path=work_path+shock_path


da=DataAnalysis(survey_data_path, shock_data_path, output_data_path='', column_id='', run_name=args.run_name)

da.analyse_time(step=10000)
da.analyse_wb(step=20000)

da.analyse_time_steps(step=10000)
