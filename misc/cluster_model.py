#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:23:13 2025

@author: insauer
"""

import sys
sys.path.append('/p/projects/ebm/inga/hhrm/hhrm_recurrent_events')

import psutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='run hhwb for different shock series')



parser.add_argument(
    '--country', type=str, default='shocks',
    help='string for run name')

parser.add_argument(
    '--file_name', type=str, default='shocks',
    help='run type')

parser.add_argument(
    '--out_put_data_path', type=str, default='',
    help='run time in years')

parser.add_argument(
    '--work_path', type=str, default='',
    help='random state of household distribution')

parser.add_argument(
    '--hh_path', type=str, default='',
    help='eta parameter in well-being')

parser.add_argument(
    '--shock_path', type=str, default='',
    help='parameter indicating level of subsistence')

parser.add_argument(
    '--survey_file', type=float, default='',
    help='productivity of capital stock')

parser.add_argument(
    '--start_year', type=int, default=200,
    help='time horizon of optimization')


args = parser.parse_args()

"""save configuration of the run"""

"""Parameters to adjust"""

lambda_precision=4

# run_time (in years)
run_time=30

#eta parameter in well-being
ETA=1.5

#level of subsistence line
subsistence_line=2.15*365

# time horizon of optimization
T_RNG= 15

lambda_path='/hhrm_recurrent_events/data/global_test/lambdas_{}.csv'.format(args.country)



cnt_params=pd.read_csv('/p/projects/ebm/inga/hhrm/hhrm_recurrent_events/data/global_test/parameters_{}.csv'.format(args.country))

PI=cnt_params['PI'].values[0]

k_pub=cnt_params['k_pub'].values[0]

R=cnt_params['R'].values[0]


"""save configuration of the run"""
                



params=pd.DataFrame(data={'PI':PI,
                  'ETA':ETA,
                  'SUBS_SAV_RATE':R,
                  'T_RNG':T_RNG,
                  'K_PUB':k_pub,
                  'COUNTRY': args.country,
                  'OUTPUT_DATA_PATH': args.output_data_path,
                  'LAMBDA_PATH': args.work_path+args.lambda_path,
                  'LAMBDA_PRECISION': lambda_precision,
                  'SUBSISTENCE_LINE':subsistence_line}, index=[0])


params.to_csv('params.csv')

from hhwb.agents.government import Government
from hhwb.agents.hh_register import HHRegister

from hhwb.agents.shock import Shock
from hhwb.application.climate_life import ClimateLife


if __name__ == "__main__":
    
    cores=psutil.cpu_count(logical = True)
    


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
    
    
        
    print('Number threads = ' + str(cores))
    
    
    
    hh_reg = HHRegister()
    
    
    """ generates the household agents from a csv, the parameter correspond to the relevant column names"""
    
    hh_reg.set_from_csv(work_path=args.work_path, path=args.hh_path,  id_col='fhhid', weight_col='weight',
                          income_col='income', file_name=args.survey_file,
                          decile='decile', subsistence_line=subsistence_line)
    # print('Households registered')
    ## get number of registered households
    
    all_hhs = hh_reg.hh_list
    
    hh=all_hhs[0]
    
    
    """ set up of the government agent """
    
    gov = Government()
    gov.set_tax_rate(all_hhs)
    
    print(gov.K_pub)
    
    
    """ set up of the shock agent """
    
    fld = Shock()
    fld.read_vul_shock(path=args.hh_path, output_path=args.output_data_path,
                        file=args.survey_file, start_year=args.start_year)
    
    
    """ set up dynamic modeling """
    
    cl = ClimateLife(all_hhs, fld, gov)
    # cl.start(work_path=work_path, result_path='/data/output_'+args.run_name+'/',
    #           cores=cores, reco_period=args.run_time)
    
    """ call of the dynamic modeling """
    cl.start(work_path=args.work_path, result_path=args.output_data_path,
              cores=cores, reco_period=run_time)
    """ generate short data analysis"""
    
    
    
    # da=DataAnalysis(hh_path, hh_file=survey_file, output_data_path=output_data_path, run_name='test')
    
    # da.analyse_time(step=1000)
    # da.analyse_wb(step=1000)

