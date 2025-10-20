#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 20:30:34 2025

@author: insauer
"""

import os
import pandas as pd
import subprocess



country='CHN'

"""Parameters to adjust"""

work_path=os.path.abspath(os.path.join(os.pardir, os.pardir))

output_data_path=os.path.join(work_path, 'example_output/')

hh_path ='/hhrm_recurrent_events/data/china_test/forcing_{}_120as.zip'.format(country)
    
shock_path = '/hhrm_recurrent_events/data/china_test/forcing_{}_120as.zip'.format(country)

survey_file='forcing_{}_120as.csv'.format(country)

lambda_path='/hhrm_recurrent_events/data/china_test/lambdas_{}.csv'.format(country)

start_year=2000

cores=10

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

# r minimum recovery_rate
#k_pub = 0.25


cnt_params=pd.read_csv(work_path+'/hhrm_recurrent_events/data/china_test/parameters_{}.csv'.format(country))

PI=cnt_params['PI'].values[0]

k_pub=cnt_params['k_pub'].values[0]

R=cnt_params['R'].values[0]


"""save configuration of the run"""


params=pd.DataFrame(data={'PI':PI,
                  'ETA':ETA,
                  'SUBS_SAV_RATE':R,
                  'T_RNG':T_RNG,
                  'K_PUB':k_pub,
                  'COUNTRY': country,
                  'OUTPUT_DATA_PATH': output_data_path,
                  'LAMBDA_PATH': work_path+lambda_path,
                  'LAMBDA_PRECISION': lambda_precision,
                  'SUBSISTENCE_LINE':subsistence_line}, index=[0])

if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

params.to_csv(output_data_path+'params.csv')
params.to_csv('params.csv')

# Run script with arguments
subprocess.run(["python", "local_run.py",
                country,
                str(subsistence_line),
                output_data_path,
                str(run_time),
                work_path,
                hh_path,
                shock_path,
                survey_file,
                str(start_year),
                str(cores)])
