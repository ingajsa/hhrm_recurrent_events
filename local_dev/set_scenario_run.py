#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:51:49 2025

@author: insauer
"""

import os
import pandas as pd
import subprocess
from pathlib import Path


country='PHL'


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

work_path=os.path.abspath(os.path.join(os.pardir, os.pardir))

forcing_folder = Path("/home/insauer/projects/STP/global_STP_paper/data/isimip3_forcing/final_PHL")

def get_time_period(df):
    
    # Convert column names to datetime
    dates = pd.to_datetime(df.columns, errors="coerce")
    
    # Drop invalid conversions (if some columns aren't dates)
    dates = dates.dropna()
    
    return dates.min().year
    

for filename in os.listdir(forcing_folder):
    if os.path.isfile(os.path.join(forcing_folder, filename)):
        print(filename)
        
        hh_path =os.path.join(forcing_folder, filename)
        
        shock_path = hh_path
        
        names = filename.split("_")
        
        output_data_path=os.path.join(work_path, f'projects/TipESM/data/scenarios/{filename[:-4]}/')
        print(output_data_path)
        print(hh_path)
        
        hh_data=pd.read_csv(hh_path)
        
        start_year=get_time_period(hh_data)
        

        if os.path.exists(output_data_path):
            print("File exists.")
            continue
        
        survey_file=f'{filename[:-4]}.csv'


        lambda_path='/hhrm_recurrent_events/data/global_test/lambdas_{}.csv'.format(country)
        
        
        
        cores=10


        cnt_params=pd.read_csv('/home/insauer/projects/STP/global_STP_paper/'+
                                'data/parameters/forcing_param_countries/'+
                                'parameters_{}.csv'.format(country))
        
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
