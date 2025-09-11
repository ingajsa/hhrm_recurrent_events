#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:53:44 2022

@author: insauer
"""
import sys
#sys.path.append('/home/insauer/projects/hhrm/hhrm_recurrent_events')
from hhwb.agents.government import Government
from hhwb.agents.hh_register import HHRegister

from hhwb.agents.shock import Shock
from hhwb.application.climate_life import ClimateLife
from hhwb.application.data_analysis import DataAnalysis

if __name__ == "__main__":
    print("Arguments received:", sys.argv)
    
    country=sys.argv[1]
    subsistence_line=float(sys.argv[2])
    output_data_path=sys.argv[3]
    run_time=int(sys.argv[4])
    work_path=sys.argv[5]
    hh_path=sys.argv[6]
    shock_path=sys.argv[7]
    survey_file=sys.argv[8]
    start_year=int(sys.argv[9])
    cores=int(sys.argv[10])

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
    
    hh_reg.set_from_csv(work_path=work_path, path=work_path+hh_path,  id_col='fhhid', weight_col='weight',
                          income_col='income', file_name=survey_file,
                          decile='decile', subsistence_line=subsistence_line)
    # print('Households registered')
    ## get number of registered households
    
    all_hhs = hh_reg.hh_list
    
    hh=all_hhs[0]
    
    
    """ set up of the government agent """
    
    gov = Government()
    gov.set_tax_rate(all_hhs)
    
    """ set up of the shock agent """
    
    fld = Shock()
    fld.read_vul_shock(path=work_path+hh_path, output_path=work_path+output_data_path,
                        file=survey_file, start_year=start_year)
    
    
    """ set up dynamic modeling """
    
    cl = ClimateLife(all_hhs, fld, gov)
    # cl.start(work_path=work_path, result_path='/data/output_'+args.run_name+'/',
    #           cores=cores, reco_period=args.run_time)
    
    """ call of the dynamic modeling """
    cl.start(work_path=work_path, result_path=work_path+output_data_path,
              cores=cores, reco_period=run_time)
    """ generate short data analysis"""
    
    
    
    da=DataAnalysis(work_path+hh_path, hh_file=survey_file, output_data_path=work_path+output_data_path, run_name='test')
    
    da.analyse_time(step=1000)
    da.analyse_wb(step=1000)

