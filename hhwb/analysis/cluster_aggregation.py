#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:29:18 2023

@author: insauer
"""

import sys
import os
import inspect
from hhwb.agents.shock import Shock
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse
import psutil

parser = argparse.ArgumentParser(
    description='run hhwb for different shock series')

parser.add_argument(
    '--seed', type=int, default=0,
    help='runoff model')

args = parser.parse_args()

CLUSTER=True

if CLUSTER==True:
    ebm='/p/projects/ebm/inga/hhwb'
    primap='/p/projects/primap/inga/hhwb'
    cores=psutil.cpu_count(logical = True)
else:
    work_path='/home/insauer/projects/WB_model/hhwb'
    cores=7
    
run_names= ['shocks_7',
    'shocks_19',
    'shocks_20',
    'shocks_26',
    'shocks_28',
    'shocks_32',
    'shocks_34',
    'shocks_38',
    'shocks_47',
    'shocks_48',
    'shocks_53',
    'shocks_59',
    'shocks_64',
    'shocks_66',
    'shocks_76',
    'shocks_80',
    'shocks_83',
    'shocks_86',
    'shocks_91',
    'shocks_92',
    'shocks_99',
    'shocks_101',
    'shocks_110',
    'shocks_117',
    'shocks_118',
    'shocks_123',
    'shocks_127',
    'shocks_130',
    'shocks_132',
    'shocks_133',
    'shocks_142',
    'shocks_144',
    'shocks_162',
    'shocks_166',
    'shocks_169',
    'shocks_180',
    'shocks_182',
    'shocks_201']


result_file=ebm +'/factual/run_shocks_{}/survey_shocks_{}_analysed.csv'.format(str(args.seed), str(args.seed))


result_file_c1=ebm +'/counterfactual_1/run_shocks_syn_{}/survey_shocks_syn_{}_analysed.csv'.format(str(args.seed), str(args.seed))


result_path_c2= primap+'/counterfactual_2/'

    
shocks=pd.read_csv(ebm + '/factual/run_shocks_{}/shocks_aggregated_shocks_{}.csv'.format(str(args.seed), str(args.seed)))
del_cols = [col for col in shocks.columns if 'Unnamed' in col]
shocks=shocks.drop(columns=del_cols)
shocks['fhhid']=shocks.index
shocks['n_events']=shocks.iloc[:,:-1].sum(axis=1)


shocks_c1=pd.read_csv(ebm +'/counterfactual_1/run_shocks_syn_{}/shocks_aggregated_shocks_syn_{}.csv'.format(str(args.seed), str(args.seed)))
del_cols = [col for col in shocks_c1.columns if 'Unnamed' in col]
shocks_c1=shocks_c1.drop(columns=del_cols)
shocks_c1['fhhid']=shocks_c1.index
shocks_c1['n_events']=shocks_c1.iloc[:,:-1].sum(axis=1)

"""" replace number events by the aggregated number of events"""

survey=pd.read_csv(result_file)
survey['n_events']=shocks['n_events']
survey['pc_income']=survey['income']/survey['n_individuals']
del_cols = [col for col in survey.columns if 'Unnamed' in col]
survey=survey.drop(columns=del_cols)
shocks['region']=survey['region']


survey_c1=pd.read_csv(result_file_c1)
survey_c1['n_events']=shocks_c1['n_events']
survey_c1['hh_weight']=survey_c1['weight']/survey_c1['n_individuals']
survey_c1['pc_income']=survey_c1['income']/survey_c1['n_individuals']
del_cols = [col for col in survey_c1.columns if 'Unnamed' in col]
survey_c1=survey_c1.drop(columns=del_cols)
shocks_c1['region']=survey_c1['region']

ts=[7,  19,  20,  26,  28,  32,  34,  38,  47,  48,  53,  59,  64,
             66,  76,  80,  83,  86,  91,  92,  99, 101, 110, 117, 118, 123,
            127, 130, 132, 133, 142, 144, 162, 166, 169, 180, 182, 201]

"""sum over all single event runs in counterfactual 2, in order to derive the aggregated impacts per
   household in k_eff, cons, cons_sm, wb, and wb_sm
"""


survey['wb_loss_c2']=0
survey['wb_loss_sm_c2']=0
survey['d_keff_c2']=0
survey['cons_c2']=0
survey['cons_sm_c2']=0


for i,run in enumerate(run_names):
    print(run)
    
    single_frame=pd.read_csv(result_path_c2 +'/run_shocks_{}_{}/survey_shocks_{}_analysed.csv'.format(str(ts[i]), args.seed, str(ts[i])))

    add=np.array(single_frame['keff_{}'.format(str(ts[i]))])
    survey['d_keff_c2']=survey['d_keff_c2']+add
    
    add=np.array(single_frame['cons_tot'])
    survey['cons_c2']=survey['cons_c2']+add
    
    add=np.array(single_frame['cons_sm_tot'])
    survey['cons_sm_c2']=survey['cons_sm_c2']+add
    
    add=np.array(single_frame['wb_loss'])
    survey['wb_loss_c2']=survey['wb_loss_c2']+add

    add=np.array(single_frame['wb_loss_sm'])
    survey['wb_loss_sm_c2']=survey['wb_loss_sm_c2']+add
    
    survey.to_csv('survey_processed_{}.csv'.format(str(args.seed)))
    

"""calculate k_eff_0"""

survey=pd.read_csv('survey_processed_{}.csv'.format(str(args.seed)))

PI=0.33
ETA=1.5
tax_rate= (survey['hh_weight']*survey['income_sp']).sum()/(survey['hh_weight']*survey['income']).sum()
survey['k_eff_0']= (survey['income']-survey['income_sp'])/((1-tax_rate)*PI)
tax_rate= (survey_c1['hh_weight']*survey_c1['income_sp']).sum()/(survey_c1['hh_weight']*survey_c1['income']).sum()
survey_c1['k_eff_0']= (survey_c1['income']-survey_c1['income_sp'])/((1-tax_rate)*PI)


"""monitarization of well-being loss"""
mean_income = ((survey['income']*survey['weight']).sum())/survey['weight'].sum()
c_avg=mean_income
wb_cross=c_avg**(-ETA)

survey['wb_c_equ']=survey['wb_loss']/wb_cross
survey['wb_c_equ_sm']=survey['wb_loss_sm']/wb_cross
survey_c1['wb_c_equ']=survey_c1['wb_loss']/wb_cross
survey_c1['wb_c_equ_sm']=survey_c1['wb_loss_sm']/wb_cross
survey['wb_c_equ_c2']=survey['wb_loss_c2']/wb_cross
survey['wb_c_equ_sm_c2']=survey['wb_loss_sm_c2']/wb_cross

"""calculate d_k_eff_tot"""

keff_cols = [col for col in survey.columns if 'keff_diff' in col]
keff_df=survey[keff_cols].clip(lower=0)
survey['d_keff_tot']=keff_df.sum(axis=1)

keff_cols = [col for col in survey_c1.columns if 'keff_diff' in col]
keff_df=survey_c1[keff_cols].clip(lower=0)
survey_c1['d_keff_tot']=keff_df.sum(axis=1)

""" calculate resilience """

survey['soc_res_cons']=survey['d_keff_tot']/survey['cons_tot']
survey['soc_res_cons_sm']=survey['d_keff_tot']/survey['cons_sm_tot']

survey_c1['soc_res_cons']=survey_c1['d_keff_tot']/survey_c1['cons_tot']
survey_c1['soc_res_cons_sm']=survey_c1['d_keff_tot']/survey_c1['cons_sm_tot']

survey['soc_res_cons_c2']=survey['d_keff_tot']/survey['cons_c2']
survey['soc_res_cons_sm_c2']=survey['d_keff_tot']/survey['cons_sm_c2']

survey['soc_res_wb_mon']= survey['d_keff_tot']/survey['wb_c_equ']
survey['soc_res_wb_mon_sm']= survey['d_keff_tot']/survey['wb_c_equ_sm']
survey['soc_res_wb_mon'] = survey['soc_res_wb_mon'].replace([-np.inf, np.inf],[np.nan, np.nan])
survey['soc_res_wb_mon_sm'] = survey['soc_res_wb_mon_sm'].replace([-np.inf, np.inf],[np.nan, np.nan])

survey['soc_res_c2_wb_mon_sm']= survey['d_keff_c2']/survey['wb_c_equ_sm_c2']
survey['soc_res_c2_wb_mon']= survey['d_keff_c2']/survey['wb_c_equ_c2']
survey['soc_res_c2_wb_mon'] = survey['soc_res_c2_wb_mon'].replace([-np.inf, np.inf],[np.nan, np.nan])
survey['soc_res_c2_wb_mon_sm'] = survey['soc_res_c2_wb_mon_sm'].replace([-np.inf, np.inf],[np.nan, np.nan])

survey_c1['soc_res_wb_mon']= survey_c1['d_keff_tot']/survey_c1['wb_c_equ']
survey_c1['soc_res_wb_mon_sm']= survey_c1['d_keff_tot']/survey_c1['wb_c_equ_sm']
survey_c1['soc_res_wb_mon'] = survey_c1['soc_res_wb_mon'].replace([-np.inf,np.inf],[np.nan, np.nan])
survey_c1['soc_res_wb_mon_sm'] = survey_c1['soc_res_wb_mon'].replace([-np.inf,np.inf],[np.nan, np.nan])

""" determine time under subsistence line """

time_cols = [col for col in survey.columns if 'time' in col]
survey[time_cols]=(survey[time_cols]/910)*100
survey[time_cols] = survey[time_cols].clip(upper=100)

time_cols = [col for col in survey_c1.columns if 'time' in col]
survey_c1[time_cols]=(survey_c1[time_cols]/910)*100
survey_c1[time_cols] = survey_c1[time_cols].clip(upper=100)

survey.to_csv('survey_processed_{}.csv'.format(str(args.seed)))
survey_c1.to_csv('survey_processed_c1_{}.csv'.format(str(args.seed)))


# calculate region data

regions=list(set(survey['region']))

phl_df=pd.DataFrame()
for reg in regions:
    
    region_df=pd.DataFrame()
    df = survey.loc[(survey['region']==reg), ['hh_weight', 'pc_income', 'k_eff_0','wb_loss', 'd_keff_tot', 'wb_c_equ_sm',
                                              'cons_tot','reco_time_sm', 'time_under_sub_sm', 'n_events', 'soc_res_wb_mon_sm','income']]

    df_ev = survey.loc[(survey['region']==reg) & (survey['n_events']>0), ['hh_weight', 'pc_income', 'k_eff_0','wb_loss', 'd_keff_tot', 'wb_c_equ_sm',
                                              'cons_tot','reco_time_sm', 'time_under_sub_sm', 'n_events', 'soc_res_wb_mon_sm', 'income']]

    df_ovsub =survey.loc[(survey['region']==reg)& (survey['income']>survey['subsistence_line']), ['hh_weight', 'pc_income', 'wb_loss', 'd_keff_tot',
                                              'reco_time_sm', 'time_under_sub_sm', 'n_events']]
    region_df.loc[0,'region']=reg
    region_df['pc_income']=(df['pc_income']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['n_events']=shocks.loc[shocks['region']==reg].iloc[:,:-3].sum().clip(upper=1).sum()
    region_df['reco_time']=(df['reco_time_sm']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['reco_time_ev']=(df_ev['reco_time_sm']*df_ev['hh_weight']).sum()/df_ev['hh_weight'].sum()
    region_df['time_under_sub_rec']=(df_ovsub['time_under_sub_sm']*df_ovsub['hh_weight']).sum()/df_ovsub['hh_weight'].sum()
    region_df['time_under_sub_tot']=(df['time_under_sub_sm']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['d_keff_abs']=(df['d_keff_tot']*df['hh_weight']).sum()
    region_df['wb_loss_pc']=(df['wb_loss']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['wb_loss_abs']=(df['wb_loss']*df['hh_weight']).sum()
    region_df['soc_res']=(df_ev['d_keff_tot']*df_ev['hh_weight']).sum()/(df_ev['wb_c_equ_sm']*df_ev['hh_weight']).sum()
    region_df['soc_res_cons']=(df_ev['d_keff_tot']*df_ev['hh_weight']).sum()/(df_ev['cons_tot']*df_ev['hh_weight']).sum()
    region_df['d_cons_pc']=(df['cons_tot']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['wb_loss_pc_log']=np.log10(region_df['wb_loss_pc'])
    region_df['wb_loss_abs_log']=np.log10(region_df['wb_loss_abs'])
    region_df['d_keff_abs_log']=np.log10(region_df['d_keff_abs'])
    phl_df=phl_df.append(region_df, ignore_index=True)
    
phl_df.to_csv('region_results_{}.csv'.format(str(args.seed)))

phl_df_c1=pd.DataFrame()
for reg in regions:
    
    region_df=pd.DataFrame()
    df = survey_c1.loc[(survey_c1['region']==reg), ['hh_weight', 'pc_income', 'k_eff_0','wb_loss', 'd_keff_tot', 'wb_c_equ_sm',
                                              'cons_tot','reco_time_sm', 'time_under_sub_sm', 'n_events', 'soc_res_wb_mon_sm', 'income']]
    df_ev = survey_c1.loc[(survey_c1['region']==reg) & (survey_c1['n_events']>0), ['hh_weight', 'pc_income', 'k_eff_0','wb_loss', 'd_keff_tot', 'wb_c_equ_sm',
                                              'cons_tot','reco_time_sm', 'time_under_sub_sm', 'n_events', 'soc_res_wb_mon_sm', 'income']]


    df_ovsub =survey_c1.loc[(survey_c1['region']==reg)& (survey_c1['income']>survey_c1['subsistence_line']), ['hh_weight', 'pc_income', 'wb_loss_sm', 'd_keff_tot',
                                              'reco_time_sm', 'time_under_sub_sm', 'n_events']]
    region_df.loc[0,'region']=reg
    region_df['pc_income']=(df['pc_income']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['n_events']=shocks.loc[shocks['region']==reg].iloc[:,:-3].sum().clip(upper=1).sum()
    region_df['reco_time']=(df['reco_time_sm']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['reco_time_ev']=(df_ev['reco_time_sm']*df_ev['hh_weight']).sum()/df_ev['hh_weight'].sum()
    region_df['time_under_sub_rec']=(df_ovsub['time_under_sub_sm']*df_ovsub['hh_weight']).sum()/df_ovsub['hh_weight'].sum()
    region_df['time_under_sub_tot']=(df['time_under_sub_sm']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['d_keff_abs']=(df['d_keff_tot']*df['hh_weight']).sum()
    region_df['wb_loss_pc']=(df['wb_loss']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['wb_loss_abs']=(df['wb_loss']*df['hh_weight']).sum()
    region_df['soc_res']=(df_ev['d_keff_tot']*df_ev['hh_weight']).sum()/(df_ev['wb_c_equ_sm']*df_ev['hh_weight']).sum()
    region_df['soc_res_cons']=(df_ev['d_keff_tot']*df_ev['hh_weight']).sum()/(df_ev['cons_tot']*df_ev['hh_weight']).sum()
    region_df['d_cons_pc']=(df['cons_tot']*df['hh_weight']).sum()/df['hh_weight'].sum()
    region_df['wb_loss_pc_log']=np.log10(region_df['wb_loss_pc'])
    region_df['wb_loss_abs_log']=np.log10(region_df['wb_loss_abs'])
    region_df['d_keff_abs_log']=np.log10(region_df['d_keff_abs'])
    phl_df_c1=phl_df_c1.append(region_df, ignore_index=True)

phl_df_c1.to_csv('region_results_c1_{}.csv'.format(str(args.seed)))

inc_df=pd.DataFrame()
variables=['reco_time', 'reco_time_ev','time_under_sub_rec', 'time_under_sub_tot', 'd_keff_abs',
           'wb_loss_pc', 'wb_loss_abs', 'soc_res', 'soc_res_cons', 'd_cons_pc', 'wb_loss_pc_log',
           'wb_loss_abs_log', 'd_keff_abs_log']
for v in variables:
    inc_df[v]=(phl_df[v]-phl_df_c1[v])*100/phl_df_c1[v]
inc_df['region']=phl_df_c1['region']

inc_df.to_csv('region_results_inc_{}.csv'.format(str(args.seed)))

deciles=list(set(survey['pcinc_decile']))

dec_data_all=pd.DataFrame()

for i, dec in enumerate(deciles):
    dec_data=pd.DataFrame()
    dec_data.loc[0,'decile']=dec
    dec_data['time_reco_sm']= ((survey.loc[survey['pcinc_decile']==dec,'reco_time_sm']*\
    survey.loc[survey['pcinc_decile']==dec,'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()

    dec_data['time_reco_sm_ev']= ((survey.loc[(survey['pcinc_decile']==dec) & (survey['n_events']>0),'reco_time_sm']*\
    survey.loc[(survey['pcinc_decile']==dec) & (survey['n_events']>0),'hh_weight']).sum())/survey.loc[(survey['pcinc_decile']==dec) & (survey['n_events']>0),'hh_weight'].sum()

    dec_data['time_reco_syn_sm']=((survey_c1.loc[survey_c1['pcinc_decile']==dec,'reco_time_sm']*\
    survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()

    dec_data['time_reco_syn_sm_ev']= ((survey_c1.loc[(survey_c1['pcinc_decile']==dec) & (survey_c1['n_events']>0),'reco_time_sm']*\
    survey_c1.loc[(survey_c1['pcinc_decile']==dec) & (survey_c1['n_events']>0),'hh_weight']).sum())/survey_c1.loc[(survey_c1['pcinc_decile']==dec) & (survey_c1['n_events']>0),'hh_weight'].sum()

    
    dec_data['time_sub_sm']=((survey.loc[(survey['pcinc_decile']==dec),'time_under_sub_sm']*\
    survey.loc[survey['pcinc_decile']==dec,'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['time_sub_syn_sm']=((survey_c1.loc[(survey_c1['pcinc_decile']==dec),'time_under_sub_sm']*\
    survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['time_subreco_sm'] = ((survey.loc[(survey['pcinc_decile']==dec)& (survey['income']>survey['subsistence_line']),
                                    'time_under_sub_sm']*survey.loc[(survey['pcinc_decile']==dec)& (survey['income']>survey['subsistence_line']),
                                    'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['time_subreco_syn_sm'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec)&
                                            (survey_c1['income']>survey_c1['subsistence_line']),
                                            'time_under_sub_sm']*survey_c1.loc[(survey_c1['pcinc_decile']==dec)&
                                            (survey_c1['income']>survey_c1['subsistence_line']),
                                            'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()

    dec_data['d_keff_ev'] = ((survey.loc[(survey['pcinc_decile']==dec) &
                                       (survey['n_events']>0) , 'd_keff_tot']*survey.loc[(survey['pcinc_decile']==dec) &
                                       (survey['n_events']>0) , 'hh_weight']).sum())/survey.loc[(survey['pcinc_decile']==dec) & (survey['n_events']>0) ,'hh_weight'].sum()
    
    dec_data['d_keff_ev_syn'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec) &
                                       (survey_c1['n_events']>0) , 'd_keff_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec) &
                                       (survey_c1['n_events']>0) , 'hh_weight']).sum())/survey_c1.loc[(survey_c1['pcinc_decile']==dec) & (survey_c1['n_events']>0),'hh_weight'].sum()
    
    dec_data['d_keff'] = ((survey.loc[(survey['pcinc_decile']==dec), 'd_keff_tot']*survey.loc[(survey['pcinc_decile']==dec),
                                                                                            'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['d_keff_syn'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                            'd_keff_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                                                         'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()
    dec_data['d_cons'] = ((survey.loc[(survey['pcinc_decile']==dec), 'cons_tot']*survey.loc[(survey['pcinc_decile']==dec),
                                                                                            'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    
    dec_data['d_cons_syn'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                            'cons_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                                                         'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()
    
    

    
    dec_data['wb_means_sm'] = ((survey.loc[(survey['pcinc_decile']==dec),
                                         'wb_loss_sm']*survey.loc[(survey['pcinc_decile']==dec),
                                         'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['wb_means_syn'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                              'wb_loss']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                              'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['wb_means_syn_sm'] = ((survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                            'wb_loss_sm']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),
                                            'hh_weight']).sum())/survey_c1.loc[survey_c1['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['wb_means'] = ((survey.loc[(survey['pcinc_decile']==dec),
                                      'wb_loss']*survey.loc[(survey['pcinc_decile']==dec),
                                      'hh_weight']).sum())/survey.loc[survey['pcinc_decile']==dec,'hh_weight'].sum()
    
    dec_data['soc_res'] = (survey.loc[(survey['pcinc_decile']==dec),'d_keff_tot']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()/(survey.loc[(survey['pcinc_decile']==dec),'wb_c_equ_sm']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()
    
    dec_data['soc_res_syn'] = (survey_c1.loc[(survey_c1['pcinc_decile']==dec),'d_keff_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),'hh_weight']).sum()/(survey_c1.loc[(survey_c1['pcinc_decile']==dec),'wb_c_equ_sm']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),'hh_weight']).sum()
    
    dec_data['soc_res_cons'] = (survey.loc[(survey['pcinc_decile']==dec),'d_keff_tot']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()/(survey.loc[(survey['pcinc_decile']==dec),'cons_tot']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()
    
    dec_data['soc_res_syn_cons_sm'] = (survey_c1.loc[(survey_c1['pcinc_decile']==dec),'d_keff_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),'hh_weight']).sum()/(survey_c1.loc[(survey_c1['pcinc_decile']==dec),'cons_tot']*survey_c1.loc[(survey_c1['pcinc_decile']==dec),'hh_weight']).sum()
    
    #dec_data['soc_res_cons'] = (survey.loc[(survey['pcinc_decile']==dec),'d_keff_tot']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()/(survey.loc[(survey['pcinc_decile']==dec),'cons_tot']*survey.loc[(survey['pcinc_decile']==dec),'hh_weight']).sum()
    
    #dec_data['soc_res_syn_cons'] = (survey_syn.loc[(survey_syn['pcinc_decile']==dec),'d_keff_tot']*survey_syn.loc[(survey_syn['pcinc_decile']==dec),'hh_weight']).sum()/(survey_syn.loc[(survey_syn['pcinc_decile']==dec),'cons_tot']*survey_syn.loc[(survey_syn['pcinc_decile']==dec),'hh_weight']).sum()
    dec_data_all = dec_data_all.append(dec_data, ignore_index=True)

dec_data_all.to_csv('dec_data_{}.csv'.format(str(args.seed)))

n_events=list(set(survey['n_events']))

keff_dec_events=np.zeros((len(deciles), len(n_events)))
keff_dec_events_rel=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'd_keff_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        mult2=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'd_keff_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_mult2_pc=mult2/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_dec_events[i,j]=(keff_mult1_pc-keff_mult2_pc)
        keff_dec_events_rel[i,j]=((keff_mult1_pc/keff_mult2_pc)*100-100)

keff_dec_events_df=pd.DataFrame(data=keff_dec_events)
keff_dec_events_df.to_csv('abs_diff_keff_{}.csv'.format(str(args.seed)))
keff_dec_events_rel_df=pd.DataFrame(data=keff_dec_events_rel)
keff_dec_events_rel_df.to_csv('rel_diff_keff_{}.csv'.format(str(args.seed)))

keff_dec_events_s=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'd_keff_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_dec_events_s[i,j]=keff_mult1_pc
keff_dec_events_df_s=pd.DataFrame(data=keff_dec_events_s)
keff_dec_events_df_s.to_csv('keff_c2_{}.csv'.format(str(args.seed)))



keff_dec_events_m=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'd_keff_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        keff_dec_events_m[i,j]=keff_mult1_pc
keff_dec_events_df_m=pd.DataFrame(data=keff_dec_events_m)
keff_dec_events_df_m.to_csv('keff_factual_{}.csv'.format(str(args.seed)))

cons_dec_events=np.zeros((len(deciles), len(n_events)))
cons_dec_events_rel=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        mult2=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_mult2_pc=mult2/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_dec_events[i,j]=cons_mult1_pc-cons_mult2_pc
        cons_dec_events_rel[i,j]=((cons_mult1_pc/cons_mult2_pc)*100-100)
        

cons_dec_events_m=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        cons_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_dec_events_m[i,j]=cons_mult1_pc
        

cons_dec_events_s=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        cons_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_dec_events_s[i,j]=cons_mult1_pc


cons_dec_events_df_s=pd.DataFrame(data=cons_dec_events_s)
cons_dec_events_df_s.to_csv('cons_c2_{}.csv'.format(str(args.seed)))
cons_dec_events_df_m=pd.DataFrame(data=cons_dec_events_m)
cons_dec_events_df_m.to_csv('cons_factual_{}.csv'.format(str(args.seed)))
cons_dec_events_df=pd.DataFrame(data=cons_dec_events)
cons_dec_events_df.to_csv('abs_diff_cons_{}.csv'.format(str(args.seed)))
cons_dec_events_rel_df=pd.DataFrame(data=cons_dec_events_rel)
cons_dec_events_rel_df.to_csv('rel_diff_cons_{}.csv'.format(str(args.seed)))



cons_sm_dec_events=np.zeros((len(deciles), len(n_events)))
cons_sm_dec_events_rel=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_sm_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        mult2=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_sm_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_sm_mult2_pc=mult2/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_sm_dec_events[i,j]=cons_sm_mult1_pc-cons_sm_mult2_pc
        cons_sm_dec_events_rel[i,j]=((cons_sm_mult1_pc/cons_sm_mult2_pc)*100-100)
        

cons_sm_dec_events_m=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_sm_tot']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        cons_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_sm_dec_events_m[i,j]=cons_sm_mult1_pc
        

cons_sm_dec_events_s=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'cons_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        cons_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        cons_sm_dec_events_s[i,j]=cons_sm_mult1_pc


cons_sm_dec_events_df_s=pd.DataFrame(data=cons_sm_dec_events_s)
cons_sm_dec_events_df_s.to_csv('cons_sm_c2_{}.csv'.format(str(args.seed)))
cons_sm_dec_events_df_m=pd.DataFrame(data=cons_sm_dec_events_m)
cons_sm_dec_events_df_m.to_csv('cons_sm_factual_{}.csv'.format(str(args.seed)))
cons_sm_dec_events_df=pd.DataFrame(data=cons_sm_dec_events)
cons_sm_dec_events_df.to_csv('abs_diff_cons_sm_{}.csv'.format(str(args.seed)))
cons_sm_dec_events_rel_df=pd.DataFrame(data=cons_sm_dec_events_rel)
cons_sm_dec_events_rel_df.to_csv('rel_diff_cons_sm_{}.csv'.format(str(args.seed)))

wb_dec_events=np.zeros((len(deciles), len(n_events)))
wb_dec_events_rel=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        mult2=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_mult2_pc=mult2/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_dec_events[i,j]=wb_mult1_pc-wb_mult2_pc
        
        wb_dec_events_rel[i,j]=((wb_mult1_pc/wb_mult2_pc)*100-100)

wb_dec_events_df=pd.DataFrame(data=wb_dec_events)
wb_dec_events_df.to_csv('abs_diff_wb_{}.csv'.format(str(args.seed)))
wb_dec_events_rel_df=pd.DataFrame(data=wb_dec_events_rel)
wb_dec_events_rel_df.to_csv('rel_diff_wb_{}.csv'.format(str(args.seed)))


wb_dec_events_m=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_dec_events_m[i,j]=wb_mult1_pc
wb_dec_events_df_m=pd.DataFrame(data=wb_dec_events_m)
wb_dec_events_df_m.to_csv('wb_factual_{}.csv'.format(str(args.seed)))

wb_dec_events_s=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_dec_events_s[i,j]=wb_mult1_pc
wb_dec_events_df_s=pd.DataFrame(data=wb_dec_events_s)
wb_dec_events_df_s.to_csv('wb_c2_{}.csv'.format(str(args.seed)))


wb_sm_dec_events=np.zeros((len(deciles), len(n_events)))
wb_sm_dec_events_rel=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_sm']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        mult2=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_sm_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_sm_mult2_pc=mult2/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_sm_dec_events[i,j]=wb_sm_mult1_pc-wb_sm_mult2_pc
        
        wb_sm_dec_events_rel[i,j]=((wb_sm_mult1_pc/wb_sm_mult2_pc)*100-100)

wb_sm_dec_events_df=pd.DataFrame(data=wb_sm_dec_events)
wb_sm_dec_events_df.to_csv('abs_diff_wb_sm_{}.csv'.format(str(args.seed)))
wb_sm_dec_events_rel_df=pd.DataFrame(data=wb_sm_dec_events_rel)
wb_sm_dec_events_rel_df.to_csv('rel_diff_wb_sm_{}.csv'.format(str(args.seed)))


wb_sm_dec_events_m=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_sm']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_sm_dec_events_m[i,j]=wb_sm_mult1_pc
wb_sm_dec_events_df_m=pd.DataFrame(data=wb_sm_dec_events_m)
wb_sm_dec_events_df_m.to_csv('wb_sm_factual_{}.csv'.format(str(args.seed)))

wb_sm_dec_events_s=np.zeros((len(deciles), len(n_events)))
for i, dec in enumerate(deciles):
    for j, c_ev in enumerate(n_events):
        
        mult1=(survey.loc[(survey['pcinc_decile']==dec) & 
                        (survey['n_events']==c_ev), 'wb_loss_sm_c2']*survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        
        wb_sm_mult1_pc=mult1/(survey.loc[(survey['pcinc_decile']==dec)&
                                                                          (survey['n_events']==c_ev), 'hh_weight']).sum()
        wb_sm_dec_events_s[i,j]=wb_sm_mult1_pc
wb_sm_dec_events_df_s=pd.DataFrame(data=wb_sm_dec_events_s)
wb_sm_dec_events_df_s.to_csv('wb_sm_c2_{}.csv'.format(str(args.seed)))



