#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:41:42 2020

@author: insauer
"""
import os
import random
import numpy as np
import pandas as pd
import re
import multiprocessing as mp
from functools import partial
from hhwb.util.constants import  DT_STEP, RECO_PERIOD
from datetime import datetime, date
from zipfile import ZipFile


AGENT_TYPE = 'SH'

REGION_DICT = { 
                15: 'PH150000000',
                14: 'PH140000000',
                13: 'PH130000000',
                1: 'PH010000000',
                2: 'PH020000000',
                3: 'PH030000000',
                41: 'PH040000000',
                42: 'PH170000000',
                9: 'PH090000000',
                5: 'PH050000000',
                6: 'PH060000000',
                7: 'PH070000000',
                8: 'PH080000000',
                10: 'PH100000000',
                11: 'PH110000000',
                12: 'PH120000000',
                16: 'PH160000000'}


REGIONS = list(REGION_DICT)



class Shock():
    """Shock definition. This class builds the intersection with the hazard forcing and provides direct
       damage obtained from hazard forcing and the affected households.

        Attributes:
            aff_hh (list): list with affected households
            unaff_hh (list): list with uneffected households
            aff_hh_id (list): list with IDs of affected households
            unaff_hh_id (list): list with IDs of unaffected households
            L (float): total damage
    """

    def __init__(self):

        self.__time_stemps = []
        self.__event_names = []
        self.__aff_ids = np.array([[]])

        self.__L = 0

    @property
    def aff_ids(self):
        return self.__aff_ids

    @property
    def time_stemps(self):
        return self.__time_stemps

    @property
    def unaff_hh(self):
        return self.__unaff_hh

    @property
    def L(self):
        return self.__L

    @property
    def dt(self):
        return self.__dt
    
    @staticmethod
    def apply_shock(hh, L, K, L_pub, dt_reco, affected_hhs, vuls):
        """wrapper function for the shocking"""

        if hh.hhid in affected_hhs:
            hh.shock(aff_flag=True, L_pub=L_pub, L=L, K=K, dt=dt_reco, vul=vuls[int(hh.hhid)])
        else:
            hh.shock(aff_flag=False, L_pub=L_pub, L=L, K=K, dt=dt_reco, vul=vuls[int(hh.hhid)])
        return hh
    

    def contains_date_format(self, series):
        
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
        
        return series[series.astype(str).str.contains(date_pattern)]
    
    

    # Function to extract dates from a string
    def __extract_dates(self, string):
        # Regular expression to match dates
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
        return date_pattern.findall(string)
    
    def read_shock(self, work_path, path, event_identifier,run):
        
        """this is a function that only reads already preprocessed shock data (households - events)
           """
           
        with ZipFile(work_path + path, 'r') as zip_ref:
            # Extract the CSV file to a temporary directory
            zip_ref.extract('test_shocks.csv', path='temp')
        
        shock_data = pd.read_csv(work_path + path)
        
        start_date = date(2008,1,1)
        
        event_names = [col for col in shock_data.columns if event_identifier in col]
        
        event_data=shock_data[event_names]
        
        dates_list = [datetime.strptime(event, '%Y-%m-%d').date() for event in event_names]
        
        month = [np.round((event_date - start_date).days/28) for event_date in dates_list]
        
        self.__aff_ids=np.zeros((event_data.shape[0],len(list(set(month)))))
        
        week_nums=list(set(month))
        
        week_nums.sort()
        
        for m, week in enumerate(week_nums):
            
            locat = np.where(np.array(month)==week)
            
            self.__aff_ids[:, m]=event_data.iloc[:, locat[0]].sum(axis=1).clip(upper=1)
        
        self.__time_stemps = week_nums
        
        shock_df = pd.DataFrame(data=self.__aff_ids, columns=np.array(week_nums).astype(int).astype(str))
        
        """Some events are aggregated as they fall within the same time step of the model and are
        treated as one event, this aggregation is saved here"""
        
        shock_df.to_csv('shocks_aggregated.csv')
        #shock_df.to_csv(work_path + '/data/output/shocks/shocks_syn_aggregated.csv')
        
        return
    
    def __get_months(self, start_date, strings):
    
        # Regular expression to match dates
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')

        # Reference date
        reference_date = datetime.strptime("2000-01-01", "%Y-%m-%d")

        # Extract dates and calculate days since reference date
        date_differences = []

        for s in strings:
            dates = date_pattern.findall(s)
            for date_str in dates:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                days_since_reference = (date_obj - reference_date).days
                date_differences.append(int(np.round(days_since_reference/28)))
    
        return date_differences
    
    def read_vul_shock(self, path, output_path, file, start_year):
        
        """this is a function that only reads already preprocessed shock data (households - events)
           """
           
           
        with ZipFile(path, 'r') as zip_ref:
            # Extract the CSV file to a temporary directory
            zip_ref.extract(file, path='temp')
        
        shock_data = pd.read_csv(path)
        
        # Select columns that contain the date format
        event_names = self.contains_date_format(shock_data.columns)

        # Filter the DataFrame with the selected columns
        selected_shocks = shock_data[event_names]
        
        selected_shocks=selected_shocks.loc[:,selected_shocks.sum()>0]
        
        event_names =selected_shocks.columns
        
        # Apply the function to the DataFrame
        months=self.__get_months(start_year, event_names)

        self.__aff_ids=np.zeros((selected_shocks.shape[0],len(list(set(months)))))
        
        n_agg_events=list(set(months))
        
        n_agg_events.sort()
        
        for m, week in enumerate(n_agg_events):
            
            locat = np.where(np.array(months)==week)
            
            self.__aff_ids[:, m]=selected_shocks.iloc[:, locat[0]].max(axis=1)
        
        self.__time_stemps = n_agg_events
        
        shock_df = pd.DataFrame(data=self.__aff_ids, columns=np.array(n_agg_events).astype(int).astype(str))

        shock_df.to_csv(output_path+'shocks_aggregated.csv')
        
        return
    
    
   
    
    def shock(self, event_index, gov, hhs, dt_reco, cores):
        """This function shocks affected households. Distributes shocks to multiple cores

            Parameters
            ----------
            event_index (int) : index of the current event
            gov (Gov) : Government
                the government of all households
            hhs (list HH) : households
            dt_reco (float) : time step
            cores (int) : number available threads
        """
        test=self.__aff_ids
        affected_hhs = np.where(self.__aff_ids[:, event_index]>0.00000000001)[0]
        vuls=self.__aff_ids[:, event_index]
        
        # L_t = 0.
        # L_pub_t = 0.
        L_t = gov.L_t
        L_pub_t = gov.L_pub_t
        print('current')
        
        
        for h_ind, hh in enumerate(hhs):
             if hh.hhid in affected_hhs:
            
                L_pub_t += vuls[int(hh.hhid)] * (hh.k_pub_0 - hh.d_k_pub_t)*hh.weight
                L_t += vuls[int(hh.hhid)]  * ((hh.k_pub_0 - hh.d_k_pub_t) + (hh.k_priv_0 - hh.d_k_priv_t))*hh.weight
        
        

                
        gov.shock(aff_flag=True, L=L_t, L_pub=L_pub_t, dt=dt_reco)
        p = mp.Pool(cores)       
        prod_x = partial(Shock.apply_shock, L=L_t, K=gov.K,L_pub=L_pub_t,
                         dt_reco=dt_reco, affected_hhs=affected_hhs, vuls=vuls)
        
        hhs = p.map(prod_x, hhs)
        p.close()
        p.join()
        # print('shock shocks')
        # for h_ind, hh in enumerate(hhs):
        #     if h_ind in affected_hhs:
        #         hh.shock(aff_flag=True, L=L_t, K=gov.K, dt=dt_reco)
        #     else:
        #         hh.shock(aff_flag=False, L=L_t, K=gov.K, dt=dt_reco)

        

        return hhs


    def set_random_shock(self, n_events=3, n_hhs=10):
        """The function selects households randomly and shocks them. (Function is only a
           placeholder for a real Hazard intersection.

            Parameters
            ----------
            hhs : list
                list with all households
        """

        self.__aff_ids = np.zeros((n_hhs, n_events))
        self.__set_time_stemps(n_events)

        for ev in range(n_events):
            sel_hhs = self.__set_aff_hhs(n_hhs)
            self.__aff_ids[sel_hhs, ev] = 1

        return
    
    def set_shock_from_csv(self, work_path='/home/insauer/projects/WB_model/hhwb',
                           path_haz='/data/hazard_data/PHL_sat/haz_dat_30as_{}_vul.csv',
                           path_hh='/data/surveys_prepared/PHL/region_hh_full_pack_PHL.csv',
                           hh_reg=None, k_eff=0, seed=2020):
        
        """this function generates shocks based on distributed households and hazard affected grid
         cells"""


        df_hh = pd.read_csv(work_path + path_hh)
        
        
        for r, reg in enumerate(REGIONS):

            event_names, weeks = self.__set_time_stemps_disaster_set(work_path, path_haz, reg, '-')
            
            if r==0:
                self.__aff_ids = np.zeros((len(df_hh), len(event_names)))
                self.__time_stemps = weeks
                

            self.__set_aff_hhs_disaster_set(work_path, path_haz, path_hh, reg, event_names, seed)
        
        
        shock_df = pd.DataFrame(data=self.__aff_ids, columns=event_names)
        
        shock_df['region']=df_hh['region']
        
        shock_df.to_csv('/home/insauer/mnt/ebm/inga/hhwb/data/shock_data/shocks_seed/shocks_{}.csv'.format(str(seed)))


        return
    
    def generate_single_shocks(self, work_path='/home/insauer/projects/WB_model/hhwb',
                           path_haz='/data/output/shocks/shocks_99.csv',
                           path_hh='/data/surveys_prepared/PHL/region_hh_full_pack_PHL_pop.csv',
                           path_hh_orig='/data/surveys_prepared/PHL/survey_PHL.csv',
                           hh_reg=None, k_eff=0, seed=2020):
        
        df_hh = pd.read_csv(work_path + path_hh)
        df_hh_orig = pd.read_csv(work_path + path_hh_orig)
        df_shock = pd.read_csv(work_path + '/data/shock_data/shocks_seed/shocks_{}.csv'.format(str(seed)))
        
        self.__time_stemps = df_shock.columns[1:-1]#.astype(int)
        
        event_names = df_shock.columns[1:]
        
        df_shock['region']=df_hh['region']
        
        df_shock['fhhid']=df_shock.index
        
        new_survey_data=pd.DataFrame()
        
        for r, reg in enumerate(REGIONS):
            start_date = date(2002,1,1)
            print(reg)
            
            df_hh_reg = df_hh.loc[df_hh['region']==reg]
            df_hh_orig_reg = df_hh_orig.loc[df_hh_orig['region']==reg]
            df_shock_reg = df_shock.loc[df_shock['region']==reg]
            
            aff_hh_data = self.__shock_hh_without_location(df_hh_reg, df_hh_orig_reg, df_shock_reg, reg, seed)
            
            new_survey_data=new_survey_data.append(aff_hh_data, ignore_index=True)
        
        self.__aff_ids = np.zeros((new_survey_data.shape[0], len(event_names)))
        
        for i, sh in enumerate(self.__time_stemps):
            
            self.__aff_ids[:,i]=(new_survey_data.loc[:, 'event']==i).astype(int)
        
        shocks = pd.DataFrame(data=self.__aff_ids, columns=event_names)
        
        shocks['region']=new_survey_data['region']
        
        new_survey_data.to_csv(work_path + '/data/survey_data/PHL/survey_seed/region_hh_full_pack_PHL_pop_syn_{}.csv'.format(seed))
        
        shocks.to_csv(work_path + '/data/shock_data/shocks_syn_seed/shocks_syn_{}'.format(seed))
        
        return


    def __shock_hh_without_location(self, df_hh_reg, df_hh_orig_reg, df_shock_reg, reg, seed):
        
        """this function generates shocks based on distributed households and hazard affected grid
         cells"""
        
        np.random.seed(seed)
        
        c_aff_hhs = 0
        
        hhids=list(df_hh_orig_reg['hhid'])
        
        df_hh_orig_reg['n_copied']=0
        df_hh_orig_reg['event']=-1
        df_hh_orig_reg['hh_instance']=0
        
        aff_hh_data = pd.DataFrame()
        
        for i,shock in enumerate(self.__time_stemps):
            print(shock)
            aff_ids_shock = list(df_shock_reg.loc[df_shock_reg[shock]==1,'fhhid'])
            n_aff_shock = df_hh_reg.loc[df_hh_reg['fhhid'].isin(aff_ids_shock), 'n_individuals'].sum()
            
            c_hh = 0
            
            while c_hh < n_aff_shock:
                
                hh_ind = np.random.choice(hhids)
                #print(len(hhids))
                if df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'weight'].sum() <= \
                    df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'n_individuals'].sum():
                    hhids.remove(hh_ind)
                    print('stop')
                    print(len(hhids))
                    continue
                
                aff_hh_data = aff_hh_data.append(df_hh_orig_reg.loc[df_hh_orig_reg['hhid']==hh_ind], ignore_index=True)
                aff_hh_data.loc[c_aff_hhs,'weight'] =  df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'n_individuals'].sum()
                aff_hh_data.loc[c_aff_hhs,'hh_instance'] = df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'n_copied'].sum()+1
                aff_hh_data.loc[c_aff_hhs, 'event'] = i
                df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind, 'n_copied'] += 1
                df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind, 'weight'] -= df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'n_individuals'].sum()
                
                c_hh +=df_hh_orig_reg.loc[df_hh_orig_reg['hhid']== hh_ind,'n_individuals'].sum()

                c_aff_hhs += 1
                
            aff_hh_data['region']=reg
        aff_hh_data = df_hh_orig_reg.append(aff_hh_data, ignore_index=True)
        
        return aff_hh_data

    def __set_time_stemps_disaster_set(self, work_path, path_haz, reg, event_identifier):
        
        start_date = date(2000,1,1)
        
        shock_series = pd.read_csv((work_path + path_haz).format(reg))

        event_names = [col for col in shock_series.columns if '-' in col]
        
        dates_list = [datetime.strptime(event, '%Y-%m-%d').date() for event in event_names]
        
        weeks = [np.round((event_date - start_date).days/7) for event_date in dates_list]
        
        return event_names, weeks
    
    
    
    # def __set_aff_hhs_disaster_set_damage(self, df_haz, df_hh, hh_reg, k_eff_reg):

    #     region_dict = {'PH150000000': 15,
    #                    'PH140000000': 14,
    #                    'PH130000000': 13,
    #                    'PH010000000': 1,
    #                    'PH020000000': 2,
    #                    'PH030000000': 3,
    #                    'PH040000000': 41,
    #                    'PH170000000': 42,
    #                    'PH090000000': 9,
    #                    'PH050000000': 5,
    #                    'PH060000000': 6,
    #                    'PH070000000': 7,
    #                    'PH080000000': 8,
    #                    'PH100000000': 10,
    #                    'PH110000000': 11,
    #                    'PH120000000': 12,
    #                    'PH160000000': 16}
        
    #     aff_ids = np.zeros((len(hh_reg), len(self.__time_stemps)))
        
    #     for ev_id, event in enumerate(self.__event_names):
    #         aff_hhs = []
    #         affected_cells = df_haz.loc[df_haz[event] !=0, 'Centroid_ID']
    #         for cell in affected_cells:
    #             hhids = np.array(df_hh.loc[df_hh['Centroid_ID']==cell, 'fhhid'])
    #             min_k_eff = 0
    #             sum_k_eff = k_eff_reg
    #             rel_dam = df_haz.loc[(df_haz['Centroid_ID']==cell),event].values[0]
    #             abs_dam = rel_dam * sum_k_eff
    #             dam = 0
    #             print(abs_dam)
    #             while dam<abs_dam:
    #                 print(dam)
    #                 print(ev_id)
    #                 print(cell)
    #                 hh_ind = random.randint(0, hhids.shape[0]-1)
    #                 if not np.isin(hhids[hh_ind], aff_hhs):
    #                     aff_hhs.append(hhids[hh_ind])
    #                     dam += hh_reg[hhids[hh_ind]].weight*hh_reg[hhids[hh_ind]].vul*(hh_reg[hhids[hh_ind]].k_eff_0 - hh_reg[hhids[hh_ind]].d_k_eff_t)
    #                 all_aff = np.isin(hhids, aff_hhs).sum()
    #                 if all_aff == hhids.shape[0]:
    #                     hh_ind = random.randint(0, hhids.shape[0]-1)
    #                     new_vul = hh_reg[hhids[hh_ind]].vul+0.1
    #                     if new_vul < 0.6:
    #                         hh_reg[hhids[hh_ind]].set_vul(new_vul)
    #                         dam+= 0.1* hh_reg[hhids[hh_ind]].weight*(hh_reg[hhids[hh_ind]].k_eff_0 - hh_reg[hhids[hh_ind]].d_k_eff_t)
                        
    #         aff_ids[aff_hhs, ev_id]=1
    #     return
    
    def __set_aff_hhs_disaster_set(self, work_path, path_haz, path_hh, reg, event_names, seed):
        np.random.seed(seed)
        print('region')
        print(reg)
        df_haz = pd.read_csv((work_path + path_haz).format(reg))
        df_hh = pd.read_csv(work_path + path_hh)
        df_hh = df_hh[df_hh['region'] == reg]
        for ev_id, event in enumerate(event_names):
            print(event)
            aff_hhs = []
            affected_cells = df_haz.loc[df_haz[event] !=0, 'Centroid_ID']
            if affected_cells.shape[0]==0:
                continue
            for cell in affected_cells:
                hhids = list(df_hh.loc[df_hh['centroid_id'] == cell, 'fhhid'])
                
                n_people = df_hh.loc[df_hh['centroid_id'] == cell, 'weight'].sum()
                if n_people == 0.0:
                    continue
                frac = df_haz.loc[(df_haz['Centroid_ID'] == cell), event].values[0]
                n_aff_people = np.round(frac* n_people)
                c_n_aff_hhs = 0
                while (c_n_aff_hhs < n_aff_people) and (len(hhids)>0):
                    hh_ind = np.random.choice(hhids)
                    aff_hhs.append(hh_ind)
                    c_n_aff_hhs += df_hh.loc[df_hh['fhhid']==hh_ind].n_individuals.values[0]
                    hhids.remove(hh_ind)
                
            self.__aff_ids[aff_hhs, ev_id]=1
                
        return

    # def __get_k_eff_reg(self, hh_reg):

    #     for reg in hh_reg.region_hhs:
            
    #         for hhid in hh_reg.region_hhs[reg]:
                

    #         #print('hä')


    def __set_aff_hhs(self, n_hhs):
        """The function selects households randomly and shocks them. (Function is only a
           placeholder for a real Hazard intersection.
    
            Parameters
            ----------
            hhs : list
                list with all households
        """
        n_aff_hh = 6
        sel_hhs = []

        # while len(sel_hhs) < n_aff_hh:
        #     hh = random.randint(0, n_hhs-1)
        #     if not np.isin(hh, sel_hhs):
        #         sel_hhs.append(hh)
        #         print('Household appended')

        return [0,1,2,3,4,5]

        
    def __set_time_stemps(self, n_events=2):
        """The function selects households randomly hocks them. (Function is only a
           placeholder for a real Climade intersection.

            Parameters
            ----------
            n_events : int
                number events
        """

        self.__time_stemps = [50]
        self.__event_names = ['0']
        add = 0
        for run in range(n_events-1):
            self.__time_stemps.append(random.randint(200+add, 300+add))
            self.__event_names.append(str(self.__time_stemps[run]))
            add+=300
     

    def __plot(self, ax):
        """The function selects households randomly and plots their recovery.

            Parameters
            ----------
            gov : Government
                the government of all households
            hhs : Households
                list with households
        """

        ax.set_xlim((0, RECO_PERIOD*DT_STEP+1))
        ax.set_xlabel('weeks after shock')
        ax.set_ylabel('')
        ax.set_title('Consumption_recovery')
        #ax.legend()

        return
