#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:19:13 2020

@author: insauer
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp
from functools import partial
import time
from hhwb.agents.government import Government
from hhwb.util.constants import DT_STEP, TEMP_RES, RECO_PERIOD
import pandas as pd
import csv



class ClimateLife():
    
    """This class builds the dynamic environment of the model. It controls the runtime and the
       interaction of the agents as well as the data storage.

        Attributes:
            @param hhs (list): list with all households
            @param gov (Gov): list with uneffected households
            @param pt (np.array): array with timesteps
            @param dt_life: list with IDs of unaffected households

    """

    def __init__(self, hhs, shock, gov):
        
        """! constructor"""
        
        self.__hhs = hhs
        self.__shock = shock
        self.__gov = gov
        self.__pt = np.array([])
        self.__dt_life = np.array([])

        
        
    @property
    def dt_life(self):
        
        return self.__dt_life
    

    @staticmethod
    def update_reco(hh, gov):
        """update households recovery state"""
        hh.update_reco(gov.L_t, gov.L_pub_t, gov.K_pub, gov.K)

        return hh
    
    def __update_records(self, t_i):
        
        kpub = np.zeros((len(self.__hhs)))
        kpriv = np.zeros((len(self.__hhs)))
        
        ipub = np.zeros((len(self.__hhs)))
        ipriv = np.zeros((len(self.__hhs)))
        inc = np.zeros((len(self.__hhs)))
        inc_sp = np.zeros((len(self.__hhs)))
        
        cons = np.zeros((len(self.__hhs)))
        cons_priv = np.zeros((len(self.__hhs)))
        cons_sm = np.zeros((len(self.__hhs)))
        cons_priv_sm = np.zeros((len(self.__hhs)))
        
        wb = np.zeros((len(self.__hhs)))
        wb_sm = np.zeros((len(self.__hhs)))
        sav = np.zeros((len(self.__hhs)))
        
        for hh in self.__hhs:

            if t_i % TEMP_RES == 0:
                kpub[int(hh.hhid)] = hh.d_k_pub_t
                kpriv[int(hh.hhid)] = hh.d_k_priv_t
                
                ipub[int(hh.hhid)] = hh.d_inc_pub_t
                ipriv[int(hh.hhid)] = hh.d_inc_priv_t
                
                inc[int(hh.hhid)] = hh.d_inc_t
                inc_sp[int(hh.hhid)] = hh.d_inc_sp_t
                
                cons[int(hh.hhid)]  = hh.d_con_eff_t
                cons_priv[int(hh.hhid)] = hh.d_con_priv_t
                
                cons_sm[int(hh.hhid)] = hh.d_con_eff_sm
                cons_priv_sm[int(hh.hhid)] = hh.d_con_priv_sm

                wb[int(hh.hhid)] = hh.d_wb_t
                wb_sm[int(hh.hhid)] = hh.d_wb_sm
                
                sav[int(hh.hhid)] = hh.savings
            hh.t += hh.dt
        self.__gov.t+=self.__gov.dt

        return kpub, kpriv, ipub, ipriv, inc, inc_sp, cons, cons_priv, cons_sm, cons_priv_sm, wb, wb_sm, sav
    
    

    def start(self, work_path='/home/insauer/projects/WB_model/hhwb',
              result_path='/data/output/', cores=1, reco_period=0):
        """main method that starts and controls the dynamic model
        
        Parameters
        ----------
        work_path : str
            general path of the model
        result_path : str
            output path for results
        cores : int
            number of available threads used for multi-core processing
        reco_period : TYPE
            run time of the model in years
    
        Returns
        -------
        None. """
        
        print('Life started')
        
        
        """set up of runtime and temporal resolution"""
        pt = np.linspace(0, reco_period, reco_period*DT_STEP+1)
        dt_reco = np.diff(pt)[0]
        
        self.__pt = pt[::4]
        
        self.__dt_life = np.arange(0, reco_period*DT_STEP+1)
        

        #print('Tax rate: ' + str(self.__gov))
        print('Total expenditure on social programs: ' + str(self.__gov.sp_cost))
        print('Total national capital stock: ' + str(self.__gov.K))
        
        """"set up data storage"""
        
        colnames = np.arange(len(self.__hhs)).astype(str)
        
        keff = pd.DataFrame(columns=colnames)
        kpub = pd.DataFrame(columns=colnames)
        kpriv = pd.DataFrame(columns=colnames)
        
        ipub = pd.DataFrame(columns=colnames)
        ipriv = pd.DataFrame(columns=colnames)
        inc_ =  pd.DataFrame(columns=colnames)
        inc_sp_ =  pd.DataFrame(columns=colnames)
        cons_ =  pd.DataFrame(columns=colnames)
        cons_priv_ = pd.DataFrame(columns=colnames)
        cons_sm_ =  pd.DataFrame(columns=colnames)
        cons_priv_sm_ =  pd.DataFrame(columns=colnames)
        wb_ =  pd.DataFrame(columns=colnames)
        wb_sm_ =  pd.DataFrame(columns=colnames)
        sav_ =  pd.DataFrame(columns=colnames)
        gov_ =  pd.DataFrame(columns=['kpub', 'kpriv', 'ipub', 'ipriv','inc', 'inc_sp', 'cons',
                                      'cons_priv', 'cons_sm', 'cons_priv_sm', 'wb', 'wb_sm'])
        
        keff.to_csv(work_path+result_path+'keff.csv')
        kpub.to_csv(work_path+result_path+'kpub.csv')
        kpriv.to_csv(work_path+result_path+'kpriv.csv')
        
        ipub.to_csv(work_path+result_path+'ipub.csv')
        ipriv.to_csv(work_path+result_path+'ipriv.csv')
        
        inc_.to_csv(work_path+result_path+'inc.csv')
        inc_sp_.to_csv(work_path+result_path+'inc_sp.csv')
        cons_.to_csv(work_path+result_path+'cons.csv')
        cons_priv_.to_csv(work_path+result_path+'cons_priv.csv')
        cons_sm_.to_csv(work_path+result_path+'cons_sm.csv')
        cons_priv_sm_.to_csv(work_path+result_path+'cons_priv_sm.csv')
        wb_.to_csv(work_path+result_path+'wb.csv')
        wb_sm_.to_csv(work_path+result_path+'wb_sm.csv')
        sav_.to_csv(work_path+result_path+'sav.csv')
        gov_.to_csv(work_path+result_path+'gov.csv')
        

        n_shock = 0
        
        
        with open(work_path+result_path+'keff.csv', 'w', newline='') as f_keff,\
             open(work_path+result_path+'kpub.csv', 'w', newline='') as f_kpub,\
             open(work_path+result_path+'kpriv.csv', 'w', newline='') as f_kpriv,\
             open(work_path+result_path+'cons_priv.csv', 'w', newline='') as f_cons_priv,\
             open(work_path+result_path+'cons_priv_sm.csv', 'w', newline='') as f_cons_priv_sm,\
             open(work_path+result_path+'cons_sm.csv', 'w', newline='') as f_cons_sm,\
             open(work_path+result_path+'cons.csv', 'w', newline='') as f_cons,\
             open(work_path+result_path+'wb.csv', 'w', newline='') as f_wb,\
             open(work_path+result_path+'ipub.csv', 'w', newline='') as f_ipub,\
             open(work_path+result_path+'ipriv.csv', 'w', newline='') as f_ipriv,\
             open(work_path+result_path+'inc.csv', 'w', newline='') as f_inc,\
             open(work_path+result_path+'inc_sp.csv', 'w', newline='') as f_inc_sp,\
             open(work_path+result_path+'wb_sm.csv', 'w', newline='') as f_wb_sm,\
             open(work_path+result_path+'sav.csv', 'w', newline='') as f_sav,\
             open(work_path+result_path+'gov.csv', 'w', newline='') as f_gov:
                 
                 
                 # open(work_path+result_path+'cons.csv', 'w', newline='') as f_cons,\
                 
            writer_keff=csv.writer(f_keff, delimiter=',')
            writer_kpub=csv.writer(f_kpub, delimiter=',')
            writer_kpriv=csv.writer(f_kpriv, delimiter=',')
            
            writer_ipub=csv.writer(f_ipub, delimiter=',')
            writer_ipriv=csv.writer(f_ipriv, delimiter=',')
            
            writer_inc=csv.writer(f_inc, delimiter=',')
            writer_incsp=csv.writer(f_inc_sp, delimiter=',')
            
            writer_cons = csv.writer(f_cons, delimiter=',')
            writer_cons_sm = csv.writer(f_cons_sm, delimiter=',')
            writer_cons_priv = csv.writer(f_cons_priv, delimiter=',')
            writer_cons_priv_sm = csv.writer(f_cons_priv_sm, delimiter=',')
            
            writer_wb=csv.writer(f_wb, delimiter=',')
            writer_wb_sm=csv.writer(f_wb_sm, delimiter=',')
            writer_sav=csv.writer(f_sav, delimiter=',')
            writer_gov=csv.writer(f_gov, delimiter=',')
            
            for t_i in self.__dt_life:
                print(t_i)
                print(dt_reco)
    
                if t_i in self.__shock.time_stemps:
                    print('shock start')
                    self.__hhs = self.__shock.shock(n_shock, self.__gov, self.__hhs, dt_reco, cores)
                    n_shock += 1
                    #dt_s = 0
                
                else:
                    self.__gov.update_reco(self.__hhs)
                    p = mp.Pool(cores)
                    prod_x=partial(ClimateLife.update_reco, gov=self.__gov)
                    self.__hhs=p.map(prod_x, self.__hhs)
                    p.close()
                    p.join()
                    
                self.__gov.collect_hh_info(t_i, self.__hhs)
    
                kpub, kpriv, ipub, ipriv, inc, inc_sp, cons, cons_priv, cons_sm, cons_priv_sm, wb, wb_sm, sav = self.__update_records(t_i)
                
                
                gov_res = [self.__gov.L_t, self.__gov.d_k_priv_t, self.__gov.L_pub_t, self.__gov.d_inc_t,
                           self.__gov.d_inc_sp_t, self.__gov.d_con_priv_t, self.__gov.d_con_eff_t,
                           self.__gov.d_con_eff_sm, self.__gov.d_con_priv_sm,
                           self.__gov.d_wb_t, self.__gov.d_wb_sm, self.__gov.pub_debt]
                

                writer_keff.writerow(list(kpub+kpub))

                writer_kpub.writerow(list(kpub))
                writer_kpriv.writerow(list(kpriv))
                
                writer_ipub.writerow(list(ipub))
                writer_ipriv.writerow(list(ipriv))
                
                writer_inc.writerow(list(inc))
                writer_incsp.writerow(list(inc_sp))

                writer_cons.writerow(list(cons))
                
                writer_cons_sm.writerow(list(cons_sm))
                writer_cons_priv.writerow(list(cons_priv))
                writer_cons_priv_sm.writerow(list(cons_priv_sm))
                
                writer_wb.writerow(list(wb))
                writer_wb_sm.writerow(list(wb_sm))
                writer_sav.writerow(list(sav))
                writer_gov.writerow(gov_res)
            
            f_keff.close()
            f_kpub.close()
            f_kpriv.close()
            f_ipub.close()
            f_ipriv.close()
            f_inc.close()
            f_inc_sp.close()
            f_cons.close()
            f_cons_priv.close()
            f_cons_sm.close()
            f_cons_priv_sm.close()
            f_wb.close()
            f_wb_sm.close()
            f_sav.close()
            f_gov.close()
    
                # self.__plot_info(n_plot_hhs=5, plot_hhs=plot_ids)
    
                # if t_i%12 ==0:
                #     #plt.tight_layout()
                #     plt.show(block=False)
                #     plt.pause(0.01)
        return

    def _set_agents(self):

        return
    
    def __get_plot_hhs(self, n_plot_hhs=10):
        
        sort_aff = np.argsort(self.__shock.aff_ids.sum(axis=1))
        plot_hhs = sort_aff[-n_plot_hhs:]
        #print(plot_hhs)
        return [0, 1, 5, 7]
    
    def __plot_info(self, n_plot_hhs=4, plot_hhs=None):

        self.__plot_cons(n_plot_hhs=4, plot_hhs=plot_hhs)
        self.__plot_inc(n_plot_hhs=4, plot_hhs=plot_hhs)
        self.__plot_inc_sp(n_plot_hhs=4, plot_hhs=plot_hhs)
        self.__plot_k_eff(n_plot_hhs=4, plot_hhs=plot_hhs)
        self.__plot_wb(n_plot_hhs=4, plot_hhs=plot_hhs)
        self.__plot_info_gov()

        #self.__info_summary.suptitle('Household Resilience Model', va= 'top')

    def __plot_cons(self, n_plot_hhs=4, plot_hhs=None):
        self.__abs_cons.clear()
        self.__abs_cons.set_xlim((-1, RECO_PERIOD))
        #self.__abs_cons.set_ylim((0, 30000))
        self.__abs_cons.set_title('HH total consumption loss USD')
        self.__abs_cons.set_ylabel('Absolute loss in USD')
        #self.__abs_cons.set_xlabel('time in years')
        colours = ['blue', 'red', 'green', 'gold', 'brown', 'black', 'purple', 'pink','seagreen', 'firebrick']

        for h, phh in enumerate(plot_hhs):
            self.__abs_cons.plot(self.__pt, self.cons_reco[:,phh], color = colours[h])
            # self.__abs_cons.plot(self.__pt, self.cons_reco_sm[:,phh], color = colours[h])
            self.__abs_cons.axhline(y=self.__hhs[phh].consum_0 - self.__hhs[phh].subsistence_line, color = colours[h])

    def __plot_inc(self, n_plot_hhs=4, plot_hhs=None):
        self.__abs_inc.clear()
        self.__abs_inc.set_xlim((-1, RECO_PERIOD))
        #self.__abs_inc.set_ylim((0, 20000))
        self.__abs_inc.set_title('HH total income loss USD')
        self.__abs_inc.set_ylabel('Absolute loss in USD')
        self.__abs_inc.set_xlabel('time in years')
        colours = ['blue', 'red', 'green', 'gold', 'brown', 'black', 'purple', 'pink','seagreen', 'firebrick']
        
        for h, phh in enumerate(plot_hhs):
            self.__abs_inc.plot(self.__pt, self.inc_reco[:, phh], color = colours[h],
                               label='HH' + str(self.__hhs[phh].hhid))
        
        self.__abs_inc.legend()
        
    def __plot_inc_sp(self, n_plot_hhs=4, plot_hhs=None):
        self.__abs_inc_sp.clear()
        self.__abs_inc_sp.set_xlim((-1, RECO_PERIOD))
        #self.__abs_inc_sp.set_ylim((0, 3000))
        self.__abs_inc_sp.set_title('HH income loss from social programs USD')
        self.__abs_inc_sp.set_ylabel('Absolute loss in USD')
        self.__abs_inc_sp.set_xlabel('time in years')
        colours = ['blue', 'red', 'green', 'gold', 'brown', 'black', 'purple', 'pink','seagreen', 'firebrick']
        
        for h, phh in enumerate(plot_hhs):
            self.__abs_inc_sp.plot(self.__pt, self.inc_sp_reco[:, phh], color = colours[h],
                               label='HH' + str(self.__hhs[phh].hhid))
        
        
    def __plot_k_eff(self, n_plot_hhs=4, plot_hhs=None):
        self.__abs_k.clear()
        self.__abs_k.set_xlim((-1, RECO_PERIOD))
        #self.__abs_k.set_ylim((0, 80000))
        self.__abs_k.set_title('HH capital stock damage')
        self.__abs_k.set_ylabel('Absolute damage in USD')
        self.__abs_k.set_xlabel('time in years')
        colours = ['blue', 'red', 'green', 'gold', 'brown', 'black', 'purple', 'pink','seagreen', 'firebrick']
        
        for h, phh in enumerate(plot_hhs):
            self.__abs_k.plot(self.__pt, self.k_eff_reco[:, phh], color = colours[h],
                               label='HH' + str(self.__hhs[phh].hhid))
    
    def __plot_wb(self, n_plot_hhs=4, plot_hhs=None):
        self.__abs_wb.clear()
        self.__abs_wb.set_xlim((-1, RECO_PERIOD))
        #self.__abs_wb.set_ylim((0, 100000))
        self.__abs_wb.set_title('Accumlated HH WB loss')
        self.__abs_wb.set_ylabel('')
        #self.__abs_wb.set_xlabel('time in years')
        colours = ['blue', 'red', 'green', 'gold', 'brown', 'black', 'purple', 'pink','seagreen', 'firebrick']
        for h, phh in enumerate(plot_hhs):
            self.__abs_wb.plot(self.__pt, self.wb_reco[:, phh], color = colours[h],
                                label='HH' + str(self.__hhs[phh].hhid) + ' dec: ' + str(self.__hhs[phh].decile))
            # self.__abs_wb.plot(self.__pt, self.wb_reco_sm[:, phh], color = colours[h],
            #                     label='HH' + str(self.__hhs[phh].hhid) + ' dec: ' + str(self.__hhs[phh].decile))
        self.__abs_wb.legend()

    def __plot_info_gov(self):
        self.__info_gov.clear()
        self.__info_gov.set_xlim((-1, RECO_PERIOD))
        #self.__info_gov.set_ylim((0, 250000))
        self.__info_gov.set_title('National losses')
        self.__info_gov.set_ylabel('Absolute national loss USD')
        self.__info_gov.set_xlabel('time in years')
        self.__info_gov.plot(self.__pt, self.__gov.k_eff_reco, color = 'blue',
                               label='national capital stock damage', alpha = 0.5)
        self.__info_gov.plot(self.__pt, self.__gov.inc_reco, color = 'red',
                               label='national income loss', alpha = 0.5)
        self.__info_gov.plot(self.__pt, self.__gov.inc_sp_reco, color = 'green',
                               label='national loss in social spendings', alpha = 0.5)
        self.__info_gov.plot(self.__pt, self.__gov.cons_reco, color = 'gold',
                               label='loss in national consumption', alpha = 0.5)
        self.__info_gov.legend()
    
    # def write_output_files(self, t_i, result_path):
        
    #     #path = '/home/insauer/projects/WB_model/hhwb/data/output/'

    #     k_eff = pd.DataFrame(data = self.k_eff_reco)
    #     k_eff.to_csv(result_path + 'k_eff.csv')
    #     del k_eff
    #     inc = pd.DataFrame(data = self.inc_reco)
    #     inc.to_csv(result_path + 'inc.csv')
    #     del inc
    #     inc_sp = pd.DataFrame(data = self.inc_sp_reco)
    #     inc_sp.to_csv(result_path + 'inc_sp.csv')
    #     del inc_sp
    #     hh_wb = pd.DataFrame(data = self.wb_reco)
    #     hh_wb.to_csv(result_path + 'hh_wb.csv')
    #     del hh_wb
    #     cons = pd.DataFrame(data = self.cons_reco)
    #     cons.to_csv(result_path + 'cons.csv')
    #     del cons
    #     cons_sav = pd.DataFrame(data = self.cons_reco_sm)
    #     cons_sav.to_csv(result_path + 'cons_sav.csv')
    #     del cons_sav

    #     return

