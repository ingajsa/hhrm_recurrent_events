#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:19:13 2020

@author: insauer
"""
import os
import zipfile
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
        """
        Main method that starts and controls the dynamic model.
        """
        print("Life started new")
 
        # --- Setup temporal resolution ---
        self._setup_time(reco_period)

        # --- Display initial government info ---
        print("Total expenditure on social programs:", self.__gov.sp_cost)
        print("Total national capital stock:", self.__gov.K)

        # --- Setup CSV writers and initial result files ---
        writers = self._setup_results(result_path)

        n_shock = 0

        # --- Main simulation loop ---
        for t_i in self.__dt_life:
            print(t_i)
            print("Time step dt:", self.__dt_reco)

            if t_i in self.__shock.time_stamps:
                print("shock start")
                self.__hhs = self.__shock.shock(n_shock, self.__gov, self.__hhs, self.__dt_reco, cores)
                n_shock += 1
            else:
                self._recover_households(cores)

            # Collect government info
            self.__gov.collect_hh_info(t_i, self.__hhs)

            # Update and write household & government records
            self._write_records(t_i, writers)

        # Close all CSV files
        self._close_writers(writers, result_path)

    # ---------------------- Helper Functions ---------------------- #

    def _setup_time(self, reco_period):
        """Set up model time steps and resolution."""
        pt = np.linspace(0, reco_period, reco_period * DT_STEP + 1)
        self.__dt_reco = np.diff(pt)[0]
        self.__pt = pt[::4]
        self.__dt_life = np.arange(0, reco_period * DT_STEP + 1)

    def _setup_results(self, result_path):
        """
        Create CSV files and return a dictionary of CSV writers.
        """
        colnames = np.arange(len(self.__hhs)).astype(str)
        files = {}
        writers = {}

        # Household-level files
        hh_files = [
            "keff", "kpub", "kpriv", "ipub", "ipriv",
            "inc", "inc_sp", "cons", "cons_priv",
            "cons_sm", "cons_priv_sm", "wb", "wb_sm", "sav"
        ]
        for fname in hh_files:
            path = result_path + fname + ".csv"
            f = open(path, "w", newline="")
            writer = csv.writer(f)
            files[fname] = f
            writers[fname] = writer

        # Government-level file
        gov_path = result_path + "gov.csv"
        f_gov = open(gov_path, "w", newline="")
        writer_gov = csv.writer(f_gov)
        files["gov"] = f_gov
        writers["gov"] = writer_gov

        return {"files": files, "writers": writers}

    def _recover_households(self, cores):
        """Update households during recovery period using multiprocessing."""
        self.__gov.update_reco(self.__hhs)
        p = mp.Pool(cores)
        prod_x = partial(ClimateLife.update_reco, gov=self.__gov)
        self.__hhs = p.map(prod_x, self.__hhs)
        p.close()
        p.join()

    def _write_records(self, t_i, writers_dict):
        """
        Update household and government records for the current time step
        and write them to CSV.
        """
        # Update household records
        kpub, kpriv, ipub, ipriv, inc, inc_sp, cons, cons_priv, \
        cons_sm, cons_priv_sm, wb, wb_sm, sav = self.__update_records(t_i)

        # Update government results
        gov_res = [
            self.__gov.L_t, self.__gov.d_k_priv_t, self.__gov.L_pub_t,
            self.__gov.d_inc_t, self.__gov.d_inc_sp_t,
            self.__gov.d_con_priv_t, self.__gov.d_con_eff_t,
            self.__gov.d_con_eff_sm, self.__gov.d_con_priv_sm,
            self.__gov.d_wb_t, self.__gov.d_wb_sm, self.__gov.pub_debt
        ]

        writers = writers_dict["writers"]
        # Write household data
        writers["keff"].writerow(list(kpub))  # replaced doubled kpub
        writers["kpub"].writerow(list(kpub))
        writers["kpriv"].writerow(list(kpriv))
        writers["ipub"].writerow(list(ipub))
        writers["ipriv"].writerow(list(ipriv))
        writers["inc"].writerow(list(inc))
        writers["inc_sp"].writerow(list(inc_sp))
        writers["cons"].writerow(list(cons))
        writers["cons_priv"].writerow(list(cons_priv))
        writers["cons_sm"].writerow(list(cons_sm))
        writers["cons_priv_sm"].writerow(list(cons_priv_sm))
        writers["wb"].writerow(list(wb))
        writers["wb_sm"].writerow(list(wb_sm))
        writers["sav"].writerow(list(sav))
        # Write government data
        writers["gov"].writerow(gov_res)
        
    def _close_writers(self, writers_dict, result_path=None, zip_name="results.zip"):
        """Close all CSV files and optionally compress them into a ZIP file."""
        # Close all open files
        for f in writers_dict["files"].values():
            f.close()
    
        # Compress CSV files into a zip archive if result_path is provided
        if result_path:
            zip_path = os.path.join(result_path, zip_name)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for fname in writers_dict["files"]:
                    file_path = os.path.join(result_path, fname + ".csv")
                    if os.path.exists(file_path):
                        zipf.write(file_path, arcname=fname + ".csv")
                        os.remove(file_path)
            print(f"All results compressed into {zip_path}")
        
    


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

