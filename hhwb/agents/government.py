#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:41:02 2020

@author: insauer
"""

import numpy as np
import pandas as pd
from hhwb.util.constants import  DT_STEP, TEMP_RES, DT, OPT_DYN, RHO

#from misc.cluster_sim import PI

AGENT_TYPE = 'GV'

pams= pd.read_csv('params.csv')

PI = pams['PI'].values[0]
ETA=pams['ETA'].values[0]
SUBS_SAV_RATE=pams['SUBS_SAV_RATE'].values[0]/13.
T_RNG=pams['T_RNG'].values[0]
K_PUB=pams['K_PUB'].values[0]
COUNTRY=pams['COUNTRY'].values[0]
OUTPUT_DATA_PATH=pams['OUTPUT_DATA_PATH'].values[0]


class Government():

    """Government definition. Computed from the FIES and interacts with the classes
    Households and Shock.

    Attributes:
        ***Attributes regarding the predisaster situation***
        tax_rate (float): flat income tax
        sp_cost (float): total spendings on social transfers
        K (float): Total national capital stock

        ***Attributes related to disaster recovery***
        L_0 (list): total national damage experienced at each shock
        vul (list): conceptional vulnerability at each shock as ratio L/K
        lmbda (float): optimal recovery rate (not used under private recovery)
        tau (float): time at which 95% of capital stock is recovered
        L_t (float): loss at timestep t after disaster
        """

    def __init__(self):
        """Empty Definition"""
        
        self.__damage = [0.]
        self.__damage_pub = [0.]
        
        self.__cnt_ts = 0.
        self.__tax_rate = 0.
        self.__sp_cost = 0.
        self.__K = 0.
        self.t = 0.

        self.__L_0 = 0
        self.__vul = []

        self.__lmbda = [0.]

        self.__L_t = 0.
        
        self.__k_eff_0 = None
        
        self.__d_k_eff_t = 0.
        self.__d_k_pub_t = 0.
        self.__d_k_priv_t = 0.
        
        
        # share of income_0 coming from social transfers

        self.__d_inc_t = 0.
        self.__d_inc_sp_t = 0.
        self.__d_con_eff_t = 0.
        self.__d_con_eff_sm = 0.
        self.__d_con_priv_t = 0.
        self.__d_con_priv_sm = 0.
        
        self.__d_wb_t = 0.
        self.__d_wb_sm = 0.
        
        self.__dt = 0.
        self.__c_shock = 0
        
        self.__aff = []
        
        self.__pub_debt = 0.



    @property
    def tax_rate(self):
        return self.__tax_rate
    
    @property
    def dt(self):
        return self.__dt

    @property
    def sp_cost(self):
        return self.__sp_cost
    
    @property
    def lmbda(self):
        return self.__lmbda[-1]

    @property
    def K(self):
        return self.__K
    
    @property
    def K_priv(self):
        return self.__K_priv
    
    @property
    def K_pub(self):
        return self.__K_pub
    
    @property
    def L_t(self):
        return self.__d_k_eff_t
    
    @property
    def aff_k_pub_t(self):
        return self.__aff_k_pub_t
    
    @property
    def L_pub_t(self):
        return self.__d_k_pub_t
    
    @property
    def d_k_priv_t(self):
        return self.__d_k_priv_t
    
    @property
    def d_inc_t(self):
        return self.__d_inc_t
    
    @property
    def d_inc_sp_t(self):
        return self.__d_inc_sp_t
    
    @property
    def d_con_eff_t(self):
        return self.__d_con_eff_t
    
    @property
    def d_con_priv_sm(self):
        return self.__d_con_priv_sm
    
    @property
    def d_con_priv_t(self):
        return self.__d_con_priv_t
    
    @property
    def d_con_eff_sm(self):
        return self.__d_con_eff_sm
    
    @property
    def d_wb_t(self):
        return self.__d_wb_t
    
    @property
    def d_wb_sm(self):
        return self.__d_wb_sm
    
    @property
    def pub_debt(self):
        return self.__pub_debt
    


    def set_tax_rate(self, reg_hh):
        """
        Sets the optimal tax rate basing on the total enrollment for social transfers and
        extracts capital stock from each hh to get total national capital stock
            Parameters:
                reg_hh (list): all registered households
        """

        tot_inc = 0.
        tot_sp = 0.
        k_pub=0.
        k_priv=0.
        # get total income and total social transfers from all households
        for hh in reg_hh:
            tot_inc += hh.weight*hh.income_0
            tot_sp += hh.weight*hh.income_sp

        # set governments total expenditure on social transfers and required tax rate
        self.__sp_cost = tot_sp
        self.__tax_rate = tot_sp/tot_inc
        # set tax rate for all households and get capital stock
        for hh in reg_hh:
            hh.set_tax_rate(self.__tax_rate)
            hh.init_life()
            self.__K += hh.weight*hh.k_eff_0
            k_pub+=hh.weight*hh.k_pub_0
            k_priv+=hh.weight*hh.k_priv_0
        
        self.__K_pub=K_PUB * self.__K
        self.__K_priv=(1-K_PUB) * self.__K
        
        print(self.__K_pub/(self.__K_pub+self.__K_priv))
        
        return
    
    def collect_hh_info(self, t_i=0., hh_reg=None):
        """
        Parameters
        ----------

        t_i : TYPE
            DESCRIPTION.
        hh_reg : list 
            List with all households.

        Returns
        -------
        None.
    
        """

        self.__collect_all(hh_reg)

        return
    
    def update_reco(self, hh_reg= None):
        
        self.__update_k_pub()

        return
        
    
    def shock(self, aff_flag=False,
              L=None, K=None, L_pub=None,dt=None):
        """This function causes the agent to be shocked. The recovery track is set up and the
           post disaster state is generated for all indicators.
        Parameters
        ----------
        aff_flag : bool, optional
            Bool indicating whether the household is affected by the current shock.
            The default is False.
        L : float, optional
            Total national damage.
        K : float, optional
            Total national capital stock.
        """
        self.__dt = dt
        self.t = 0.
        #print(self._c_shock)
        #print(aff_flag)
        if aff_flag:
            self.__c_shock += 1
        self.__aff.append(aff_flag)
        #  set the affected state
        self.__set_shock_state(L, L_pub, K, aff_flag)

        return


    # def update(self, reg_hh):
    #     self.__L_t = 0.
    #     self.__d_cons = 0.
    #     for hh in reg_hh:
    #         self.__L_t += hh.d_k_eff_t
    #     return

    def __collect_all(self, reg_hh):
        
        """gather all government relevant variables"""
        #self.__d_k_pub_t = 0.
        self.__d_con_eff_t = 0.
        self.__d_k_priv_t = 0.
        self.__d_con_priv_t = 0.
        self.__d_inc_t = 0.
        self.__d_inc_sp_t = 0.
        self.__d_wb_t = 0.
        self.__d_con_eff_sm = 0.
        self.__d_con_priv_sm = 0.
        self.__d_wb_sm = 0.
        
        for hh in reg_hh:
            
            self.__d_k_priv_t += hh.weight* hh.d_k_priv_t

            self.__d_inc_t += hh.weight*hh.d_inc_t
            self.__d_inc_sp_t += hh.weight*hh.d_inc_sp_t
            self.__d_wb_t += hh.weight*hh.d_wb_t
            self.__d_wb_sm += hh.weight*hh.d_wb_sm
            self.__d_con_eff_t += hh.weight*hh.d_con_eff_t
            self.__d_con_priv_t += hh.weight*hh.d_con_priv_t
            self.__d_con_eff_sm += hh.weight*hh.d_con_eff_sm
            self.__d_con_priv_sm += hh.weight*hh.d_con_priv_sm
            
        self.__d_k_eff_t = self.__d_k_priv_t + self.__d_k_pub_t
        
        return
    
    def __set_shock_state(self, L, L_pub, K, aff_flag):
        """This function calculates the initial damage and the initial loss in income and
           consumption.
        Parameters
        ----------
        L : float
            Total national damage. The default is 0.
        K : float
            Total national capital stock. The default is 0.
        aff_flag: bool
            indicates whether agent is shocked.
        """
        #self._dt = dt

        self.__vul = L/self.__K
        #self._optimize_reco()
        if aff_flag:
            self.__d_k_eff_t = L
            self.__d_k_pub_t = L_pub
            self.__d_k_priv_t = L - L_pub
            #self.__add_pds_expenditure()
            
        else:
            self.__d_k_eff_t = 0
            
        opt_vul = self.__d_k_pub_t/self.__K_pub
        
        if self.__K_pub==0:
            
            opt_vul=0.0
        
        self.__damage.append(self.__d_k_eff_t)
        self.__damage_pub.append(self.__d_k_pub_t)
        self.__d_inc_sp_t = (L/self.__K) * self.__sp_cost
        self.__d_inc_t = PI * L + self.__d_inc_sp_t
        self.__d_con_eff_t = np.nan


        self.__get_reco_from_lookup(vul=opt_vul)
        


        return
    
    # def __add_pds_expenditure(self):
        
    #     if PDS=='k_priv':
    #         self.__pub_debt += self.__d_k_priv_t
            
    #     return
    
    def __optimize_reco(self, vul=0.3):
        """
        This is the core optimization function, that numerically optimizes
        the optimal reconstruction rate lmbda of the household 
        TODO (- eventually implement a static jit version
              - this must be done multicore)
        """
        if vul == 0:
            return
        
        if vul == np.nan:
            
            print('invalid k_eff')
            print(self.__hhid)
            vul = 0.3

        last_integ = 0.
        last_lambda = 0.

        lmbda = 0.0

        #print('vul='+str(vul))

        c=0

        while c<1000:

            integ = 0.0
            
            if self.__cnt_ts == 0:
            
                for dt in np.linspace(0, T_RNG, DT_STEP*T_RNG):
                    h=PI - (PI + lmbda) * vul * np.e**(-lmbda * dt)
                    h=abs(h)
                    integ += np.e**(-dt * (RHO + lmbda)) * ((PI + lmbda) * dt - 1)*h**(-ETA)
                    if np.isnan(integ)==True:

                        break
                    
                if np.isnan(integ)==True:
                    break
                    
            else:
                
                for dt in np.linspace(0, T_RNG, DT_STEP*T_RNG)[:-self.__cnt_ts]:
                    integ +=np.e**(-dt * (RHO + lmbda)) * ((PI + lmbda) * dt - 1) * (PI - (PI + lmbda) * vul * np.e**(-lmbda * dt))**(-ETA)
            
            # if self.__cnt_ts == 0:
            
            #     for dt in np.linspace(0, T_RNG, DT_STEP*T_RNG):
            #         integ += np.e**(-dt * lmbda) * ((PI + lmbda) * dt - 1) 
            # else:
                
            #     for dt in np.linspace(0, T_RNG, DT_STEP*T_RNG)[:-self.__cnt_ts]:
            #         integ += np.e**(-dt * lmbda) * ((PI + lmbda) * dt - 1) 
            #print(integ)
            
            if last_integ and ((last_integ < 0 and integ > 0) or
                                (last_integ > 0 and integ < 0)):
                # print('\n Found the Minimum!\n lambda = ', last_lambda,
                #       '--> integ = ', last_integ)
                # print('lambda = ', lmbda, '--> integ = ', integ, '\n')

                out = (lmbda+last_lambda)/2

                self.__lmbda.append(out)

                return

            last_integ = integ
            if last_integ is None:
                assert(False)
                
            last_lambda = lmbda
                
            if (lmbda<0.05) & (vul > 0.74):
                lmbda+=0.002
            else:
                lmbda+=0.05
                

            c+=1
        
        self.__lmbda.append(0.1)
        print('optimization failed')
        print(vul)
    
    def __get_reco_fee(self, t1=None, t2=None):
        """
        Calculates the recovery spending for one timestep under exponential recovery.
        Returns
        -------
        TYPE
            recovery fee.
        """
        if not t1:
            t1 = self.t

        if not t2:
            t2 = self.t + self.dt

        dam_0 = self.__damage_pub[self.__c_shock] * np.e**(-t1*self.__lmbda[self.__c_shock])
        dam_1 = self.__damage_pub[self.__c_shock] * np.e**(-t2*self.__lmbda[self.__c_shock])

        return dam_0-dam_1
    
    def __update_k_pub(self):
        """
        Updates capital stock damage.

        """
        
        opt_vul = self.__d_k_pub_t/self.__K_pub
        
        self.__get_reco_from_lookup(opt_vul)
        
        self.__d_k_pub_t -= self.__get_reco_fee()
        self.__pub_debt += self.__get_reco_fee()
        
        return
    
    def __get_reco_from_lookup(self, vul):
        
        lambdas= pd.read_csv('/home/insauer/projects/STP/global_STP_paper/data/results/first_try/'+'lambdas_{}.csv'.format(COUNTRY))
        
        if vul < 0.001:
            vul=0.001
        
        try:

            self.__lmbda.append(lambdas.loc[np.round(lambdas['vul'],3)==np.round(vul,3),'lmbda'].values[0])
            
        except IndexError:
            
            print(np.round(vul,3))
        
        return 

    # def _update_income_sp(self):
    #     self.__d_inc_sp_t = (self.__d_k_eff_t/self.__K) * self.__sp_cost
    #     return

    # def _update_income(self):
    #     self.__d_inc_t = PI * self.__d_k_eff_t
    #     return

    # def _update_consum(self, t):
    #     self._d_con_t = self.__d_inc_t + self._lmbda * self._get_reco_fee(t=t)
    #     return

    # def _update_k_eff(self, t):

    #     self.__d_k_eff_t = self._get_reco_fee(t=t)
    #     return
