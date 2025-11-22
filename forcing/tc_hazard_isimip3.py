#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 20:54:25 2025

@author: insauer
"""

import os
import numpy as np
from climada.hazard import Hazard
from climada.util.coordinates import get_land_geometry
from climada.entity import LitPop
from climada.engine import ImpactCalc
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
import pandas as pd

import xarray as xr

TIME_PERIOD = np.arange(2000,2022)

IMPACT_FUNCTION_SET = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()

IF_DICT = IMPACT_FUNCTION_SET.get_countries_per_region()[3]

REF_YEAR=2015

RESOLUTION=120

WINDFIELDS_PATH ='/home/insauer/projects/STP/global_STP_paper/data/windfields/'


OUTPUT = '/home/insauer/projects/STP/global_STP_paper/data/isimip3_forcing/tc_forcing/'



#------------------------------------------------------------------------------
# Future runs

#WINDFIELDS_FUTURE = '/home/insauer/tunnel_foote/home/insauer/tc_isimip2b/'

WINDFIELDS_FUTURE = '/home/insauer/tunnel_foote/p/projects/isimip/isimip/dannellq/TC_ISIMIP3/data/output_H08_300as/'

KNOTS_TO_MperS= 0.514444

WARMING_LEVELS=[1.0, 1.5, 2.0, 3.0, 4.0]

WL_MODEL_LOOKUP= '/home/insauer/projects/STP/global_STP_paper/data/isimip3_forcing/warming_level/cmip6_warming_levels_isimip.csv'

class TC_impact():
    """! Household definition. Computed from the FIES and interacts with the classes
    Government and Shock.
    Attributes:
        @param hhid (int): household id
        @param n_inds (int): number individuals living in the household
        @param weight (float): household weight (summing up over all household weights returns the total
                        population of the administrative unit)
        @param vul (float): household vulnerability (independent of disaster magnitude)
        

    """

    def __init__(self, country='PHL', realizations=[0]):
        """! constructor"""

        #  Attributes set during initialisation

        self.__country = country
        self.__realizations = realizations
        self.__geometry = [get_land_geometry(country_names=[self.__country])]
        self.__region = self.__find_key_by_item(IF_DICT, self.__country)
        self.__if_id = IMPACT_FUNCTION_SET.get_countries_per_region()[1][self.__region]
        
    
    @property
    def country(self):
        return self.__country
    
    @property
    def geometry(self):
        """Returns the land geometry for the selected country."""
        return self.__geometry

    @property
    def region(self):
        """Returns the region the country belongs to (as per IF_DICT mapping)."""
        return self.__region
    
    @property
    def if_id(self):
        """Returns the impact function ID for the region."""
        return self.__if_id
    
    def generate_tc_impact(self):
        
        exp = self.__set_exposure()
        
        
        self.__tc_imp = self.__intersect(exp)

        self.__tc_imp.gdf.to_csv(OUTPUT + f'/h08_TC_affected_120_as{self.__country}.csv')
    
        return
    
    def generate_tc_impact_future(self):
        
        self.__forcing_file_info= self.__gather_files()
        
        
        
        for identifier  in self.__forcing_file_info:
            
            print(identifier)
            
            split=identifier.split('_')
                
            wl_name=int(self.__forcing_file_info[identifier][3]*10)
            
            save_path= f'/{self.__country}/{split[0]}_{split[1]}_{str(wl_name)}'
            
            if not os.path.exists(OUTPUT + save_path):
                os.makedirs(OUTPUT + save_path)
            
            for r in self.__realizations:
                print(r)
                exp = self.__set_exposure()
                
            
                haz= self.__get_future_hazard(self.__forcing_file_info[identifier][0],
                                              self.__forcing_file_info[identifier][1],
                                              self.__forcing_file_info[identifier][2],
                                              r)
                
                if not haz:
                    break
            
                self.__tc_imp = self.__get_future_impact(exp, haz)
                
            
                self.__tc_imp.gdf.to_csv(OUTPUT + save_path + f'/{split[0]}_{split[1]}_{str(wl_name)}_{self.__country}_{r}.csv')

        return
    
    def __find_key_by_item(self, my_dict, item):
        """
        Find and return the key in a dictionary whose associated value list contains a given item.
    
        Parameters
        ----------
        my_dict : dict
            A dictionary where each value is a list of items.
        item : any
            The item to search for within the value lists.
    
        Returns
        -------
        key : any or None
            The key whose value list contains the specified item.
            Returns None if the item is not found in any list.
        
        Example
        -------
        >>> __find_key_by_item({'A': [1, 2], 'B': [3, 4]}, 3)
        'B'
        """
        for key, value_list in my_dict.items():
            # Check if the target item exists in the current list of values
            if item in value_list:
                return key
        # If item is not found in any list, return None
        return None
    
    def __set_exposure(self):
        
        exp = LitPop.from_countries(countries=[self.__country],
                                    fin_mode='pop',res_arcsec=RESOLUTION,
                                    reference_year=REF_YEAR)
        
    
        exp.gdf.rename(columns={"impf_": "impf_" + 'TC'}, inplace=True)
        exp.gdf['impf_TC']=self.__if_id
        exp.gdf['total_population']=exp.gdf['value']
        exp.gdf['value']=1
    
        return exp
    
    def __intersect(self, exp):
        
        for y in TIME_PERIOD:
            
            print(y)
        
            path = WINDFIELDS_PATH + f'h08_obsclim_historical_windlifetimemax_{y}.nc'
            
            bands, storms = self.__storms(path)
            
            hazl = Hazard.from_raster(files_intensity=path, haz_type='TC',
                                      geometry=self.__geometry, band=bands)
            
            imp = ImpactCalc(exp, IMPACT_FUNCTION_SET, hazl).impact(save_mat=True)
            
            indices = np.where(imp.at_event > 0.00001)[0]
            res=imp.imp_mat[indices]
            for e,r in enumerate(np.arange(res.shape[0])):
                exp.gdf[np.array(storms['storm'])[indices][e]]=res.toarray()[e,:]
                
            exp.gdf.to_csv(OUTPUT + f'/h08_TC_affected_120_as{self.__country}.csv')
        
        return exp
    
    def __get_future_impact(self,exp, haz):
        
    
        time=[]
    
        for i,h in enumerate(haz):

            if i ==0:
                
                fin_haz = Hazard.from_xarray_raster(h, "TC", "")
                fin_haz.event_name = [f"{i}-{item}" for i, item in enumerate(fin_haz.event_name)]
                #fin_haz.date = [value + i * 0.001 for i, value in enumerate(fin_haz.date)]
                

                
            else:
                
                temp_hazl= Hazard.from_xarray_raster(h, "TC", "")
                temp_hazl.event_name = [f"{i}-{item}" for i, item in enumerate(temp_hazl.event_name)]
                #temp_hazl.date = [value + i * 0.001 for i, value in enumerate(temp_hazl.date)]
                
                fin_haz.append(temp_hazl)
                
            time.extend(list(h['time'].values.astype(str)))
                 
        
        imp = ImpactCalc(exp, IMPACT_FUNCTION_SET, fin_haz).impact(save_mat=True)
        
        indices = np.where(imp.at_event > 0.00001)[0]
        res=imp.imp_mat[indices]
        
        columns=[s[:10] for s in time]
        for e,r in enumerate(np.arange(res.shape[0])):
            exp.gdf[np.array(columns)[indices][e]]=res.toarray()[e,:]
            
        #exp.gdf.to_csv(OUTPUT + f'/IPSL-CM5A-LR_rcp60_15_{self.__country}.csv')
        
        return exp
        
    
    def __get_future_hazard(self, filenames, start_year, end_year, r):
        
        dsets=[]
        
        for f in filenames:
        
            meta_data= pd.read_csv(WINDFIELDS_FUTURE + f+'/draws.csv')
            
            
            
            meta_data_subset=meta_data.loc[(meta_data['year'].isin(np.arange(start_year, end_year+1))) &
                                (meta_data['real_id']==r)]
            
            events=list(meta_data_subset.index)
            
            
            path = WINDFIELDS_FUTURE + f+'/draws.nc'
            
                    
            # Open the original dataset
            ds = xr.open_dataset(path)
    
            # Define the indices or condition for the events you want
            selected_event_indices = events # example: choose specific events
    
            # Subset the dataset
            ds_subset = ds.isel(event=selected_event_indices)
    
            dset = xr.Dataset(
                 dict(
                     intensity=(
                         ["time", "latitude", "longitude"],
                         ds_subset['wind'].values*KNOTS_TO_MperS,
                     )
                 ),
                 dict(
                     time=ds_subset['time'].values,
                     latitude=ds_subset['lat'].values,
                     longitude=ds_subset['lon'].values,
                ),
            )
            
            dsets.append(dset)
        
        return dsets
    
    def __gather_files(self):
        
        file_info=pd.read_csv(WL_MODEL_LOOKUP)
        
        wl_info=file_info.loc[file_info[' warming_level'].isin(WARMING_LEVELS)]
        
        forcing_file_info={}
        
        for _, cell in wl_info.iterrows():
            
            gcm=cell['model']
            rcp=cell[' exp']
            start_year=cell[' start_year']
            end_year=cell[' end_year']
            wl=cell[' warming_level']
            
            identifier=f'{gcm}_{rcp[1:]}_{wl}_{str(start_year)}_{str(end_year)}'
            
            filenames=[]
            
            if end_year <=2014:
            
                filename=f'{gcm}_20th_WPN_1850_2014_100'
            
                path = WINDFIELDS_FUTURE + filename+'/draws.nc'
            
                if os.path.exists(path):
                    print("✅ Path exists")
                    filenames.append(filename)
                else:
                    print("❌ Path does not exist")
                    
                    continue
                
            elif start_year <= 2014 and end_year > 2014:
                
                filename=f'{gcm}_20th_WPN_1850_2014_100'
                
                path = WINDFIELDS_FUTURE + filename+'/draws.nc'
            
                if os.path.exists(path):
                    print("✅ Path exists")
                    filenames.append(filename)
                else:
                    print("❌ Path does not exist")
                    
                    continue
                
                filename=f'{gcm}_{rcp[1:]}_WPN_2015_2100_100'
                
                path = WINDFIELDS_FUTURE + filename+'/draws.nc'
                
                if os.path.exists(path):
                    print("✅ Path exists")
                    filenames.append(filename)
                else:
                    print("❌ Path does not exist")
                    
                    continue
                
            else:
                
                filename=f'{gcm}_{rcp[1:]}_WPN_2015_2100_100'
                
                path = WINDFIELDS_FUTURE + filename+'/draws.nc'
                
                if os.path.exists(path):
                    print("✅ Path exists")
                    filenames.append(filename)
                else:
                    print("❌ Path does not exist ")
                    
                    if start_year >= 2061:
                        
                        filename=f'{gcm}_{rcp[1:]}_WPN_2061_2100_100'
                        
                        path = WINDFIELDS_FUTURE + filename+'/draws.nc'
                        
                    else:
                        print("❌ Years do not exist ")
                        
                        continue
                    
                    if os.path.exists(path):
                        print("✅ Path exists")
                        filenames.append(filename)
                    else:
                        print("❌ Path does not exist ")
                        continue
                    
            
            split=identifier.split('_')
            
            wl_name=int(wl*10)
        
            save_path= f'{self.__country}/{split[0]}_{split[1]}_{str(wl_name)}'
            
            if os.path.exists(OUTPUT +save_path):
                print("✅ Result exists")
                continue
            else:
                print("❌ Result does not exist")
                
            forcing_file_info[identifier] = [filenames,start_year, end_year, wl]
        
        return forcing_file_info
    
    
    
    def __storms(self, path):
    
        ds = xr.open_dataset(path)
        n_bands=ds.variables['windlifetimemax'].shape[0]
        bands=list(np.arange(1, n_bands+1))
        storms=ds['storm']
        ds.close()
        
        return bands, storms
    

realizations=np.arange(10,25)


haz=TC_impact('PHL', realizations)
haz.generate_tc_impact_future()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    