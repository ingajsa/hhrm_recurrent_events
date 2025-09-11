#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:41:42 2020

@author: insauer
"""
import os
import numpy as np
import pandas as pd
import re
import multiprocessing as mp
from functools import partial
from hhwb.util.constants import  DT_STEP, RECO_PERIOD
from datetime import datetime, date
from zipfile import ZipFile




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

        self.__time_stamps = []
        self.__event_names = []
        self.__aff_ids = np.array([[]])

        self.__L = 0

    @property
    def aff_ids(self):
        return self.__aff_ids

    @property
    def time_stamps(self):
        return self.__time_stamps

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
        """
        Apply a shock to a household (hh) depending on whether it is in the affected set.
    
        Parameters
        ----------
        hh : object
            Household object, must have `hhid` attribute and `.shock(...)` method.
        L : float
            Some labor-related input (private labor supply, etc.).
        K : float
            Capital-related input.
        L_pub : float
            Public labor-related input.
        dt_reco : float
            Recovery time step (duration over which shock is applied).
        affected_hhs : set or list
            Collection of household IDs that are affected by the shock.
        vuls : dict or list
            Vulnerability values indexed by household ID (or castable to int).
        
        Returns
        -------
        hh : object
            The updated household object after shock is applied.
        """
    
        # Check if this household is in the set of affected households
        if hh.hhid in affected_hhs:
            # Apply shock with "affected" flag = True
            hh.shock(
                aff_flag=True,
                L_pub=L_pub,
                L=L,
                K=K,
                dt=dt_reco,
                vul=vuls[int(hh.hhid)]  # vulnerability corresponding to this household
            )
        else:
            # Apply shock with "affected" flag = False (unaffected households)
            hh.shock(
                aff_flag=False,
                L_pub=L_pub,
                L=L,
                K=K,
                dt=dt_reco,
                vul=vuls[int(hh.hhid)]
            )
    
        # Return the updated household object
        return hh
    

    def contains_date_format(self, series):
        """
        Check which entries in a pandas Series match a YYYY-MM-DD date format.
    
        Parameters
        ----------
        series : pandas.Series
            The series to check (can be any dtype, will be cast to string).
    
        Returns
        -------
        pandas.Series
            Subset of the original series containing only the entries
            that match the date pattern.
        """
    
        # Compile a regular expression that matches dates in the form YYYY-MM-DD
        # Example: "2023-09-05"
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
    
        # Convert the series to string, check which values match the regex,
        # and return only those matching entries
        return series[series.astype(str).str.contains(date_pattern)]
    


    def read_shock(self, work_path, path, event_identifier, run):
        """
        Read already preprocessed shock data (households × events).
        This function should only be used if household vulnerability is given and not intensity related.
    
        Parameters
        ----------
        work_path : str
            Base path to working directory.
        path : str
            Path (relative to work_path) to the zip file that contains shock data.
        event_identifier : str
            String pattern used to identify event columns in the dataset.
        run : any
            Run identifier (currently unused inside the function).
    
        Side Effects
        ------------
        - Extracts and reads a shocks CSV file from a zipped archive.
        - Aggregates events that fall within the same model time step (weeks).
        - Saves aggregated events to `shocks_aggregated.csv`.
        - Stores intermediate results in:
            self.__aff_ids (household × week matrix)
            self.__time_stamps (sorted list of week numbers)
    
        Returns
        -------
        None
        """
    
        # Path where the CSV will be temporarily extracted
        temp_dir = os.path.join(work_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        csv_filename = "test_shocks.csv"
        csv_path = os.path.join(temp_dir, csv_filename)
    
        # Extract the CSV from the zip file
        with ZipFile(os.path.join(work_path, path), 'r') as zip_ref:
            zip_ref.extract(csv_filename, path=temp_dir)
    
        # ✅ Read the extracted CSV (bug fixed)
        shock_data = pd.read_csv(csv_path)
    
        # Reference start date of the model
        start_date = date(2008, 1, 1)
    
        # Select only the event-related columns
        event_names = [col for col in shock_data.columns if event_identifier in col]
    
        # Subset DataFrame to event columns
        event_data = shock_data[event_names]
    
        # Convert event column names to datetime.date
        dates_list = [datetime.strptime(event, '%Y-%m-%d').date() for event in event_names]
    
        # Compute time steps in weeks (approx., using 28 days per step)
        month = [np.round((event_date - start_date).days / 28) for event_date in dates_list]
    
        # Initialize household × time-step matrix
        self.__aff_ids = np.zeros((event_data.shape[0], len(set(month))))
    
        # Sorted list of unique time steps
        week_nums = sorted(set(month))
    
        # Aggregate shocks by time step
        for m, week in enumerate(week_nums):
            locat = np.where(np.array(month) == week)
            self.__aff_ids[:, m] = event_data.iloc[:, locat[0]].sum(axis=1).clip(upper=1)
    
        # Save the time steps
        self.__time_stamps = week_nums
    
        # Build DataFrame with households × weeks
        shock_df = pd.DataFrame(
            data=self.__aff_ids,
            columns=np.array(week_nums).astype(int).astype(str)
        )
    
        # Save aggregated shocks
        shock_df.to_csv('shocks_aggregated.csv', index=False)
    
        return
    
    def __get_months(self, start_date, strings):
        """
        Convert strings containing dates (YYYY-MM-DD) into model time steps
        measured in 28-day "months" since a fixed reference date.
    
        Parameters
        ----------
        start_date : any
            Currently unused parameter (can probably be removed).
        strings : list-like of str
            Strings that may contain date substrings.
    
        Returns
        -------
        list of int
            List of month indices (number of 28-day periods since 2000-01-01).
        """
    
        # Regular expression to match dates in YYYY-MM-DD format
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
    
        # Fixed reference date for calculating offsets
        reference_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
    
        # List to collect converted time steps
        date_differences = []
    
        # Iterate through all input strings
        for s in strings:
            # Find all date substrings in the current string
            dates = date_pattern.findall(s)
    
            # Convert each found date to a 28-day "month" index
            for date_str in dates:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # parse string into datetime
                days_since_reference = (date_obj - reference_date).days  # difference in days
                # Divide by 28 to approximate "months" and round to nearest int
                date_differences.append(int(np.round(days_since_reference / 28)))
    
        return date_differences
    
    def read_vul_shock(self, path, output_path, file, start_year):
        """
        Read vulnerability-related shock data (households × events).
    
        Parameters
        ----------
        path : str
            Path to the zip file containing shock data.
        output_path : str
            Directory where the aggregated shocks CSV will be saved.
        file : str
            Name of the CSV file inside the zip archive.
        start_year : int
            Reference year for converting event dates to model time steps.
    
        Side Effects
        ------------
        - Extracts and reads a shocks CSV file from a zipped archive.
        - Aggregates events that fall within the same model time step (months).
        - Saves aggregated shocks to 'shocks_aggregated.csv'.
        - Updates instance variables:
            self.__aff_ids     → household × month matrix of aggregated shocks
            self.__time_stamps → sorted list of aggregated event indices
    
        Returns
        -------
        None
        """
    
        # --- Extract CSV from the zip file ---
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        csv_path = os.path.join(temp_dir, file)
    
        with ZipFile(path, "r") as zip_ref:
            zip_ref.extract(file, path=temp_dir)
    
        # ✅ Read the extracted CSV (bug fixed: not reading the zip itself)
        shock_data = pd.read_csv(csv_path)
    
        # --- Identify event columns (with YYYY-MM-DD format) ---
        event_names = self.contains_date_format(shock_data.columns)
    
        # --- Select only those event columns ---
        selected_shocks = shock_data[event_names]
    
        # Drop columns where the sum is zero (no shocks in that event)
        selected_shocks = selected_shocks.loc[:, selected_shocks.sum() > 0]
    
        # Update event names after filtering
        event_names = selected_shocks.columns
    
        # --- Convert dates to model time steps (months) ---
        months = self.__get_months(start_year, event_names)
    
        # Initialize household × month matrix
        unique_months = sorted(set(months))
        self.__aff_ids = np.zeros((selected_shocks.shape[0], len(unique_months)))
    
        # --- Aggregate shocks by month ---
        for m, month in enumerate(unique_months):
            locat = np.where(np.array(months) == month)
            # Use max across shocks in the same month (if one event hits, household is shocked)
            self.__aff_ids[:, m] = selected_shocks.iloc[:, locat[0]].max(axis=1)
    
        # Save month indices
        self.__time_stamps = unique_months
    
        # --- Build household × month DataFrame ---
        shock_df = pd.DataFrame(
            data=self.__aff_ids,
            columns=np.array(unique_months).astype(int).astype(str)
        )
    
        # Save aggregated shocks
        os.makedirs(output_path, exist_ok=True)
        shock_df.to_csv(os.path.join(output_path, "shocks_aggregated.csv"), index=False)
    
        return
    

    
    def shock(self, event_index, gov, hhs, dt_reco, cores):
        """
        Apply a shock to affected households and update government aggregates.
    
        Parameters
        ----------
        event_index : int
            Index of the current shock event in self.__aff_ids.
        gov : Gov
            Government object (contains aggregate labor/capital info).
        hhs : list[HH]
            List of household objects 
        dt_reco : float
            Recovery time step (duration of the shock).
        cores : int
            Number of CPU cores to use for parallel processing.
    
        Returns
        -------
        list[HH]
            Updated list of household objects after applying the shock.
        """
    
        # Identify affected households for this event
        # Condition: value > ~0 (tiny epsilon avoids floating-point issues)
        affected_hhs = np.where(self.__aff_ids[:, event_index] > 1e-11)[0]
    
        # Vulnerabilities for all households in this event (vector)
        vuls = self.__aff_ids[:, event_index]
    
        # Initialize labor aggregates with government baseline values
        L_t = gov.L_t
        L_pub_t = gov.L_pub_t
        print("current")
    
        # Loop over all households and accumulate government-level effects
        for h_ind, hh in enumerate(hhs):
            if hh.hhid in affected_hhs:  # household is affected by the shock
                # Update public labor loss
                L_pub_t += (
                    vuls[int(hh.hhid)]
                    * (hh.k_pub_0 - hh.d_k_pub_t)  # remaining public capital
                    * hh.weight                   # household weight
                )
                # Update total labor loss (public + private)
                L_t += (
                    vuls[int(hh.hhid)]
                    * ((hh.k_pub_0 - hh.d_k_pub_t) + (hh.k_priv_0 - hh.d_k_priv_t))
                    * hh.weight
                )
    
        # Apply the aggregated shock to the government
        gov.shock(aff_flag=True, L=L_t, L_pub=L_pub_t, dt=dt_reco)
    
        # Prepare multiprocessing pool to distribute household-level shocks
        p = mp.Pool(cores)
    
        # Build partial function to apply shocks to each household
        prod_x = partial(
            Shock.apply_shock,
            L=L_t,
            K=gov.K,
            L_pub=L_pub_t,
            dt_reco=dt_reco,
            affected_hhs=affected_hhs,
            vuls=vuls,
        )
    
        # Parallelize: apply shocks to all households
        hhs = p.map(prod_x, hhs)
    
        # Clean up multiprocessing pool
        p.close()
        p.join()
    
        # Return updated households
        return hhs




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
