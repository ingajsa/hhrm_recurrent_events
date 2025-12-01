#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:26:23 2025

@author: insauer
"""

import pandas as pd
import numpy as np
import os
import re
import zipfile


IMPACT_METRICS= [
    "ipub", "ipriv",
    "inc", "inc_sp", "cons", "cons_priv",
    "cons_sm", "cons_priv_sm", "wb", "wb_sm"]

input_folder='/home/insauer/tunnel_foote/p/projects/ebm/inga/tipESM/hhrm_output/'

hh_surveys='/home/insauer/projects/STP/global_STP_paper/data/isimip3_forcing/final_PHL/'

results= '/home/insauer/projects/hhrm/projects/TipESM/data/aggregated_results/cluster_runs/'

STEPS_PER_YEAR=13

RECOVERY_TIME=3


def check_tipping(data, shock_path, recovery_time):
    
    
    income= data['income'].values/13
    
    with zipfile.ZipFile(path, "r") as z:
        with z.open('cons.csv') as f:
            cons = pd.read_csv(f)

    
    cons.columns=np.arange(cons.shape[1])
    
    
    left_income=-cons.sub(income)
    
    subsistence = left_income.lt(data['subsistence_line'].values/13).astype(int)
    
    data['subsistence_0']=subsistence.iloc[0,:]
    data['subsistence_fin']=subsistence.iloc[-1,:]
    
    with zipfile.ZipFile(path, "r") as z:
        with z.open(shock_path) as f:
            shocks = pd.read_csv(f)

    
    data['last_shock'] = (shocks.gt(0)                       # boolean mask (True if > 0)
            .iloc[:, ::-1]               # reverse columns (so last >0 becomes first True)
            .idxmax(axis=1))             # get column name of first True per row
    
    data["tipped"] = [subsistence.iloc[row_val+STEPS_PER_YEAR*recovery_time, i] if row_val < len(subsistence) else np.nan
                      for i, row_val in enumerate(data["last_shock"].astype(int))]
    data.loc[data['n_events']>0,'affected']=1
    data.loc[data['n_events']==0,'affected']=0
    
    
    return data

def aggregate_results(data):
    
    tipped_decile = (
    data.pivot_table(
        index="decile",
        columns="tipped",
        values="weight",
        aggfunc="sum",
        fill_value=0
    )
    .rename(columns={0: "not_tipped", 1: "tipped"})
    .reset_index())
    
    tipped_n_events = (
    data.groupby(["decile", "n_events", "tipped"])["weight"]
    .sum()
    .unstack(fill_value=0)
    .rename(columns={0: "not_tipped", 1: "tipped"})
    .reset_index()
    )
    
    tipped = (
    data.groupby("tipped")["weight"]
    .sum()
    .rename({0: "not_tipped", 1: "tipped"})
    .to_frame().T
    .reset_index(drop=True)
    )
    
    cols_to_sum = [f"tot_{metric}" for metric in IMPACT_METRICS]

    # Copy data
    weighted_data = data.copy()
    
    # Multiply each metric by weight (needed for weighted averages)
    for col in cols_to_sum:
        weighted_data[col] = weighted_data[col] * weighted_data["weight"]
    
    # --- 1️⃣ Weighted averages by decile ---
    grouped_decile = weighted_data.groupby("decile")
    weighted_avg_decile = (
        grouped_decile[cols_to_sum].sum().div(grouped_decile["weight"].sum(), axis=0)
        .reset_index()
    )
    
    # --- 2️⃣ Overall sums (ignore decile) ---
    overall_sum = weighted_data[cols_to_sum].sum().to_frame().T
    overall_sum["decile"] = "Total"
    overall_sum = overall_sum[["decile"] + cols_to_sum]
    
    # --- 3️⃣ Weighted averages by event and decile ---
    group_col = "n_events"
    grouped_event_decile = weighted_data.groupby([group_col, "decile"])
    weighted_avg_event_decile = (
        grouped_event_decile[cols_to_sum].sum()
        .div(grouped_event_decile["weight"].sum(), axis=0)
        .reset_index()
    )


    return tipped_decile, tipped_n_events, tipped, weighted_avg_decile, overall_sum, weighted_avg_event_decile
    
def aggregate_results_affected(data):
    
    data=data.loc[data['affected']==1]
    
    tipped_decile = (
    data.pivot_table(
        index="decile",
        columns="tipped",
        values="weight",
        aggfunc="sum",
        fill_value=0
    )
    .rename(columns={0: "not_tipped", 1: "tipped"})
    .reset_index())
    
    tipped_n_events = (
    data.groupby(["decile", "n_events", "tipped"])["weight"]
    .sum()
    .unstack(fill_value=0)
    .rename(columns={0: "not_tipped", 1: "tipped"})
    .reset_index()
    )
    
    tipped = (
    data.groupby("tipped")["weight"]
    .sum()
    .rename({0: "not_tipped", 1: "tipped"})
    .to_frame().T
    .reset_index(drop=True)
    )
    
    cols_to_sum = [f"tot_{metric}" for metric in IMPACT_METRICS]

    # Copy data
    weighted_data = data.copy()
    
    # Multiply each metric by weight (needed for weighted averages)
    for col in cols_to_sum:
        weighted_data[col] = weighted_data[col] * weighted_data["weight"]
    
    # --- 1️⃣ Weighted averages by decile ---
    grouped_decile = weighted_data.groupby("decile")
    weighted_avg_decile = (
        grouped_decile[cols_to_sum].sum().div(grouped_decile["weight"].sum(), axis=0)
        .reset_index()
    )
    
    # --- 2️⃣ Overall sums (ignore decile) ---
    overall_sum = weighted_data[cols_to_sum].sum().to_frame().T
    overall_sum["decile"] = "Total"
    overall_sum = overall_sum[["decile"] + cols_to_sum]
    
    # --- 3️⃣ Weighted averages by event and decile ---
    group_col = "n_events"
    grouped_event_decile = weighted_data.groupby([group_col, "decile"])
    weighted_avg_event_decile = (
        grouped_event_decile[cols_to_sum].sum()
        .div(grouped_event_decile["weight"].sum(), axis=0)
        .reset_index()
    )


    return tipped_decile, tipped_n_events, tipped, weighted_avg_decile, overall_sum, weighted_avg_event_decile
    



def calc_total_impacts(survey_data,path):
    
    with zipfile.ZipFile(path, "r") as z:
        
    
        for metric in IMPACT_METRICS:
            
            # Extract the CSV file to a temporary directory
            with z.open(metric+ '.csv') as f:
                impact_file = pd.read_csv(f)
            
            if np.isin(metric, ['wb', 'wb_sm']):
                impact=impact_file.max()
            
            else:
                impact=impact_file.sum()
                
            survey_data['tot_'+metric]=impact.values
        
    return survey_data


# Empty accumulators (for each of the 6 returned DataFrames)
all_df1, all_df2, all_df3, all_df4, all_df5, all_df6 = [
    pd.DataFrame() for _ in range(6)
]


for folder in os.listdir(input_folder):
    
    print(folder)

    match = re.search(
        r"run_([^_]+)_([^_]+)_([^_]+)_PHL_([^_]+).zip", folder
    )
   

    model, ssp, wl, run = match.groups()
    
    zip_path = "/path/to/archive.zip"
    file_inside_zip = "folder_in_zip/my_file.csv"  # path inside the ZIP
    
    path= path=os.path.join(input_folder,folder) 
    
    # open the zip
    with zipfile.ZipFile(path, "r") as z:
        with z.open(f'temp/model_forcing_{model}_{ssp}_{wl}_PHL_{run}.csv') as f:
            data = pd.read_csv(f)
    

    shock_path = 'shocks_aggregated.csv'
    
    print('open file')
    

    
    print('calc_total impacts')

    data_analysed = calc_total_impacts(data, path)
    
    print('tipping calculation')

    data_analysed = check_tipping(data_analysed,shock_path, RECOVERY_TIME) 

    # Load and process
    dfs = aggregate_results(data_analysed)
    

    # Add identifying columns and append to cumulative DataFrames
    for i, df in enumerate(dfs):
        df = df.copy()
        df["model"] = model
        df["ssp"] = ssp
        df["wl"] = wl
        df["run"] = run

        # Append to the corresponding accumulator
        if i == 0:
            all_df1 = pd.concat([all_df1, df], ignore_index=True)
        elif i == 1:
            all_df2 = pd.concat([all_df2, df], ignore_index=True)
        elif i == 2:
            all_df3 = pd.concat([all_df3, df], ignore_index=True)
        elif i == 3:
            all_df4 = pd.concat([all_df4, df], ignore_index=True)
        elif i == 4:
            all_df5 = pd.concat([all_df5, df], ignore_index=True)
        elif i == 5:
            all_df6 = pd.concat([all_df6, df], ignore_index=True)
            
    all_dfs=[all_df1,all_df2, all_df3, all_df4, all_df5, all_df6]
            
    names = [
    "tipped_decile.csv",
    "tipped_n_events.csv",
    "tipped.csv",
    "weighted_avg_decile.csv",
    "overall_sum.csv",
    "weighted_avg_event_decile.csv"
        ]

    for df, name in zip(all_dfs,names):
        df.to_csv(os.path.join(results, f'{name}'), index=False)
        print(os.path.join(results, f'{name}'))

