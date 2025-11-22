#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:53:19 2025

@author: insauer
"""

import argparse
import os
import sys
import numpy as np
#import imp
import glob
import numpy
from shutil import copyfile
import pandas as pd
from pathlib import Path


"""The scripts starts a run for each combination of climate forcing and GHM
   and permits damage generation for each combination by calling schedule_sim.py
"""

parser = argparse.ArgumentParser(
    description='schedules climada runs for different parameter combinations')
# parser.add_argument(
#     '--parameters', type=str, default="parameters.py",
#     help='parameters file')
parser.add_argument(
    '--dry', action="store_true",
    help='dry run (do not run Climada)')
parser.add_argument(
    '--shared', action="store_true",
    help='share nodes on cluster')
parser.add_argument(
    '--notify', action="store_true",
    help='notify per mail when done')
parser.add_argument(
    '--minutes', type=int, default=0,
    help='maximal minutes to run on cluster (< 60)')
parser.add_argument(
    '--hours', type=int, default=24,
    help='maximal hours to run on cluster (168=week, 720=month)')
parser.add_argument(
    '--threads', type=int, default=16,
    help='maximal number of threads on cluster (<= 16)')
parser.add_argument(
    '--mem_per_cpu', type=int, default=3584,
    help='number of memory per CPU (3584 is MaxMemPerCPU on cluster)')
parser.add_argument(
    '--largemem', action="store_true",
    help='use ram_gpu partition')
parser.add_argument(
    '--verbose', action="store_true",
    help='be verbose')

args = parser.parse_args()

indices = []
sys.dont_write_bytecode = True

# shock_files


runs=np.arange(25)

country='PHL'

forcing_folder = Path("/p/projects/ebm/inga/tipESM/final_forcing/forcing_run_20")


def schedule_run(flag,
                 country,
                 run_name,
                 file_name,
                 output_data_path,
                 work_path,
                 hh_path,
                 shock_path,
                 survey_file,
                 start_year):
    
    if not flag:
        run_label = "run_%s" %(run_name)
        if os.path.exists(run_label):
        #    run_id += 1
            return
        os.mkdir(run_label)
 #       desc = run_description()
 #       f = open("%s/parameters.txt" % run_label, 'w')
 #       f.write(desc)
 #       f.close()
 #       run_index.write(run_description_csv(run_label))
 #       run_index.write("\n")
        # with open("%s/settings.yml" % run_label, 'w') as f:
        #     f.write(pyaml.dump(settings_yml))
        #     for nc in glob.glob('*.nc'):
        #         copyfile(nc,"%s/%s" % (run_label,nc))
        #run_id += 1
    else:
        run_label = "."
    if args.dry:
        return
    else:
        if (int(args.hours) <= 24):
            _class = "short"
        elif (int(args.hours) <= 24 * 7):
            _class = "medium"
        else:
            _class = "long"

        run_params = {
            "job_name": "%s/%s" % (os.path.basename(os.getcwd()), run_label),
            "minutes": args.minutes,
            "hours": args.hours,
            "class": _class,
            "initialdir": run_label,
            "node_usage": "share" if args.shared else "exclusive",
            "notification": "END,FAIL,TIME_LIMIT" if args.notify else "FAIL,TIME_LIMIT",
            "comment": "%s/%s" % (os.getcwd(), run_label),
            "environment": "ALL",
            "executable": '/p/projects/ebm/inga/hhrm/hhrm_recurrent_events/misc/cluster_model.py',
            "options": " --country %s --file_name %s --work_path %s --hh_path %s --shock_path %s --survey_file %s --start_year %i"%(country, file_name, work_path, hh_path, shock_path, survey_file, start_year),
            "num_threads": args.threads,
            "mem_per_cpu": args.mem_per_cpu if not args.largemem else 15360,   # if mem_per_cpu is larger than MaxMemPerCPU then num_threads is reduced
            "other": "" if args.largemem else ""
            #"other": "#SBATCH --partition=ram_gpu" if args.largemem else ""
        }
        

        cmd = """echo "#!/bin/sh
#SBATCH --job-name=\\\"%(job_name)s\\\"
#SBATCH --comment=\\\"%(comment)s\\\"
#SBATCH --time=%(hours)02d:%(minutes)02d:00
#SBATCH --qos=%(class)s
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --export=%(environment)s
#SBATCH --exclude=$(cat /p/projects/ebm/inga/dead_nodes.txt | tr -d "\n" | tr ":" ",")
#SBATCH --mail-type=%(notification)s
#SBATCH --%(node_usage)s
#SBATCH --account=ebm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(num_threads)1d
#SBATCH --mem-per-cpu=%(mem_per_cpu)1d
#SBATCH --chdir=%(initialdir)s
%(other)s
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=%(num_threads)1d     
conda init
conda activate climada_env   
ulimit -c unlimited
%(executable)s %(options)s
" | sbatch -Q""" % run_params

        if args.verbose:
            print(cmd)

        os.system(cmd)
        #run_cnt += 1

num = len(os.listdir(forcing_folder))

single = True if num == 1 else False
if num > 1:
    print("Number of runs to be scheduled: %s" % num)
    sys.stdout.write('Run? y/N : ')
    if sys.version_info >= (3, 0):
        if input() != "y":
            exit("Aborted")


country='PHL'


# r minimum recovery_rate
#k_pub = 0.25

work_path='/p/projects/ebm/inga/hhrm'


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
        
        run_name= f'{filename[:-4]}'

        
        hh_data=pd.read_csv(hh_path, compression='zip')
        
        start_year=get_time_period(hh_data)
        

        survey_file=f'model_forcing_{filename[:-4]}.csv'


        
        schedule_run(flag=single,
                     country=country,
                     run_name=run_name,
                     file_name=filename,
                     output_data_path='',
                     work_path=work_path,
                     hh_path=hh_path,
                     shock_path=shock_path,
                     survey_file=survey_file,
                     start_year=start_year)

if num > 1:
    print("Scheduled %s runs" % num)
