# hhrm_recurrent_events

## General description

This branch is for the further development.

This model allows to simulate household recovery after recurrent extreme events.
The model is forced by spatially explicit hazard data combined with household surveys. The model is described in:

Sauer, Inga and Walsh, Brian James and Frieler, Katja and Bresch, David N. and Otto, Christian, Understanding the Distributional Effects of Recurrent Floods in the Philippines (May 27, 2024). Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4843542

The forcing needs to be generated in a preprocessing step in hhwb/util.
The scripts in the misc folder set up the runs for cluster simulations.
In the first step, the class HHRegister in /hhwb/agents/hh_register.py 
generates Household agents from the forcing input. Household agents are
defined in hhwb/agents/household.py. In a second step, the class Shock in
hhwb/agents/shock.py generates the shock timestamps and the affected
households for each shock.The Government class in hhwb/agents/government.py
sets the tax rate and records the national recovery process. The Household
class simulates the recovery of each household instance. 
The lifetime and action of the agents as well as the data storage is controlled in the class ClimateLife in hhwb/application/climate_life.py. Finally, the class DataAnalysis defined in hhwb/application/data_analysis.py is used to
summarize household specific impact metrics.

The folder local_dev contains the script for runs on a pc.
