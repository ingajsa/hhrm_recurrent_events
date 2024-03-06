# hhrm_recurrent_events

## General description
This model allows to simulate household recovery after recurrent extreme events.
The model is forced by spatially explicit hazard data combined with household surveys. The model is described in:

https://doi.org/10.21203/rs.3.rs-2911340/v1

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
