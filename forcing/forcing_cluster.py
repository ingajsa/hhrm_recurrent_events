#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:14:12 2025

@author: insauer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import argparse
import zipfile

FUTURE_RUN=True


PIP='/p/projects/ebm/inga/tipESM/data/pip.csv'

INC_DIST='/p/projects/ebm/inga/tipESM/data/income_distribution_2015.csv'

NON_STORM_COLS=['latitude', 'longitude','geometry',	'region_id',	
                'impf_TC', 'total_population', 'centr_TC']

zip_path = '/p/projects/ebm/inga/tipESM/data/tc_input/PHL.zip'

#IBTRACS=pd.read_csv('/home/insauer/projects/STP/global_STP_paper/data/ibtracs.ALL.list.v04r00.csv')

OUTPUT_PATH=''


VUL_CELL_THRESHOLD=1e-11

UPPER_VUL_THRESHOLD=0.8
LOWER_VUL_THRESHOLD=0.2

DAYS=365.

POVERTY_INCOME=2.15

SOCIAL_TRANSFERS=0.0




class Forcing():
    """! Household definition. Computed from the FIES and interacts with the classes
    Government and Shock.
    Attributes:
        @param hhid (int): household id
        @param n_inds (int): number individuals living in the household
        @param weight (float): household weight (summing up over all household weights returns the total
                        population of the administrative unit)
        @param vul (float): household vulnerability (independent of disaster magnitude)
        

    """

    def __init__(self, country='PHL', output_addition=None,hazard_type='TC', hazard_path='None'):
        """! constructor"""

        #  Attributes set during initialisation

        self.__country = country
        self.__hazard_type = hazard_type
        self.__hazard_path = hazard_path
        self.__output_addition=output_addition
        
    
    @property
    def country(self):
        return self.__country

    @property
    def hazard_type(self):
        return self.__hazard_type
    
    @property
    def inc_dist(self):
        return self.__inc_dist
    
    @property
    def hazard(self):
        return self.__hazard
    
    @property
    def storms(self):
        return self.__storms
    
    def set_hazard(self):
        """
        Load and preprocess hazard data for the object.
    
        This method performs the following:
        1. Loads a hazard dataset from the file path specified in `self.__hazard_path`.
        2. Drops unnecessary columns (`'Unnamed: 0'` and `'value'`).
        3. Calls a helper method `__set_storms()` to process or augment the hazard data.
    
        Sets:
        -----
        self.__hazard : pd.DataFrame
            A cleaned DataFrame containing hazard information.
    

        """     

        # Read the hazard data from the specified path
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(self.__hazard_path) as f:
                self.__hazard = pd.read_csv(f)   


        # Drop columns that are not needed for further analysis
        self.__hazard = self.__hazard.drop(columns=['Unnamed: 0', 'value'])

        # Further processing, likely related to storm events
        self.__set_storms()
        
        return



    
    def set_inc_dist(self, save_inc_dist=None, ref_year=2015):
        """
        Sets the income distribution data for the current object's country.
    
        Parameters:
        -----------
        save_inc_dist : str or None, optional
            If provided, the path to save the generated income distribution CSV file.
            If None or an empty string, loads income distribution from a default source (INC_DIST).
        
        ref_year : int, optional
            The reference year for generating income distribution, used only if data is generated.
    
        Behavior:
        ---------
        - If `save_inc_dist` is not provided, the method loads pre-existing income distribution data from `INC_DIST`.
        - If `save_inc_dist` is provided, the method generates new data using `forc.generate_inc_dist(ref_year)`, saves it to the specified path, and uses it.
        - In both cases, the data is filtered by the object's country code and stored in `self.__inc_dist`.
        """
        try:
            if not save_inc_dist:
                # Load pre-existing income distribution data from a default source
                inc_dist_df = pd.read_csv(INC_DIST)
            else:
                # Generate new income distribution data based on ref_year
                inc_dist_df = forc.generate_inc_dist(ref_year)
                # Save the generated data to the specified path
                inc_dist_df.to_csv(save_inc_dist, index=False)
            
            # Filter the data for the relevant country and store it in the instance
            self.__inc_dist = inc_dist_df[inc_dist_df['country_code'] == self.__country]
    
        except Exception as e:
            # Catch and print any error that occurs during loading, generating, or filtering
            print(f"Error setting income distribution: {e}")



    def generate_forcing(self, save_path=OUTPUT_PATH):
        """
        Generate and export model forcing data for a specific country.
    
        This method performs the following steps:
        1. Identifies affected cells and populations based on hazard data.
        2. Generates a spatial impact matrix (`__spres_forcing`).
        3. Packages this into a model forcing DataFrame (`__model_forcing`).
        4. Adds socio-economic context to the forcing.
        5. Saves the forcing datasets to CSV files.
    
        Parameters:
        -----------
        save_path : str, optional
            Path where the output CSV files will be saved. Defaults to `OUTPUT_PATH`.
    
        Saves:
        ------
        - `model_forcing_<country>.csv`: The full model forcing data with socio-economic info.
        - `spres_<country>.csv`: The raw spatial impact matrix.
    
        Sets:
        -----
        self.__affected_cells : pd.DataFrame
        self.__affected_population : pd.Series or pd.DataFrame
        self.__unaffected_population : pd.Series or pd.DataFrame
        self.__spres_forcing : pd.DataFrame
        self.__model_forcing : pd.DataFrame
        """

        # Extract affected areas and population data
        self.__affected_cells, self.__affected_population, self.__unaffected_population = self.__extract_affected_cells()

        # Exit early if no affected cells were found
        if self.__affected_cells.shape[0] == 0:
            print('No hazard data found — forcing not generated.')
            return

        # Generate spatial forcing (impact matrix)
        self.__spres_forcing = self.__generate_impact_matrix()

        # Package the spatial forcing into a model-ready format
        self.__model_forcing = self.__packaging(self.__spres_forcing)

        # Add socio-economic information to the model forcing
        self.__add_socio_economic_info()

        # Save the processed outputs to CSV files
        model_forcing_filename = os.path.join(save_path, f'model_forcing_{self.__output_addition}.csv')
        spres_filename = os.path.join(save_path, f'spres_{self.__output_addition}.csv')

        self.__model_forcing.to_csv(model_forcing_filename, index=False)
        self.__spres_forcing.to_csv(spres_filename, index=False)
    
    
        return

    
    def plot_forcing(self, save_path=OUTPUT_PATH):
        """
        Plots and saves two subplots:
        1. Histogram (log10-scaled) of the number of hazard events experienced by households.
        2. Bar chart of average daily income per decile, with the international poverty line indicated.
    
        Parameters
        ----------
        save_path : str
            Directory where output plots are saved. Defaults to OUTPUT_PATH.
    
        Outputs
        -------
        Two files saved to disk:
            - forcing_<country>.pdf
            - forcing_<country>.png
        """
    
        # --- Figure settings in centimeters (converted to inches) ---
        cm = 1 / 2.54
        x_width = 11.0 * cm
        y_width = 14.5 * cm
    
        fig = plt.figure(figsize=(x_width, y_width), dpi=600)
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95, hspace=0.3)
        grid_spec = fig.add_gridspec(2, 1)
    
        # --- Top subplot: Histogram of events experienced ---
        ax1 = fig.add_subplot(grid_spec[0, 0])
    
        values = self.__model_forcing['n_events']
        weights = self.__model_forcing['weight']
    
        # Compute weighted histogram (log-scaled)
        bins = np.arange(-0.5, values.max() + 1.5, 1.0)
        hist, bin_edges = np.histogram(values, weights=weights, bins=bins)
    
        ax1.bar(bin_edges[:-1] + 0.5, np.log10(hist), color='seagreen')
        ax1.set_xlabel('Events experienced', fontsize=10)
        ax1.set_ylabel('LOG10 People', fontsize=12)
        ax1.set_title(self.__inc_dist['country_name'].values[0]+ f' {self.__output_addition}', fontsize=15)
    
        # --- Bottom subplot: Income per decile ---
        ax2 = fig.add_subplot(grid_spec[1, 0])
    
        # Extract mean income per decile
        income_columns = [f'decile{i}_mean' for i in range(1, 11)]
        income = self.__inc_dist[income_columns].values[0]
    
        ax2.bar(np.arange(1, 11), income, color='darkblue')
        ax2.axhline(y=2.15, color='firebrick', linestyle='--', label='Poverty Line (2.15 USD/day)')
        ax2.set_xlabel('Decile', fontsize=10)
        ax2.set_ylabel('USD per day', fontsize=12)
        ax2.legend(fontsize=8)
    
        # --- Save figure to disk ---
        country_code = self.__country
        fig.savefig(f'{save_path}forcing_{self.__output_addition}.pdf')
        fig.savefig(f'{save_path}forcing_{self.__output_addition}.png')
    
        return


        
    def __add_socio_economic_info(self):
        """
        Enrich the model forcing DataFrame with socio-economic attributes for each household.
    
        Specifically, this method adds:
        - Estimated income per household, based on income decile and scaled by DAYS.
        - A fixed value for social transfers (`income_sp`).
        - A subsistence income threshold (`subsistence_line`), also scaled by DAYS.
        - A unique household ID (`fhhid`), derived from the DataFrame index.
    
        Modifies:
        ---------
        self.__model_forcing : pd.DataFrame
            Adds the following columns:
            - 'income'
            - 'income_sp'
            - 'subsistence_line'
            - 'fhhid'
        """
    
        # Assign mean income per decile, scaled by number of days
        for dec in range(1, 11):
            mean_income = self.__inc_dist[f'decile{dec}_mean'].values[0]
            self.__model_forcing.loc[self.__model_forcing['decile'] == dec, 'income'] = mean_income * DAYS
    
        # Add a constant value for social transfers
        self.__model_forcing['income_sp'] = SOCIAL_TRANSFERS
    
        # Add a constant subsistence income threshold
        self.__model_forcing['subsistence_line'] = POVERTY_INCOME * DAYS
    
        # Assign a unique household ID from the row index
        self.__model_forcing['fhhid'] = self.__model_forcing.index
    
        return


    
    def __set_storms(self):
        """
        Extract and set the list of storm identifiers from the hazard DataFrame columns,
        excluding any columns specified in NON_STORM_COLS.
    
        This method updates the instance variable `self.__storms` with columns representing storms.
        """
    
        # Filter hazard DataFrame columns to exclude non-storm related columns
        self.__storms = [col for col in self.__hazard.columns if col not in NON_STORM_COLS]
    
        # No explicit return needed, but included for clarity
        return

    
    def __packaging(self, impact_matrix):
        """
        Compress the impact matrix by removing duplicate rows based on income decile and storm times,
        and calculate weights representing the total count of occurrences for each unique pattern.
    
        Parameters:
        -----------
        impact_matrix : pd.DataFrame
            DataFrame containing impact data with columns including 'decile', storm times, and 'count'.
    
        Returns:
        --------
        impact_matrix_packed : pd.DataFrame
            A reduced DataFrame with unique patterns of decile and storm impacts, their event counts,
            and a 'weight' column indicating how many original rows correspond to each unique pattern.
        """
    
        # Columns used to identify duplicates: income decile + storm time columns
        subset = ['decile'] + self.__storm_times
    
        # Drop duplicate rows based on decile and storm times to get unique patterns
        impact_matrix_packed = impact_matrix.drop_duplicates(subset=subset).reset_index(drop=True)
    
        # Keep only relevant columns for output
        impact_matrix_packed = impact_matrix_packed[subset + ['n_events']]
    
        # Merge original impact_matrix with the packed unique patterns to find corresponding indices
        df_temp = impact_matrix.merge(
            impact_matrix_packed.reset_index(),
            on=subset,
            how='left'
        )
    
        # Rename 'index' from packed df to 'corresponding_index' for clarity
        df_temp.rename(columns={'index': 'corresponding_index'}, inplace=True)
    
        # Group by corresponding_index and sum the 'count' to calculate weights
        sum_column = (
            df_temp.groupby('corresponding_index')['count']
            .sum()
            .reset_index()
            .rename(columns={'count': 'weight'})
        )
    
        # Merge the weights back into the packed impact matrix
        impact_matrix_packed = impact_matrix_packed.merge(
            sum_column,
            left_index=True,
            right_on='corresponding_index',
            how='left'
        )
    
        # Drop the helper column as it is no longer needed
        impact_matrix_packed.drop(columns=['corresponding_index'], inplace=True)
    
        return impact_matrix_packed

    
    def __add_time_stamp(self, impact_matrix):
        """
        Add date-based time stamps to the columns of an impact matrix based on storm IDs.
    
        This method uses the `IBTRACS` dataset to extract the ISO timestamp for each storm
        in `self.__storms`, then renames the columns of the `impact_matrix` using the date portion
        of those timestamps. The resulting column names reflect the storm's date (YYYY-MM-DD).
    
        Parameters:
        -----------
        impact_matrix : pd.DataFrame
            A DataFrame whose columns are storm IDs (SID), which will be renamed to date strings.
    
        Returns:
        --------
        impact_matrix : pd.DataFrame
            The same DataFrame, but with columns renamed to their corresponding storm dates.
    
        Sets:
        -----
        self.__storm_times : list of str
            A list of date strings (YYYY-MM-DD) corresponding to the storms in `self.__storms`.
    
        Raises:
        -------
        Prints an error message if processing fails.
        """

        # Filter IBTRACS to include only entries for relevant storms, and select columns
        date_time = IBTRACS.loc[IBTRACS['SID'].isin(self.__storms), ['SID', 'ISO_TIME']].copy()

        # Sort by ISO_TIME (descending) to prioritize latest entries before deduplication
        date_time = date_time.sort_values(by='ISO_TIME', ascending=False)

        # Drop duplicate SIDs, keeping the most recent entry
        date_time = date_time.drop_duplicates(subset='SID')

        # Sort the remaining entries chronologically
        date_time = date_time.sort_values(by='ISO_TIME')

        # Extract only the date part (YYYY-MM-DD) of the ISO timestamp
        date_time['ISO_TIME_day'] = date_time['ISO_TIME'].str[:10]

        # Create a mapping from storm ID to date string
        rename_mapping = dict(zip(date_time['SID'], date_time['ISO_TIME_day']))

        # Rename the columns in the impact matrix using the date strings
        impact_matrix.rename(columns=rename_mapping, inplace=True)

        # Store the ordered list of date strings for future use
        self.__storm_times = list(date_time['ISO_TIME_day'])

        return impact_matrix



    
    def __extract_affected_cells(self):
        """
        Identifies and separates affected and unaffected grid cells based on hazard (e.g., storm) data, 
        and calculates the total population in each category.
    
        Returns:
            affected_cells (DataFrame): Subset of grid cells with non-zero hazard exposure.
            affected_population (int): Total population in affected cells.
            unaffected_population (int): Total population in unaffected cells.
        """
    
        # Step 1: Calculate the share of total population for each grid cell to normalize population
        # TODO: The normalization should eventually be done using ISIMIP data instead
        self.__hazard['population_share'] = self.__hazard['total_population'] / self.__hazard['total_population'].sum()
    
        # Step 2: Scale the population share to match the total population in the incidence distribution dataset
        # and convert it to an integer count
        self.__hazard['population'] = (
            self.__hazard['population_share'] * self.__inc_dist['reporting_pop'].sum()
        ).astype(int)
    
        # Step 3: Select only the grid cells that have a non-zero population
        populated_cells = self.__hazard[self.__hazard['population'] > 0]
    
        # Step 4: Split populated cells into affected and unaffected based on storm data
        # A cell is unaffected if the sum of all storm indicators across columns is zero
        unaffected_cells = populated_cells[np.array(populated_cells[self.__storms].sum(axis=1) == 0)]
        affected_cells = populated_cells[~np.array(populated_cells[self.__storms].sum(axis=1) == 0)]
    
        # Step 5: Calculate total population in each category
        affected_population = affected_cells['population'].sum()
        unaffected_population = unaffected_cells['population'].sum()
    
        # Return the affected cells and the population stats
        return affected_cells, affected_population, unaffected_population
    
    def __generate_cell_matrix(self, cell):
        """
        Generate a DataFrame representing individual households within a grid cell,
        distributing the population evenly across 10 income deciles and initializing
        vulnerability scores for each storm to zero.
    
        Parameters:
        -----------
        cell : tuple
            A tuple from `iterrows()` where `cell[1]` contains population, longitude, latitude, etc.
    
        Returns:
        --------
        cell_matrix : pd.DataFrame
            DataFrame with one row per person in the cell, containing columns:
            - 'longitude': longitude of the cell
            - 'latitude': latitude of the cell
            - 'decile': income decile (1 to 10)
            - columns for each storm in `self.__storms`, initialized to 0
        """
    
        n_pop = cell['population']  # Total population in the cell
    
        # Base number of people per decile (integer division)
        n_pop_per_dec = int(n_pop // 10)
    
        # Residual population that can't be evenly distributed
        residual_pop = n_pop - (10 * n_pop_per_dec)
    
        # Create empty DataFrame for the cell's population,
        # columns: lon, lat, decile + storms; index: one per person
        cell_matrix = pd.DataFrame(
            columns=['longitude', 'latitude', 'decile'] + list(self.__storms),
            index=np.arange(n_pop).astype(int)
        )
    
        # Randomly choose deciles to assign the residual population
        residual_decs = np.random.choice(np.arange(1, 11), residual_pop, replace=False)
    
        if n_pop_per_dec > 0:
            i = 0  # row index counter
    
            # Assign decile numbers to rows
            for dec in range(1, 11):
                # Number of people in this decile: base plus 1 if in residual
                n_in_dec = n_pop_per_dec + (1 if dec in residual_decs else 0)
    
                # Assign decile number to the relevant rows
                cell_matrix.loc[i:i + n_in_dec - 1, 'decile'] = dec
                i += n_in_dec
    
        else:
            # If population is less than 10, assign deciles only to residual people
            cell_matrix.loc[:len(residual_decs) - 1, 'decile'] = residual_decs
    
        # Assign constant longitude and latitude values to all rows
        cell_matrix.loc[:, 'longitude'] = cell['longitude']
        cell_matrix.loc[:, 'latitude'] = cell['latitude']
        
        
    
        # Initialize storm vulnerability columns to zero
        cell_matrix.loc[:, self.__storms] = 0
    
        return cell_matrix

    
    def __distribute_tc_vulnerabilities(self, cell, cell_matrix, storm):
        """
        Distribute tropical cyclone (TC) vulnerability values across households in a cell
        based on the vulnerability factor and cell population.
    
        The function uses a vulnerability factor (`vul_fac`) from the Eberenz function to compute
        a total vulnerability value (`vul_sum`), then distributes that value across a subset of
        the cell's households depending on its magnitude and predefined thresholds.
    
        Parameters:
        -----------
        cell : tuple
            A tuple from `iterrows()`, where `cell[1]` is the row containing population and storm vulnerability.
            
        cell_matrix : pd.DataFrame
            A matrix representing households in a single grid cell, to which vulnerability values are applied.
            
        storm : str
            Storm identifier (column name) to which vulnerability values should be assigned.
    
        Returns:
        --------
        cell_matrix : pd.DataFrame
            Updated matrix with the `storm` column populated based on vulnerability distribution logic.
        """
    
        # Extract vulnerability factor for this cell and storm
        vul_fac = cell[storm]
    
        # Compute total vulnerability impact: vulnerability factor × population
        vul_sum = cell['population'] * vul_fac
    
        # Proceed only if the vulnerability factor exceeds a defined threshold
        if vul_fac > VUL_CELL_THRESHOLD:
            hh_indices = cell_matrix.index  # All household indices in the cell
    
            # Case 1: Small total vulnerability — affect one household or none
            if vul_sum < UPPER_VUL_THRESHOLD:
                if vul_sum > LOWER_VUL_THRESHOLD:
                    new_cell_vul = vul_sum
                    n_affected = 1
                else:
                    # Vulnerability is too small to affect any household
                    new_cell_vul = 0.0
                    n_affected = 0
    
            else:
                # Case 2: Moderate-to-high vulnerability — randomly assign valid vulnerability value
                new_cell_vul = np.random.randint(
                    int(LOWER_VUL_THRESHOLD * 10),
                    int(UPPER_VUL_THRESHOLD * 10),
                    size=1
                )[0] / 10.0
                n_affected = round(vul_sum / new_cell_vul)
    
            # Assign vulnerability to selected households
            if n_affected < len(hh_indices):
                # Randomly choose which households are affected
                aff_indices = np.random.choice(hh_indices, n_affected, replace=False)
    
                # Set vulnerability values: 0 for all, then assign to selected
                cell_matrix.loc[hh_indices, storm] = 0
                cell_matrix.loc[aff_indices, storm] = new_cell_vul
            else:
                # If all or more households must be affected, apply full vulnerability factor uniformly
                cell_matrix.loc[:, storm] = vul_fac
    
        # If vulnerability factor doesn't pass threshold, storm column remains unchanged (zeros)
        return cell_matrix


    def __generate_impact_matrix(self):
        """
        Generate the impact matrix for all affected grid cells and storms.
    
        This method iterates over each affected grid cell, computes its impact matrix,
        distributes tropical cyclone vulnerabilities for each storm, and then aggregates 
        results across all grid cells. It also includes unaffected population rows and 
        timestamps for each storm event.
    
        Returns:
        --------
        impact_matrix : pd.DataFrame
            A DataFrame representing all affected and unaffected populations with 
            their associated vulnerability metrics and event counts.
        """
    
        # Initialize the full impact matrix as an empty DataFrame
        
    
        # Set progress counter
        counter = 0
        
        # Output CSV path
        output_path = os.path.join(OUTPUT_PATH, f'im_{self.__output_addition}.csv')
    
        # Loop through each affected grid cell
        for _, cell in self.__affected_cells.iterrows():
            # Print progress percentage
            progress = np.round((counter / self.__affected_cells.shape[0]) * 100, 3)
            print(f"Forcing generation concluded by {progress}%")
    
            # Generate initial impact matrix for this cell
            cell_matrix = self.__generate_cell_matrix(cell)
            gc.collect()
    
            # For each storm, apply vulnerability distribution to the cell
            for storm in self.__storms:
                cell_matrix = self.__distribute_tc_vulnerabilities(cell, cell_matrix, storm)
    
            # Group by all columns and count unique patterns (rows)
            try:
                cell_matrix['count'] = cell_matrix.groupby(cell_matrix.columns.tolist()).transform('size')
                cell_matrix = cell_matrix.drop_duplicates()
            except IndexError:
                print('Error while counting or dropping duplicates — possibly empty matrix.')
    
            
            # Append to CSV incrementally
            if not cell_matrix.empty:
                # Write header only on the first iteration
                
                write_header = counter == 0
                cell_matrix.to_csv(output_path, mode='a', header=write_header, index=False)
                
            # Check if any column has at least one string
            has_strings = cell_matrix[self.__storms].applymap(type).eq(str).any().any()
            
            if has_strings:
                print("⚠️ Found strings in the DataFrame slice!")
            else:
                print("✅ No strings found, all numeric.")
    
            counter += 1
            
        impact_matrix = pd.read_csv(output_path)
        
        for col in impact_matrix.columns:
            impact_matrix[col] = pd.to_numeric(impact_matrix[col], errors="coerce")
    
        # Add unaffected population rows to impact matrix
        unaffected_rows = self.__get_unaffected_rows(impact_matrix.columns)
        impact_matrix = pd.concat([impact_matrix, unaffected_rows], ignore_index=True)
    
        # Count number of storm events with vulnerability over the threshold
        impact_matrix['n_events'] = (impact_matrix[self.__storms] >= LOWER_VUL_THRESHOLD).sum(axis=1)
    

        # Replace storm IDs in columns with timestamped versions
        if not FUTURE_RUN:
             
            impact_matrix = self.__add_time_stamp(impact_matrix)
        
        else:
            self.__storm_times = self.__storms
            
        return impact_matrix

    
    def __get_unaffected_rows(self, columns):
        """
        Generate a DataFrame representing unaffected population distribution across deciles.
    
        This method simulates how the unaffected population is distributed across 10 deciles.
        The population is evenly divided across deciles, with any remainder randomly assigned
        to deciles to ensure the total adds up to `self.__unaffected_population`.
    
        Parameters:
        -----------
        columns : list of str
            List of column names for the resulting DataFrame. These will be initialized to 0,
            and additional columns 'decile' and 'count' will be added.
    
        Returns:
        --------
        unaffected_rows : pd.DataFrame
            A DataFrame with 10 rows (one for each decile), including:
            - All specified columns initialized to 0.
            - A 'decile' column with values 1 through 10.
            - A 'count' column indicating the number of unaffected individuals in each decile.
        """
        # Initialize a DataFrame with 10 rows, all specified columns set to 0
        unaffected_rows = pd.DataFrame(0, index=range(10), columns=columns)
    
        # Add 'decile' column ranging from 1 to 10
        unaffected_rows['decile'] = np.arange(1, 11)
    
        # Distribute the unaffected population equally across the 10 deciles
        unaffected_rows['count'] = np.ones(10) * int(self.__unaffected_population // 10)
    
        # Calculate any leftover individuals due to integer division
        residual_unaffected = self.__unaffected_population - (10 * int(self.__unaffected_population // 10))
    
        # Randomly distribute the remaining individuals to deciles (without replacement)
        if residual_unaffected > 0:
            residual_decs = np.random.choice(np.arange(1, 11), residual_unaffected, replace=False)
            unaffected_rows.loc[unaffected_rows['decile'].isin(residual_decs), 'count'] += 1

        return unaffected_rows

       
    @staticmethod
    def closest_year(group, ref_year=2015):
        """
        Find the row in a DataFrame (group) with the reporting year closest to the provided reference year.
    
        Parameters:
        -----------
        group : pandas DataFrame
            A DataFrame (group) containing income distribution data for a specific country, including the `reporting_year` column.
    
        ref_year : int, optional
            The reference year to which the closest reporting year should be found. Default is 2015.
    
        Returns:
        --------
        closest_year : pandas Series
            The row from the `group` DataFrame where the `reporting_year` is closest to the `ref_year`.
            The `year_diff` column is dropped from the returned row.
    
        Process:
        --------
        1. Calculates the absolute difference between the `reporting_year` and the provided `ref_year` for each row.
        2. Identifies the row with the minimum difference (i.e., the closest year to the reference year).
        3. Returns the row (as a pandas Series) with the closest reporting year, excluding the temporary `year_diff` column.
        """
        
        # Calculate the absolute difference between the reporting year and reference year
        group['year_diff'] = abs(group['reporting_year'] - ref_year)
        
        # Find the row with the smallest year difference
        closest_year = group.loc[group['year_diff'].idxmin()]
        
        # Return the row with the closest year, dropping the 'year_diff' column
        return closest_year.drop('year_diff')

    
    @staticmethod
    def generate_inc_dist(ref_year=2015):
        """
        Generate income distribution data for different countries based on the given reference year.
    
        Parameters:
        -----------
        ref_year : int, optional
            The reference year for which income distribution data is generated. 
            Default is 2015.
    
        Returns:
        --------
        inc_dist : pandas DataFrame
            A DataFrame containing income distribution data, including mean income per decile and
            other related information such as country name, country code, and reporting year.
    
        Process:
        --------
        1. Reads income distribution data from a CSV file defined by the `PIP` constant.
        2. Filters the data to include only national-level data, with an exception for Argentina, where urban data is used.
        3. For each country, selects the closest reporting year to the provided `ref_year` using `Forcing.closest_year()`.
        4. Calculates the number of people per decile (assuming 10% of the total population per decile).
        5. Computes the total income for each country by multiplying the mean income by the total reporting population.
        6. Calculates the income per decile and the mean income per decile for each of the 10 deciles (decile1 through decile10).
        7. Selects relevant columns to be included in the returned DataFrame, such as country name, country code,
           reporting year, mean income per decile, and reporting population.
        8. Returns the DataFrame (`inc_dist`) containing the computed income distribution data.
    
        Example Usage:
        --------------
        # Generate income distribution data for the year 2015 (default)
        income_distribution_2015 = generate_inc_dist()
    
        # Generate income distribution data for the year 2020
        income_distribution_2020 = generate_inc_dist(ref_year=2020)
    
        Dependencies:
        -------------
        - pandas: For data manipulation and DataFrame operations.
        - Forcing.closest_year(): A helper function to find the closest available reporting year to `ref_year`.
        - PIP: The path or constant for the income survey data (CSV file).
    
        Columns in the Returned DataFrame:
        -----------------------------------
        - country_name: The name of the country.
        - country_code: The country code.
        - reporting_year: The reporting year of the data.
        - decile1_mean to decile10_mean: The mean income for each income decile.
        - reporting_pop: The total population of the country in the dataset.
        - people_per_decile: The number of people in each decile (10% of the population).
        """
        
        data = pd.read_csv(PIP)
        
        # We use national data except for Argentina, where we use urban data
        nat_data = data.loc[(data['survey_coverage'] == 'national') | (data['country_code'] == 'ARG')]
        
        # Filter income data for the closest year to the reference year
        filtered_data = nat_data.groupby('country_code').apply(Forcing.closest_year, ref_year=ref_year).reset_index(drop=True)
        
        # Deciles list representing 10 income deciles
        deciles = ['decile1', 'decile2', 'decile3', 'decile4', 'decile5',
                   'decile6', 'decile7', 'decile8', 'decile9', 'decile10']
        
        # People per decile are 10% of the total population
        filtered_data['people_per_decile'] = 0.1 * filtered_data['reporting_pop']
        
        # Total income is the mean income times the overall population
        filtered_data['total_income'] = filtered_data['mean'] * filtered_data['reporting_pop']
        
        # Calculate income per decile
        for dec in deciles:
            # Overall income per decile
            filtered_data['{}_total'.format(dec)] = (filtered_data[dec] * filtered_data['total_income'])
            
            # Mean income per decile
            filtered_data['{}_mean'.format(dec)] = filtered_data['{}_total'.format(dec)] / filtered_data['people_per_decile']
        
        # Select the relevant columns for the final DataFrame
        cols = ['country_name', 'country_code', 'reporting_year', 'decile1_mean',
                'decile2_mean', 'decile3_mean', 'decile4_mean', 'decile5_mean',
                'decile6_mean', 'decile7_mean', 'decile8_mean', 'decile9_mean',
                'decile10_mean', 'reporting_pop', 'people_per_decile']
    
        inc_dist = filtered_data[cols]
        
        return inc_dist




parser = argparse.ArgumentParser(
    description='run hhwb for different shock series')


parser.add_argument(
    '--file_name', type=str,
    help='forcing input file')

parser.add_argument(
    '--country', type=str, default='PHL',
    help='country')

parser.add_argument(
    '--folder_name', type=str,
    help='folder_name')

parser.add_argument(
    '--run_name', type=str,
    help='run_name')

args = parser.parse_args()

# input file directory



directory=f'PHL/{args.folder_name}/'
    
forc= Forcing(country=args.country, output_addition=args.run_name, hazard_path=directory+args.file_name)

forc.set_inc_dist()

forc.set_hazard()

forc.generate_forcing()

forc.plot_forcing()

print(forc.inc_dist)
print(forc.hazard)
print(forc.storms)

