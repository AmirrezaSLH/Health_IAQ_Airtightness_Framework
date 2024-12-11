# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:01:05 2024

@author: asalehi
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd

import json

from shapely.geometry import Polygon


from matplotlib.path import Path


from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from matplotlib.patches import PathPatch

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.image as mpimg

from statistics import mean, median

#%%

import main

def code_comply_all( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    if  (climatezone_grid_map[GRID_KEY] == 1) or (climatezone_grid_map[GRID_KEY] == 2) :
        if ach50_segment_dict['SF_V1'] > 5:
            ach50_segment_dict['SF_V1'] = 5
        if ach50_segment_dict['SF_V2'] > 5:
            ach50_segment_dict['SF_V2'] = 5
        if ach50_segment_dict['SF_V3'] > 5:
            ach50_segment_dict['SF_V3'] = 5
    else:
        if ach50_segment_dict['SF_V1'] > 3:    
            ach50_segment_dict['SF_V1'] = 3
        if ach50_segment_dict['SF_V2'] > 3:
            ach50_segment_dict['SF_V2'] = 3
        if ach50_segment_dict['SF_V3'] > 3:
            ach50_segment_dict['SF_V3'] = 3
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention


def code_comply_V1( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    if  (climatezone_grid_map[GRID_KEY] == 1) or (climatezone_grid_map[GRID_KEY] == 2) :
        if ach50_segment_dict['SF_V1'] > 5: 
            ach50_segment_dict['SF_V1'] = 5
    else:
        if ach50_segment_dict['SF_V1'] > 3: 
            ach50_segment_dict['SF_V1'] = 3
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    return Finf_intervention, ACH_intervention

def code_comply_V2( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    if  (climatezone_grid_map[GRID_KEY] == 1) or (climatezone_grid_map[GRID_KEY] == 2) :
        if ach50_segment_dict['SF_V2'] > 4:
            ach50_segment_dict['SF_V2'] = 5
    else:
        if ach50_segment_dict['SF_V2'] > 3:
            ach50_segment_dict['SF_V2'] = 3
    #print(ach50_segment_dict)
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    return Finf_intervention, ACH_intervention

def code_comply_V3( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    if  (climatezone_grid_map[GRID_KEY] == 1) or (climatezone_grid_map[GRID_KEY] == 2) :
        if ach50_segment_dict['SF_V3'] > 4:
            ach50_segment_dict['SF_V3'] = 5
    else:
        if ach50_segment_dict['SF_V3'] > 3:
            ach50_segment_dict['SF_V3'] = 3
    #print(ach50_segment_dict)
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    return Finf_intervention, ACH_intervention



#%%

import plot_functions

def plot_histogram(data, title, xlabel, ylabel, color='skyblue', edgecolor='black', save = False):
    """
    Plots a histogram with the given data and customization options.

    Parameters:
    - data: Pandas Series containing the data to plot.
    - title: String, title of the histogram.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    - color: String, color of the histogram bars. Default is 'skyblue'.
    - edgecolor: String, color of the bar edges. Default is 'black'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility
    
    if isinstance(data, list):
        data = pd.Series(data)
    data.hist(color=color, edgecolor=edgecolor)
    
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Optional: Add grid for better readability
    plt.grid(axis='y', alpha=0.75)

    plt.show()
    
def plot_boxplot(data, title, xlabel, ylabel, color='skyblue', save=False, filename='boxplot.png'):
    """
    Plots a box plot with the given data and customization options.

    Parameters:
    - data: List or Pandas Series containing the data to plot.
    - title: String, title of the box plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    - color: String, color of the box plot elements. Default is 'skyblue'.
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility
    
    # Creating the box plot
    box = plt.boxplot(data, patch_artist=True)  # 'patch_artist' must be True to fill with color
    plt.setp(box['boxes'], color=color, facecolor=color)  # Setting color of the boxes
    plt.setp(box['whiskers'], color=color)  # Setting color of the whiskers
    plt.setp(box['caps'], color=color)  # Setting color of the caps
    plt.setp(box['medians'], color='black')  # Setting color of the medians
    
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Saving the plot if required
    if save:
        plt.savefig(filename)

    plt.show()

def plot_boxplots(data, title, xlabels, ylabel, colors='skyblue', save=False, filename='boxplot.png', zero_line = False):
    """
    Plots side-by-side box plots for given datasets on a single plot.

    Parameters:
    - data: List of lists or Pandas Series to plot side by side.
    - title: String, title of the box plot.
    - xlabels: List of strings, labels for each box plot on the x-axis.
    - ylabel: String, label for the y-axis.
    - colors: List of strings or a single string for the color of the box plot elements. Default is 'skyblue'.
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility

    # Check if colors is a single color and expand it to the length of data if needed
    if isinstance(colors, str):
        colors = [colors] * len(data)

    # Creating the box plot
    box = plt.boxplot(data, patch_artist=True, positions=range(1, len(data) + 1), showfliers = False)

    # Set colors and properties for each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['caps'], color='black')
    plt.setp(box['medians'], color='black')
    
    if zero_line == True:
        plt.axhline(y=0, color='red', linestyle='dashed', linewidth=1)
        
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.ylabel(ylabel, fontsize=12)
    
    # Setting x-axis labels
    plt.xticks(range(1, len(data) + 1), xlabels, fontsize=12)
    #plt.ylim(bottom=0)
    # Saving the plot if required
    if save:
        plt.savefig(filename)

    plt.show()
    
    return plt
    
def plot_grouped_boxplots(data, title, group_labels, subplot_labels, ylabel, colors=('skyblue', 'blue', 'gray', 'red'), save=False, filename='boxplot.png'):
    """
    Plots grouped box plots with subgroups for given datasets on a single plot.

    Parameters:
    - data: List of lists, where each list contains multiple datasets for subgroups.
    - title: String, title of the box plot.
    - group_labels: List of strings, labels for each main group on the x-axis.
    - subplot_labels: List of strings, labels for each subgroup within the main groups.
    - ylabel: String, label for the y-axis.
    - colors: Tuple of strings, colors for each subgroup. Default is ('skyblue', 'green').
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(10, 6))  # Set figure size for better visibility
    num_groups = len(data)
    num_subgroups = len(data[0])
    width = 0.35  # width of each boxplot within a group
    
    # Create positions for each subgroup
    positions = []
    for i in range(num_groups):
        start = i * num_subgroups * 3
        positions += [start + j * width * 3 for j in range(num_subgroups)]
    
    # Flatten the data and adjust colors accordingly
    flat_data = [item for sublist in data for item in sublist]
    colors = list(colors) * num_subgroups

    # Create box plots
    box = plt.boxplot(flat_data, patch_artist=True, positions=positions, widths=width, showfliers = False)

    # Set colors for each subgroup
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set properties for other elements
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['caps'], color='black')
    plt.setp(box['medians'], color='black')

    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=14)
    
    # Set custom x-axis labels
    ax = plt.gca()
    tick_positions = [np.mean(positions[i:i+num_subgroups]) for i in range(0, len(positions), num_subgroups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels)

    # Add legend for subgroups
    plt.legend([ box["boxes"][i] for i in range(0, num_subgroups)], subplot_labels, loc='upper right')

    # Save the plot if required
    if save:
        plt.savefig(filename)

    plt.show()
    
def plot_grouped_boxplots2(data, title, subplot_labels, group_labels, ylabel, colors=('gray', 'skyblue', 'lime'), save=False, filename='boxplot.png'):
    """
    Plots grouped box plots with subgroups for given datasets on a single plot.

    Parameters:
    - data: List of lists, where each list contains multiple datasets for subgroups.
    - title: String, title of the box plot.
    - group_labels: List of strings, labels for each main group on the x-axis.
    - subplot_labels: List of strings, labels for each subgroup within the main groups.
    - ylabel: String, label for the y-axis.
    - colors: Tuple of strings, colors for each subgroup. Default is ('gray', 'skyblue').
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(12, 8))  # Adjusted figure size for better visibility
    num_subgroups = len(data)
    num_groups = len(data[0])
    width = 0.4  # Adjusted width for each boxplot within a group

    # Create positions for each subgroup to prevent overlap and ensure clarity
    positions = []
    group_gap = 0.8  # Increased gap between groups for clarity
    for i in range(num_groups):
        start = i * num_subgroups * group_gap
        positions += [start + j * width for j in range(num_subgroups)]

    # Flatten the data and prepare colors for each box
    flat_data = [item for pair in zip(*data) for item in pair]
    colors = list(colors) * (len(flat_data) // len(colors))

    # Create box plots
    box = plt.boxplot(flat_data, patch_artist=True, positions=positions, widths=width, showfliers=False)

    # Set colors for each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set properties for other elements
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['caps'], color='black')
    plt.setp(box['medians'], color='black')

    # Add titles and labels
    plt.title(title, fontsize=22)
    plt.ylabel(ylabel, fontsize=18)
    
    # Custom x-axis labels
    ax = plt.gca()
    tick_positions = [np.mean(positions[i:i+num_subgroups]) for i in range(0, len(positions), num_subgroups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, fontsize=16)

    # Add gridlines for better readability
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5)

    # Add legend for subgroups
    plt.legend([box["boxes"][i] for i in range(0, num_subgroups)], subplot_labels, loc='upper right', fontsize=16)

    # Save the plot if required
    if save:
        plt.savefig(filename, dpi=300)  # Higher DPI for better quality output

    plt.show()
#%%

def calculate_spatial_mean( data_list):
    
    """
    Calculate the mean of values across a list of dictionaries,
    assuming each dictionary has the same keys.

    Parameters:
    - data_list: List of dictionaries with numeric values and the same keys.

    Returns:
    - dict: Dictionary with the mean values computed across the dictionaries for each key.
    """
    data_mean = {}
    grid_list = data_list[0].keys()
    
    for g in grid_list:
        # Collect the values for each grid from all dictionaries
        values = [data[g] for data in data_list]
        
        # Calculate mean using numpy for efficiency
        data_mean[g] = np.mean(values)
    
    return data_mean
            
def process_sf_conc(data):
    pop_dict = main.init_population()
    data_national_v1 = []
    data_national_v2 = []
    data_national_v3 = []
    
    for i in range(n):
        # Initialize the accumulator for the weighted target values and total population
        national_average_value_V1  = 0
        national_average_value_V2  = 0
        national_average_value_V3  = 0
        total_pop = 0
        
        
        # Iterate through the groups to accumulate the weighted target values and total population
        for g in pop_dict.keys():
            
            data_v1 = data[i][g]['SF_V1']
            data_v2 = data[i][g]['SF_V2']
            data_v3 = data[i][g]['SF_V3']
            # Accumulate the total population
            total_pop += pop_dict[g]
            # Accumulate the weighted target value for the group
            national_average_value_V1 += data_v1 * pop_dict[g]
            national_average_value_V2 += data_v2 * pop_dict[g]
            national_average_value_V3 += data_v3 * pop_dict[g]
        # Calculate the weighted national average
        national_average_value_V1 /= total_pop
        national_average_value_V2 /= total_pop
        national_average_value_V3 /= total_pop
        
        data_national_v1.append(national_average_value_V1)
        data_national_v2.append(national_average_value_V2)
        data_national_v3.append(national_average_value_V3)
        
        
    return data_national_v1, data_national_v2, data_national_v3

def aggregate_results( dCin, dM, benefits, dY, n):
    pop_dict = main.init_population()
    dCin_national = []
    dM_national = []
    dY_national = []
    benefits_national = []
    '''costs_national = []
    net_benefits_national = []'''

    for i in range(n):
        dCin_national.append(  main.national_average(dCin[i], pop_dict) )
        dM_national.append( sum(dM[i].values()) )
        dY_national.append ( main.national_average(dY[i], pop_dict) * 100000)
        #costs_national.append( sum( costs[i].values()) / 1000000000)
        benefits_national.append( sum( benefits[i].values()) / 1000000000)
        #net_benefits_national.append( sum(net_benefits[i].values()) / 1000000000 )
        
    #return dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national
    return dCin_national, dM_national, dY_national, benefits_national

def generate_maps_IECC(dCin, dY, benefit, cost, nb, phi_value, occupy ):
    
    pop_dict = main.init_population()
    aq_grid_gdf = main.init_AQ_grids()
    '''
    dY_mean = calculate_spatial_mean(dY)
    dY_adjusted = {key : value/phi_value for key, value in dY_mean.items()}
    ticks = [0, 20, 40, 60, 80]
    plot_functions.create_contour_plot_dy(aq_grid_gdf, dY_adjusted, 'dY', '', 'Mortality Incidence Rate Reduction (Per 100,000 resident)', legend_ticks = ticks, color ='blue')
    
    benefit_mean = calculate_spatial_mean(benefit)
    benefit_capita = {key : value/(pop_dict[key]*phi_value) for key, value in benefit_mean.items()}
    ticks = [0, 1500, 3000, 4500, 6000, 7500]
    plot_functions.create_contour_plot(aq_grid_gdf, benefit_capita, 'bc', '', 'Benefits (USD per resident)', legend_ticks = ticks, color ='blue')

    cost_mean = calculate_spatial_mean(cost)
    cost_house = {key : occupy*(value/(pop_dict[key]*phi_value)) for key, value in cost_mean.items()}
    ticks = [0, 1000, 2000, 3000]
    plot_functions.create_contour_plot(aq_grid_gdf, cost_house, 'cc', '', 'Intervention Costs (USD per house)', legend_ticks=ticks)

    nb_mean = calculate_spatial_mean(nb)
    nb_capita = {key : value/(pop_dict[key]*phi_value) for key, value in nb_mean.items()}
    plot_functions.create_contour_plot_negetive(aq_grid_gdf, nb_capita, 'nbc', '', 'Net Benefits (USD per resident)', color ='red_blue')
    '''
    dCin_mean = calculate_spatial_mean(dCin)
    print(dCin_mean)
    dCin_adjusted = {key : value/phi_value for key, value in dCin_mean.items()}
    plot_functions.create_contour_plot_dC(aq_grid_gdf, dCin_adjusted, 'dC', '', r'Improvements in Indoor $PM_{2.5}$ ($\mu g/m^3$)', color ='blue')

    return 0
    
def generate_box_plots(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list):
    
    pop_dict = main.init_population()
    print(dY_national)
    plot_boxplots(dY_national, 'National Annual Mortality Incidence Rate Reduction', scenario_list,'per 100,000')
    plot_boxplots(dM_national, 'National Annual Avoided Mortality', scenario_list,'Premature deaths')
    plot_boxplots(benefits_national, 'National Annual Total Benefits', scenario_list,'Billion USD')
    plot_boxplots(costs_national, 'National Annual Total Costs', scenario_list,'Billion USD')
    plot_boxplots(net_benefits_national, 'National Annual Total Net Benefits', scenario_list,'Billion USD')
    
    #Total Cost Benefit 
    cost_benefit = [benefits_national, net_benefits_national, costs_national]  # List of datasets
    cost_benefit_group_labels  = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    plot_grouped_boxplots(cost_benefit, 'National Annual Cost Benefit Analysis', cost_benefit_group_labels, scenario_list, 'Billion USD')
    
    '''
    #Individual Cost Benefit 
    net_benefit_ind = (net_benefits_national / sum(pop_dict.values())) * 1000000000
    benefit_ind = (benefits_national / sum(pop_dict.values())) * 1000000000
    cost_ind = (costs_national / sum(pop_dict.values())) * 1000000000
    
    cost_benefit_ind = [benefit_ind, net_benefit_ind, cost_ind]  # List of datasets
    cost_benefit_ind_xlabels = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    plot_boxplots(cost_benefit_ind, 'National Annual Cost Benefit Analysis per person', cost_benefit_ind_xlabels, 'USD', colors=['skyblue', 'green', 'red'])
    '''
    plot_boxplots(dCin_national, 'National Annual $PM_2.5$ Reduction in Indoor Concentrations', scenario_list, r'$\mu g/m^3$')
    
    #plot_histogram(dY_national, 'National Annual Mortality Incidence Rate Reduction', 'per 100,000', 'count')

    return dY_national

def generate_box_plots_notitle_1(dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list):
    
    pop_dict = main.init_population()
    #print(dY_national)
    
    plot_boxplots(dCin_national, '', scenario_list, r'$\mu g/m^3$')
    
    plot_boxplots(dY_national, '', scenario_list,'per 100,000')
    plot_boxplots(dM_national, '', scenario_list,'People')
    
    plot_boxplots(benefits_national, '', scenario_list,'Billion USD')
    plot_boxplots(costs_national, '', scenario_list,'Billion USD')
    plot_boxplots(net_benefits_national, '', scenario_list,'Billion USD')
    
    #Total Cost Benefit 
    cost_benefit = [benefits_national, net_benefits_national, costs_national]  # List of datasets
    cost_benefit_group_labels  = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    plot_grouped_boxplots(cost_benefit, '', cost_benefit_group_labels, scenario_list, 'Billion USD')

    return dY_national

def generate_box_plots_notitle_pop(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list, zline = False):
    
    pop_dict = main.init_population()
    #print(dY_national)
    
    #plot_boxplots(dCin_national, '', scenario_list, r'$\mu g/m^3$')
    
    #plot_boxplots(dY_national, '', scenario_list,'per 100,000')
    plot_boxplots(dM_national, '', scenario_list,'Avoided Premature Mortality (deaths)')
    
    plot_boxplots(benefits_national, '', scenario_list,'Benefits (Billion USD)')
    plot_boxplots(costs_national, '', scenario_list,'Intervention Costs (Billion USD)')
    plot_boxplots(net_benefits_national, '', scenario_list,'Net Benefits (Billion USD)', zero_line=zline)
    
    #Total Cost Benefit 
    #cost_benefit = [benefits_national, net_benefits_national, costs_national]  # List of datasets
    #cost_benefit_group_labels  = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    #plot_grouped_boxplots(cost_benefit, '', cost_benefit_group_labels, scenario_list, 'Billion USD')

    return 0

def generate_box_plots_notitle_capita(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list, phi, occupy, zline = False):
    
    pop_dict = main.init_population()
    population = sum( pop_dict.values() )
    
    l = len(occupy)
    z = 0
    dy_list = []
    for z in range(l):
        dy = dY_national[z]
        dy = [x / phi[z] for x in dy]
        dy_list.append(dy) 
        
    plot_boxplots(dy_list, '', scenario_list,'Mortality Incidence Rate Reduction (per 100,000 resident)')
    
    z = 0
    benefits_residence = []
    for z in range(l):    
        br = [ (x * 1000000000) / (population * phi[z]) for x in benefits_national[z]]
        benefits_residence.append( br )
    plot_boxplots(benefits_residence, '', scenario_list,'Benefits (USD per resident)')
    
    z = 0
    cost_residence = []
    for z in range(l):    
        
        cr = [ occupy[z] * ((x * 1000000000) / (population * phi[z])) for x in costs_national[z]]

        cost_residence.append( cr )
    
    plot_boxplots(cost_residence, '', scenario_list,'Intervention Costs (USD per house)')
    
    z = 0
    net_benefits_residence = []
    for z in range(l):    
        nbr = [ (x * 1000000000) / (population * phi[z]) for x in net_benefits_national[z]]
        net_benefits_residence.append( nbr )
    
    plot_boxplots(net_benefits_residence, '', scenario_list,'Net Benefits (USD per resident)', zero_line=zline)
    
    return 0
        
#%%

def analysis_stats(data):
    # Calculate statistics
    avg = np.mean(data)
    mdn = np.median(data)
    minimum = np.min(data)
    maximum = np.max(data)
    p2_5 = np.percentile(data, 2.5)
    p97_5 = np.percentile(data, 97.5)
    s_std = np.std(data, ddof=1)
    
    # Print statistics
    print('median:', mdn)
    print('p2.5:', p2_5)
    print('p97.5:', p97_5)
    print('mean:', avg)
    print('min:', minimum)
    print('max:', maximum)
    print('Sample Standard Deviation:', s_std)
    
    # Return statistics in a dictionary
    return {
        'Median': mdn,
        '2.5 p': p2_5,
        '97.5 p': p97_5,
        'Mean': avg,
        'Min': minimum,
        'Max': maximum,
        'std' : s_std
    }

def all_stats(scenario_name, dC, dY, dM, benefit, cost, nb):
    # Prepare the DataFrame
    rows = []
    variable_names = ['dC', 'dY', 'dM', 'Benefit', 'Cost', 'Net Benefit', 'std']
    data_list = [dC, dY, dM, benefit, cost, nb]
    
    # Process each dataset
    for var_name, data in zip(variable_names, data_list):
        print(f'{var_name} ->')
        stats = analysis_stats(data)
        row = {
            'Scenario': scenario_name,
            'Variable': var_name,
            **stats
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Return the DataFrame
    return df
    
#%%
rR = 0.005826891
#rR = 1.073
n=1000
import main
import plot_functions

pop_dict_main = main.init_population()

c_mean, mort, ben, y = main.run_MCS(iterations =100)
dCin_national = []
for i in range(100):
    dCin_national.append(  main.national_average(c_mean[i], pop_dict_main) )



x = [ 'IECC V1', 'IECC V2', 'IECC V3', 'IECC All']
phi_IECC = [0.1, 0.33, 0.3, 0.73]
occupancy = [2.104, 2.1194, 2.197, 2.149]

generate_maps_IECC( dCin = c_mean, dY = c_mean , benefit = c_mean, cost = c_mean, nb = c_mean, phi_value= 0.73, occupy=2.149 )

#Plot Air Pol
pm_grid = main.start_centuryPM()
pm_grid_gdf = main.init_AQ_grids()
plot_functions.create_contour_plot_Air_Pol(pm_grid_gdf, pm_grid, 'C', r'', r'Ambient $PM_{2.5}$ Daily Mean Levels ($\mu g/m^3$)', color ='red')

plot_functions.create_contour_plot_Air_Pol(pm_grid_gdf, c_mean, 'C', r'', r'Ambient $PM_{2.5}$ Daily Mean Levels ($\mu g/m^3$)', color ='red')


#%%
import main
emissions = [ 0.11, 0.24, 0.43, 0.62]
ces = [ 0.5, 0.75, 0.9]

def init_rangehood(in_dir='Data/rangehoods.csv'):
    """
    Reads a CSV file using Pandas and converts each row to a tuple.
    
    Parameters:
    - filename (str): The path to the CSV file.
    
    Returns:
    - list of tuples: A list where each tuple contains three elements from one row of the CSV.
    """
    df = pd.read_csv(in_dir)
    tuple_list = [tuple(x) for x in df.to_numpy()]
    return tuple_list

hoods = init_rangehood()

results = {}

for hood in hoods:
    hood_name, hood_flow = hood
    hood_flow *= 1.69901082
    
    
    ce_level_dict = {}
    
    for ce in ces:
        
        #hood flow from CFM to m3/hr
        hood_CE = ce
        dc_list = []
        
        for emission in emissions:
            cooking_parameters = ( 0.042, emission * 1440000, hood_flow, hood_CE)
            
            dCin, dM, benefits, dY = main.run_MCS( cooking_params = cooking_parameters, iterations = 100)
            dCin_national, dM_national, dY_national, benefits_national = aggregate_results( dCin, dM, benefits, dY, n = 100)
            
            dc_list.append(dCin_national)
        ce_level_dict[ce] = dc_list
            
    results[hood_name] = ce_level_dict        
    
    
    
def plot_emissions_results_org(emissions, results):
    """
    Plots a scatter plot with emissions on the x-axis and results on the y-axis,
    distinguishing different hoods by marker and Capture Efficiencies by color.

    Parameters:
    - emissions: List of emissions levels (numerical).
    - results: Dictionary with hood names as keys and nested dictionaries as values,
               where inner dicts' keys are Capture Efficiencies and values are lists of lists
               containing 100 Monte Carlo simulation results.
    """
    plt.figure(figsize=(12, 8))  # Set the figure size for the plot
    ax = plt.gca()  # Get current axes
    
    # Setup for multiple legends
    markers = ['x', '*', '2', '+', 'o', 'v']  # Define a list of markers for differentiation
    ce_levels = [ 0.5, 0.75, 0.9]  # Capture Efficiency levels
    hoods = list(results.keys())
    color_map = plt.cm.get_cmap('viridis', len(ce_levels))  # Get a colormap based on CE levels
    #color_dict = {ce: color_map(i) for i, ce in enumerate(ce_levels)}  # Map CE to colors
    color_dict = {0.5: 'brown', 0.75: 'dodgerblue', 0.9: 'darkviolet'}

    # Plotting data
    lines = []  # To hold references for legends
    for hood_index, (hoodname, ces) in enumerate(results.items()):
        marker = markers[hood_index % len(markers)]  # Cycle through markers
        for ce, simulations in ces.items():
            # Calculate average of 100 Monte Carlo simulations for each emissions level
            avg_results = [np.mean(mc_result) for mc_result in simulations]
            # Create scatter plot for each hood at each CE
            line = plt.scatter(emissions, avg_results, label=f"{hoodname} CE: {ce}", 
                               marker=marker, color=color_dict[ce], s = 50, alpha=0.7)
            lines.append((line, f"{hoodname} (CE: {ce})"))
    
    # Adding legends and customizing plot
    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Add red dashed line at Y=0

    plt.axhline(3.3, color='blue', linestyle='--', linewidth=2)
    ax.text(0.35, 3, 'No indoor sources scenario', fontsize=24, color='blue', ha='center')  # Adjust '0.5' as needed for the x-position
    # Creating two legends
    # Legend for Hoods
    #hood_legend = plt.legend([line[0] for line in lines], [line[1] for line in lines], 
    #                         title='Hood (CE)', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    hood_legend = plt.legend([plt.Line2D([0], [0], color='black', marker= markers[hoods.index(hood) % len(markers)] , linestyle='') 
                            for hood in hoods], hoods, title='Range Hood Flow',
                            loc='upper left', bbox_to_anchor=(1, 1), fontsize=24, title_fontsize=24)
    
    # Legend for CE (capture efficiency)
    ce_legend = plt.legend([plt.Line2D([0], [0], color=color_dict[ce], marker='o', linestyle='') 
                            for ce in ce_levels], ce_levels, title='Capture Efficiency',
                            loc='lower left', bbox_to_anchor=(1, 0), fontsize=24, title_fontsize=24)
    plt.gca().add_artist(hood_legend)  # Add the first legend back after adding the second
    
    plt.ylabel(r'Improvement in Indoor $PM_{2.5}$ ($\mu g/m^3$)', fontsize=24)
    plt.xlabel(r'Cooking Emissions $(mg/min)$', fontsize=24)
    
    plt.tick_params(axis='both', labelsize=24)
    
    plt.grid(False)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig("Figures/Indoor.png", dpi=300)
    plt.show()

plot_emissions_results_org(emissions, results)


def plot_emissions_results(emissions, results):
    """
    Plots a scatter plot with emissions on the x-axis and results on the y-axis,
    distinguishing different hoods by marker and Capture Efficiencies by color.

    Parameters:
    - emissions: List of emissions levels (numerical).
    - results: Dictionary with hood names as keys and nested dictionaries as values,
               where inner dicts' keys are Capture Efficiencies and values are lists of lists
               containing 100 Monte Carlo simulation results.
    """
    plt.figure(figsize=(8, 6))  # Set the figure size for the plot
    ax = plt.gca()  # Get current axes

    # Define markers and colors
    markers = ['x', '*', '+', 'o', 'v', '^']  # Distinct markers for each hood
    ce_levels = [0.25, 0.5, 0.75, 0.9]  # Capture Efficiency levels
    color_dict = {'0.25': 'gray', '0.5': 'brown', '0.75': 'dodgerblue', '0.9': 'darkviolet'}

    # Plotting data
    hood_legend_items = []  # To track hood legend entries
    for hood_index, (hoodname, ces) in enumerate(results.items()):
        marker = markers[hood_index % len(markers)]  # Assign markers cyclically
        for ce, simulations in ces.items():
            # Compute the average results
            avg_results = [np.mean(sim) for sim in simulations]
            line = plt.scatter(emissions, avg_results, label=f"{hoodname} CE: {ce}",
                               marker=marker, color=color_dict[str(ce)], s=50, alpha=0.7)
            if ce == 0.25:  # Only add each hood to the legend once
                hood_legend_items.append(line)

    # Customizing all text elements to have font size 24
    plt.title('Emissions vs Results by Hood and CE', fontsize=24)
    plt.xlabel('Emissions ($mg/min$)', fontsize=24)
    plt.ylabel('Improvements in IAQ ($\mu g/m^3$)', fontsize=24)
    ax.tick_params(axis='both', labelsize=24)  # Set tick label size

    # Legends with font size 24
    hood_legend = plt.legend(handles=hood_legend_items, title='Range Hoods', loc='upper left', bbox_to_anchor=(1, 1), fontsize=24, title_fontsize=24)
    ax.add_artist(hood_legend)  # Add hood legend back
    ce_legend = plt.legend([plt.Line2D([0], [0], color=color_dict[str(ce)], marker='s', linestyle='') 
                            for ce in ce_levels], [f"CE {ce}" for ce in ce_levels],
                            title='Capture Efficiency', loc='lower left', bbox_to_anchor=(1, 0), fontsize=24, title_fontsize=24)

    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Horizontal line at y=0
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    
plot_emissions_results_org(emissions, results)


#%%

def sde_by_side(title1, title2): 
    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    
    # Load images
    path_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Plots\Final Theses Plots\Results\Side_By_Side\\'
    path1 = path_dir + title1 +'.png'
    path2 = path_dir + title2 +'.png'
    
    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)
    
    # Display images
    ax[0].imshow(img1)
    ax[0].axis('off')  # Turn off axis
    
    ax[1].imshow(img2)
    ax[1].axis('off')  # Turn off axis
    

    # Adjust layout to make plots closer together
    plt.tight_layout(pad=0.5)  # Adjust padding between and around subplots
    
    # Further adjust subplot parameters if needed
    fig.subplots_adjust(wspace=0, hspace=0)  # Adjust the space between the images
    

    plt.show()