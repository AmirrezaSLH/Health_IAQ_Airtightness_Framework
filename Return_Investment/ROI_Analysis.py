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

def plot_boxplots(data, title, xlabels, ylabel, colors='skyblue', save=False, filename='boxplot.png', zero_line = False, zero = False):
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
    plt.title(title, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    
    # Setting x-axis labels
    plt.xticks(range(1, len(data) + 1), xlabels, fontsize=24)
    plt.yticks(fontsize=24)
    if zero == True:
        plt.ylim(bottom=0)
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
    
def plot_grouped_boxplots_IAQ(data, title, subplot_labels, group_labels, ylabel, colors=('#AD84C6','skyblue'), save=False, filename='Figures/boxplot.png'):
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
    group_gap = 0.5  # Increased gap between groups for clarity
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
    plt.title(title, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    
    # Custom x-axis labels
    ax = plt.gca()
    tick_positions = [np.mean(positions[i:i+num_subgroups]) for i in range(0, len(positions), num_subgroups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, fontsize=24)

    # Add gridlines for better readability
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5)
    ax.tick_params(axis='y', labelsize=24)
    # Add legend for subgroups
    plt.legend([box["boxes"][i] for i in range(0, num_subgroups)], subplot_labels, loc='lower right', fontsize=24)

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

def ach_reduction_results( ach_before_list, ach_intervention_list, n):
    
    ach_reduction_list = []
    for i in range(n):
        ach_reduction_dict = {}
        ach_before = ach_before_list[i]
        ach_intervention = ach_intervention_list[i]
#        print(ach_before)
        for g in ach_before.keys():
            #print(g)
            #print(ach_before[g]['SF_V1'])
            ach_r_v1 = (( ach_before[g]['SF_V1'] - ach_intervention[g]['SF_V1'] ) / ach_before[g]['SF_V1'] ) * 100
            ach_r_v2 = (( ach_before[g]['SF_V2'] - ach_intervention[g]['SF_V2'] ) / ach_before[g]['SF_V2']  ) * 100
            ach_r_v3 = (( ach_before[g]['SF_V3'] - ach_intervention[g]['SF_V3'] ) / ach_before[g]['SF_V3']  ) * 100
            ach_r = (0.1 * ach_r_v1 + 0.33 * ach_r_v2 + 0.3 * ach_r_v3)/0.73
            ach_reduction_dict[g] = ach_r
        ach_reduction_list.append(ach_reduction_dict)
    return ach_reduction_list

def aggregate_results( dCin, dM, benefits, dY, costs, net_benefits, n):
    pop_dict = main.init_population()
    dCin_national = []
    dM_national = []
    dY_national = []
    benefits_national = []
    costs_national = []
    net_benefits_national = []

    for i in range(n):
        dCin_national.append(  main.national_average(dCin[i], pop_dict) )
        dM_national.append( sum(dM[i].values()) )
        dY_national.append ( main.national_average(dY[i], pop_dict) * 100000)
        costs_national.append( sum( costs[i].values()) / 1000000000)
        benefits_national.append( sum( benefits[i].values()) / 1000000000)
        net_benefits_national.append( sum(net_benefits[i].values()) / 1000000000 )
        
    return dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national

def generate_maps(dCin, dY, benefit, cost, nb ):
    
    dY_mean = calculate_spatial_mean(dY)
    
    pop_dict = main.init_population()
    
    benefit_mean = calculate_spatial_mean(benefit)
    benefit_capita = {key : value/pop_dict[key] for key, value in benefit_mean.items()}
    
    cost_mean = calculate_spatial_mean(cost)
    cost_capita = {key : value/pop_dict[key] for key, value in cost_mean.items()}
    
    nb_mean = calculate_spatial_mean(nb)
    nb_capita = {key : value/pop_dict[key] for key, value in nb_mean.items()}
    
    
    
    #cost_capita_mean = calculate_spatial_mean(cost_capita)
    #nb_capita_mean = calculate_spatial_mean(nb_capita)
    dCin_mean = calculate_spatial_mean(dCin)
    
    aq_grid_gdf = main.init_AQ_grids()
    legend_ticks = [5, 15, 25, 35, 45, 55]
    plot_functions.create_contour_plot_dy(aq_grid_gdf, dY_mean, 'dY', 'Annual Mortality Incidence Rate Reduction', 'Per 100,000 people', color ='blue')
    plot_functions.create_contour_plot_dC(aq_grid_gdf, dCin_mean, 'dC', 'Average Residence $PM_2.5$ Reduction', r'$\mu g/m^3$', color ='blue')
    
    plot_functions.create_contour_plot(aq_grid_gdf, benefit_capita, 'bc', 'Benefit per Capita', 'USD', color ='blue')
    plot_functions.create_contour_plot(aq_grid_gdf, cost_capita, 'cc', 'Cost per Capita', 'USD', color ='blue')
    plot_functions.create_contour_plot_negetive(aq_grid_gdf, nb_capita, 'nbc', 'Net Benefit per Capita', 'USD', color ='red_blue')
    
    return 0

def generate_maps_IECC(dCin, dY, benefit, cost, nb, phi_value, occupy ):
    
    pop_dict = main.init_population()
    aq_grid_gdf = main.init_AQ_grids()

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

    dCin_mean = calculate_spatial_mean(dCin)
    dCin_adjusted = {key : value/phi_value for key, value in dCin_mean.items()}
    plot_functions.create_contour_plot_dC(aq_grid_gdf, dCin_adjusted, 'dC', '', r'Reduction in Outdoor-origin $PM_{2.5}$ ($\mu g/m^3$)', color ='blue')

    return 0

def generate_ach_reduction_map(ach_reduction_list):
    aq_grid_gdf = main.init_AQ_grids()
    ach_reduction_mean = calculate_spatial_mean(ach_reduction_list)
    plot_functions.create_contour_plot_ach_reduction(aq_grid_gdf, ach_reduction_mean, 'ach_reduction', '', 'Air Leakage Reduction (%)', color ='blue')

    
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

def generate_box_plots_notitle_pop(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list, zline = False):
    
    pop_dict = main.init_population()
    #print(dY_national)
    
    #plot_boxplots(dCin_national, '', scenario_list, r'$\mu g/m^3$')
    
    #plot_boxplots(dY_national, '', scenario_list,'per 100,000')
    plot_boxplots(dM_national, '', scenario_list,'Avoided Premature Mortality (deaths)', zero = True)
    
    plot_boxplots(benefits_national, '', scenario_list,'Benefits (Billion USD)')
    plot_boxplots(costs_national, '', scenario_list,'Intervention Costs (Billion USD)', zero = True)
    plot_boxplots(net_benefits_national, '', scenario_list,'Net Benefits (Billion USD)', zero_line=zline)
    
    #Total Cost Benefit 
    #cost_benefit = [benefits_national, net_benefits_national, costs_national]  # List of datasets
    #cost_benefit_group_labels  = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    #plot_grouped_boxplots(cost_benefit, '', cost_benefit_group_labels, scenario_list, 'Billion USD')

    return 0

def generate_box_plots_notitle_pop_sensitivity(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list, phi_value, zline = False):
    
    z = 0
    dC_list = []
    for z in range(5):
        dC = dCin_national[z]
        dC = [x / phi_value for x in dC]
        dC_list.append(dC) 
    plot_boxplots(dC_list, '', scenario_list, r'Reduction in Outdoor-origin $PM_{2.5}$ ($\mu g/m^3$)')

    plot_boxplots(dM_national, '', scenario_list,'Avoided Premature Mortality (deaths)')
    
    plot_boxplots(benefits_national, '', scenario_list,'Benefits (Billion USD)')
    plot_boxplots(costs_national, '', scenario_list,'Intervention Costs (Billion USD)')
    plot_boxplots(net_benefits_national, '', scenario_list,'Net Benefits (Billion USD)', zero_line=zline)

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
        
    plot_boxplots(dy_list, '', scenario_list,'per 100,000 resident', zero = True, save = True, filename="Figures/indcidence.png")
    
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
    
    plot_boxplots(cost_residence, '', scenario_list,'Intervention Costs (USD per house)', zero = True)
    
    z = 0
    net_benefits_residence = []
    for z in range(l):    
        nbr = [ (x * 1000000000) / (population * phi[z]) for x in net_benefits_national[z]]
        net_benefits_residence.append( nbr )
    
    plot_boxplots(net_benefits_residence, '', scenario_list,'Net Benefits (USD per resident)', zero_line=zline)
    
    return 0
        
def scenario_UC1(re_risk, int_func, itr, hf):
    dCin, dM, benefits, dY, costs, net_benefits, finf0, finf1, achn0, achn1, sfd, sfb, sfi = main.run_MCS(  intervention_function = int_func, rr_beta = re_risk , iterations = itr, HF = hf)
    dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national = aggregate_results( dCin, dM, benefits, dY, costs, net_benefits, n)
    
    return dCin, dM, benefits, dY, costs, net_benefits, dCin_national, dM_national, benefits_national, dY_national, costs_national, net_benefits_national, finf0, finf1, achn0, achn1, sfd, sfb, sfi

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


def payback_period( benefits_list, costs_list, airtightness_decay_rate):
    
    
    payback_period_list = []
    
    for i in range( len(benefits_list) ):
        z = 1
        benefit = benefits_list[i]
        cost = costs_list[i]
        
        net_benefit = benefit - cost
        while net_benefit <= 0:
            if z<=3:
                net_benefit += benefit * ( (1-airtightness_decay_rate)**z )
            else:
                net_benefit += benefit * ( (1-airtightness_decay_rate)**3 )
            z += 1
            print(i, ' : ',z, ' : ', net_benefit, ' : ', benefit, ' : ', cost)
        payback_period_list.append(z)
        print(i)
        
    return payback_period_list

def payback_period_continuous( benefits_list, costs_list, airtightness_decay_rate, airtightness_decay_len = 3):
    
    
    payback_period_list = []
    z_max = airtightness_decay_len
    for i in range( len(benefits_list) ):
        z = 1
        
        benefit = benefits_list[i]
        cost = costs_list[i]
        
        net_benefit = benefit - cost
        while net_benefit <= -cost:
            if z<=z_max:
                net_benefit += benefit * ( (1-airtightness_decay_rate)**z )
            else:
                net_benefit += benefit * ( (1-airtightness_decay_rate)**3 )
            z += 1
            print( ' : ',z, ' : ', net_benefit, ' : ', benefit, ' : ', cost)
        z += (-net_benefit)/benefit
        print( ' : ',z, ' : ', net_benefit, ' : ', benefit, ' : ', cost)
        payback_period_list.append(z)
        print(i)
        
    return payback_period_list

def plot_violin(data_lists, labels=None, plot_title='Violin Plot', x_label='Categories', y_label='Values'):
    """
    Creates a violin plot for multiple data lists.

    Parameters:
    - data_lists: List of lists, each containing numerical data.
    - labels: List of strings, labels for each data list. Defaults to None.
    - plot_title: String, the title of the plot. Defaults to 'Violin Plot'.
    - x_label: String, label for the x-axis. Defaults to 'Categories'.
    - y_label: String, label for the y-axis. Defaults to 'Values'.
    """
    
    data_lists = [
        np.array(data)[(np.array(data) >= np.percentile(data, 2.5)) & (np.array(data) <= np.percentile(data, 97.5))]
        for data in data_lists
    ]
    
    
    plt.figure(figsize=(10, 6))  # Set the figure size for the plot
    ax = plt.gca()  # Get current axes

    # Create the violin plot
    parts = ax.violinplot(data_lists, widths = 0.7,showmeans=False, showmedians=True)

    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)
        
    # Coloring each violin plot
    for pc in parts['bodies']:
        pc.set_linewidth(1)
        pc.set_facecolor('#AD84C6')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # Adding custom labels to the x-axis if provided
    if labels:
        plt.xticks(range(1, len(labels) + 1), labels)
        
    # Setting labels and title
    plt.xlabel(x_label)
    
    # Add titles and labels
    plt.title(plot_title, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.xlabel(x_label, fontsize=24)
    
    ax.tick_params(axis='both', labelsize=24)
    # Grid for better readability
    plt.grid(False)

    plt.savefig("Figures/violin.png", dpi=300)
    # Display the plot
    plt.show()


#%%
rR = 0.005826891
#rR = 1.073
n=1000
import main

pop_dict_main = main.init_population()

sdCin1, sdM1, sbenefits1, sdY1, scosts1, snet_benefits1, delta_mean_C_in1, dMort1, benefit1, dY1, C1, nb1, f0_1,f1_1,an0_1,an1_1, sfd1, sfb1, sfi1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
sdCin2, sdM2, sbenefits2, sdY2, scosts2, snet_benefits2, delta_mean_C_in2, dMort2, benefit2, dY2, C2, nb2, f0_2,f1_2,an0_2,an1_2, sfd2, sfb2, sfi2 = scenario_UC1( re_risk = rR, int_func = code_comply_V1, itr = n, hf = 'HF_3')
sdCin3, sdM3, sbenefits3, sdY3, scosts3, snet_benefits3, delta_mean_C_in3, dMort3, benefit3, dY3, C3, nb3, f0_3,f1_3,an0_3,an1_3, sfd3, sfb3, sfi3 = scenario_UC1( re_risk = rR, int_func = code_comply_V2, itr = n, hf = 'HF_3')
sdCin4, sdM4, sbenefits4, sdY4, scosts4, snet_benefits4, delta_mean_C_in4, dMort4, benefit4, dY4, C4, nb4, f0_4,f1_4,an0_4,an1_4, sfd4, sfb4, sfi4 = scenario_UC1( re_risk = rR, int_func = code_comply_V3, itr = n, hf = 'HF_3')

sfd1_list = process_sf_conc(sfd1)

t0 = [ -x for x in (sfd1_list[0]) ]
t1 = [ -x for x in (sfd1_list[1]) ]
t2 = [ -x for x in (sfd1_list[2]) ]

sfd1_list = (t0, t1, t2)
sfd1_stat = analysis_stats(sfd1_list[0])
sfd2_stat = analysis_stats(sfd1_list[1])
sfd3_stat = analysis_stats(sfd1_list[2])

sfb1_list = process_sf_conc(sfb1)

tp0 = [ x for x in (sfb1_list[0]) ]
tp1 = [ x for x in (sfb1_list[1]) ]
tp2 = [ x for x in (sfb1_list[2]) ]

sfb11_list = (tp0, tp1, tp2)
sf_bx_list = [sfb1_list, sfd1_list]
sfi1_list = process_sf_conc(sfi1)

tpp0 = [ x for x in (sfi1_list[0]) ]
tpp1 = [ x for x in (sfi1_list[1]) ]
tpp2 = [ x for x in (sfi1_list[2]) ]
sfi1_list = (tpp0, tpp1, tpp2)
sf_bx2_list = [sfb1_list, sfi1_list]
plot_grouped_boxplots_IAQ(sf_bx2_list, '', ['Baseline', 'Alternative'], ['Vintage 1', 'Vintage 2', 'Vintage 3'], r'Indoor $PM_{2.5}$ of Outdoor-origin ($\mu g/m^3$)', save=True)

analysis_stats(delta_mean_C_in2)

df1 = all_stats('ALL', delta_mean_C_in1, dY1, dMort1, benefit1, C1, nb1)
df2 = all_stats('V1', delta_mean_C_in2, dY2, dMort2, benefit2, C2, nb2)
df3 = all_stats('V2L', delta_mean_C_in3, dY3, dMort3, benefit3, C3, nb3)
df4 = all_stats('V3', delta_mean_C_in4, dY4, dMort4, benefit4, C4, nb4)
merged_df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
path = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Analysis\IECC_results.csv'
merged_df.to_csv(path)

p1_test = payback_period( benefit1, C1, 0.05)
p4_test = payback_period( benefit4, C4, 0.05)




delta_mean_C_in = [delta_mean_C_in2, delta_mean_C_in3, delta_mean_C_in4, delta_mean_C_in1]
dMort = [ dMort2, dMort3, dMort4, dMort1]
benefit = [ benefit2, benefit3, benefit4, benefit1]
dY = [ dY2, dY3, dY4, dY1]
C = [ C2, C3, C4, C1]
nb = [ nb2, nb3, nb4, nb1]

max(C1)
#ACH Reduction
ach_reduction_comply_all = ach_reduction_results( an0_1, an1_1, n)

ach_reduction_national_average = []
for i in range(n):
    ach_reduction_national_average.append(  main.national_average(ach_reduction_comply_all[i], pop_dict_main) )

print( mean(ach_reduction_national_average) )
generate_ach_reduction_map(ach_reduction_comply_all)

analysis_stats(benefit1)


x = [ 'IECC V1', 'IECC V2', 'IECC V3', 'IECC All']
phi_IECC = [0.1, 0.33, 0.3, 0.73]
occupancy = [2.104, 2.1194, 2.197, 2.149]
population_box_plots = generate_box_plots_notitle_pop(dCin_national =delta_mean_C_in, dM_national=dMort, dY_national=dY, costs_national=C, benefits_national=benefit, net_benefits_national=nb, n=n, scenario_list=x, zline = True)
capita_box_plots = generate_box_plots_notitle_capita(dCin_national =delta_mean_C_in, dM_national=dMort, dY_national=dY, costs_national=C, benefits_national=benefit, net_benefits_national=nb, n=n, scenario_list=x, phi = phi_IECC, occupy=occupancy, zline = True)



import plot_functions

generate_maps_IECC( dCin = sdCin1, dY = sdY1 , benefit = sbenefits1, cost = scosts1, nb = snet_benefits1, phi_value= 0.73, occupy=2.149 )

generate_maps_IECC( dCin = sdCin1, dY = sdY1 , benefit = sbenefits1, cost = scosts1, nb = snet_benefits2, phi_value= 0.1, occupy=2.1 )
generate_maps_IECC( dCin = sdCin1, dY = sdY1 , benefit = sbenefits1, cost = scosts1, nb = snet_benefits3, phi_value= 0.33, occupy=2.12 )
generate_maps_IECC( dCin = sdCin1, dY = sdY1 , benefit = sbenefits1, cost = scosts1, nb = snet_benefits4, phi_value= 0.3, occupy=2.2 )


#Plot Air Pol
pm_grid = main.start_centuryPM()
pm_grid_gdf = main.init_AQ_grids()
plot_functions.create_contour_plot_Air_Pol(pm_grid_gdf, pm_grid, 'C', r'', r'Ambient $PM_{2.5}$ Daily Mean Levels ($\mu g/m^3$)', color ='red')

#%% Payback Period IECC
'''
Payback Period IECC
'''

p1 = payback_period_continuous( benefit1, C1, 0.05)
p2 = payback_period_continuous( benefit2, C2, 0.05)
p3 = payback_period_continuous( benefit3, C3, 0.05)
p4 = payback_period_continuous( benefit4, C4, 0.05)

pay = [ p2, p3, p4, p1]

plot_boxplots(pay, '', x,'Intervention Payback Period (Year)', zero = True)

print(p2)
print( sum(p3) / len(p3))
min(benefit1)
max(benefit1)

plot_violin(data_lists = pay, labels=x, plot_title='', x_label='', y_label='Payback Period (Year)')

#%% Sensitivity Analysis
'''
Sensitivity Analysis
'''

def reduction_energy_star( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.75
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.75
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.75
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention


def reduction_20( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.8
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.8
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.8
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

def reduction_40( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.6
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.6
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.6
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

def reduction_60( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.4
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.4
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.4
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention


n = 1000
rR = 0.005826891
#spatial concentration, spatial mortality, spatial benefit, spatial dY, spatial cost, spation net benefit, national concentration, national benefit, national dY, national costs, national net benefit, spatial F0, Spatial F1, Spatial ACH0, Spatial ACH1, spatial concentration change
spatial_delta_concentrationS2, spatial_delta_mortalityS2, spatial_benefitsS2, spatial_delta_yS2, spatial_costsS2, spatial_net_benefitsS2, national_avg_concentration_changeS2, national_delta_mortalityS2, national_benefitsS2, national_delta_yS2, national_costsS2, national_net_benefitsS2, spatial_filter0_S2, spatial_filter1_S2, spatial_ach0_S2, spatial_ach1_S2, spatial_feature_deltaS2, _, _ = scenario_UC1(re_risk=rR, int_func=reduction_20, itr=n, hf='HF_3')
spatial_delta_concentrationS3, spatial_delta_mortalityS3, spatial_benefitsS3, spatial_delta_yS3, spatial_costsS3, spatial_net_benefitsS3, national_avg_concentration_changeS3, national_delta_mortalityS3, national_benefitsS3, national_delta_yS3, national_costsS3, national_net_benefitsS3, spatial_filter0_S3, spatial_filter1_S3, spatial_ach0_S3, spatial_ach1_S3, spatial_feature_deltaS3, _, _ = scenario_UC1(re_risk=rR, int_func=reduction_energy_star, itr=n, hf='HF_3')
spatial_delta_concentrationS4, spatial_delta_mortalityS4, spatial_benefitsS4, spatial_delta_yS4, spatial_costsS4, spatial_net_benefitsS4, national_avg_concentration_changeS4, national_delta_mortalityS4, national_benefitsS4, national_delta_yS4, national_costsS4, national_net_benefitsS4, spatial_filter0_S4, spatial_filter1_S4, spatial_ach0_S4, spatial_ach1_S4, spatial_feature_deltaS4, _, _ = scenario_UC1(re_risk=rR, int_func=reduction_40, itr=n, hf='HF_3')
spatial_delta_concentrationS5, spatial_delta_mortalityS5, spatial_benefitsS5, spatial_delta_yS5, spatial_costsS5, spatial_net_benefitsS5, national_avg_concentration_changeS5, national_delta_mortalityS5, national_benefitsS5, national_delta_yS5, national_costsS5, national_net_benefitsS5, spatial_filter0_S5, spatial_filter1_S5, spatial_ach0_S5, spatial_ach1_S5, spatial_feature_deltaS5, _, _ = scenario_UC1(re_risk=rR, int_func=reduction_60, itr=n, hf='HF_3')

analysis_stats(national_delta_mortalityS2)

analysis_stats(national_avg_concentration_changeS2)
analysis_stats(delta_mean_C_in1)

sc2 = all_stats('R20', national_avg_concentration_changeS2, national_delta_yS2, national_delta_mortalityS2, national_benefitsS2, national_costsS2, national_net_benefitsS2)
sc3 = all_stats('R25', national_avg_concentration_changeS3, national_delta_yS3, national_delta_mortalityS3, national_benefitsS3, national_costsS3, national_net_benefitsS3)
sc4 = all_stats('R40', national_avg_concentration_changeS4, national_delta_yS4, national_delta_mortalityS4, national_benefitsS4, national_costsS4, national_net_benefitsS4)
sc5 = all_stats('R60', national_avg_concentration_changeS5, national_delta_yS5, national_delta_mortalityS5, national_benefitsS5, national_costsS5, national_net_benefitsS5)

merged_df = pd.concat([sc2, sc3, sc4, sc5], axis=0, ignore_index=True)
path = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Analysis\Sensitivity_results.csv'
merged_df.to_csv(path)


delta_mean_C_in = [ [-rc for rc in national_avg_concentration_changeS2], [-rc for rc in national_avg_concentration_changeS3], [-rc for rc in national_avg_concentration_changeS4], [-rc for rc in national_avg_concentration_changeS5], [-rc for rc in delta_mean_C_in1] ]
dMort = [ national_delta_mortalityS2, national_delta_mortalityS3, national_delta_mortalityS4, national_delta_mortalityS5, dMort1]
benefit = [ national_benefitsS2, national_benefitsS3, national_benefitsS4, national_benefitsS5, benefit1]
dY = [national_delta_yS2, national_delta_yS3, national_delta_yS4, national_delta_yS5, dY1]
C = [ national_costsS2, national_costsS3, national_costsS4, national_costsS5, C1]
nb = [national_net_benefitsS2, national_net_benefitsS3, national_net_benefitsS4, national_net_benefitsS5, nb1]
 
x = [ 'R20', 'Energy Star (R25)', 'R40', 'R60', 'IECC ALL']

phi_IECC = [0.73, 0.73, 0.73, 0.73, 0.73]
occupancy = [2.149, 2.149, 2.149, 2.149, 2.149]
population_box_plots = generate_box_plots_notitle_pop_sensitivity(dCin_national =delta_mean_C_in, dM_national=dMort, dY_national=dY, costs_national=C, benefits_national=benefit, net_benefits_national=nb, n=n, scenario_list=x, phi_value=0.73, zline = True)
capita_box_plots = generate_box_plots_notitle_capita(dCin_national =delta_mean_C_in, dM_national=dMort, dY_national=dY, costs_national=C, benefits_national=benefit, net_benefits_national=nb, n=n, scenario_list=x, phi = phi_IECC, occupy=occupancy, zline = True)

import plot_functions
generate_maps_IECC( dCin = spatial_delta_concentrationS2, dY = spatial_delta_yS2 , benefit = spatial_benefitsS2, cost = spatial_costsS2, nb = spatial_net_benefitsS2, phi_value= 0.73, occupy=2.149 )
generate_maps_IECC( dCin = spatial_delta_concentrationS2, dY = spatial_delta_yS2 , benefit = spatial_benefitsS2, cost = spatial_costsS2, nb = spatial_net_benefitsS4, phi_value= 0.73, occupy=2.149 )
generate_maps_IECC( dCin = spatial_delta_concentrationS2, dY = spatial_delta_yS2 , benefit = spatial_benefitsS2, cost = spatial_costsS2, nb = spatial_net_benefitsS5, phi_value= 0.73, occupy=2.149 )

#%% Payback Period IECC
'''
Payback Period IECC
'''

pS2 = payback_period_continuous( national_benefitsS2, national_costsS2, 0.05)
pS3 = payback_period_continuous( national_benefitsS3, national_costsS3, 0.05)
pS4 = payback_period_continuous( national_benefitsS4, national_costsS4, 0.05)
pS5 = payback_period_continuous( national_benefitsS5, national_costsS5, 0.05)

payS = [ pS2, pS3, pS4, pS5, p1]
plot_boxplots(payS, '', x,'Intervention Payback Period (Year)', zero = True)

plot_violin(data_lists = payS, labels=x, plot_title='Intervention Payback Period', x_label='Vintage', y_label='Year')
