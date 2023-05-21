import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################################# SETTINGS #################################

fontdict={
    'family':'Tahoma',
    'color':'black',
    'weight':'semibold',
    'size': 12
}

# Dictionaries for differentation
color_dict = {
    'SimpleFill': '#1f77b4',  # muted blue
    'KNN': '#ff7f0e',  # safety orange
    'SoftImpute': '#2ca02c',  # cooked asparagus green
    'IterativeImputer': '#d62728',  # brick red
    'IterativeSVD': '#9467bd',  # muted purple
    'MatrixFactorization': '#8c564b',  # chestnut brown
}
line_style_dict = {
    'SimpleFill': '-',  
    'KNN': '--',  
    'SoftImpute': '-.',  
    'IterativeImputer': ':',
    'IterativeSVD': '-',  
    'MatrixFactorization': '--',  
}
marker_style_dict = {
    'SimpleFill': 'o',  
    'KNN': 'v',  
    'SoftImpute': '^',  
    'IterativeImputer': '<',
    'IterativeSVD': '>',  
    'MatrixFactorization': 's',  
}

small_error_ticks    = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
medium_error_ticks   = [0, 0.075, 0.15, 0.225, 0.3, 0.375]
large_error_ticks    = [0, 0.10, 0.20, 0.30, 0.40, 0.50]

mv_proportions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


################################# MISSING VALUE TYPE - METHODS #################################

def generate_mv_type_bar_plots(results, dataset_name, metric):
    for mv_type in results[dataset_name].keys():

        methods = []
        values = []

        for method in results[dataset_name][mv_type].keys():
            methods.append(method)
            values.append(results[dataset_name][mv_type][method][metric])
        
        # Sort by values
        methods = [method for _, method in sorted(zip(values, methods))]
        values = sorted(values)
        
        plt.figure(figsize=(10, 6))
        
        # Use different color for each bar
        colors = [color_dict[method] for method in methods]
        bars = plt.barh(methods, values, color=colors)
        
        # Add the exact values at the end of each bar
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.3f}', 
                        va='center', ha='left')

        plt.xlabel(metric, fontdict=fontdict)
        plt.title(f'{dataset_name} - {mv_type}', fontdict=fontdict)
        plt.xticks(large_error_ticks)
        plt.show()
                

def generate_mv_type_combined_bar_plots(results, metric):
    mv_type_results = {}

    # Combine the values of all datasets
    for dataset_name in results.keys():
        for mv_type in results[dataset_name].keys():
            if mv_type not in mv_type_results:
                mv_type_results[mv_type] = {}
            for method in results[dataset_name][mv_type].keys():
                if method not in mv_type_results[mv_type]:
                    mv_type_results[mv_type][method] = []
                mv_type_results[mv_type][method].append(results[dataset_name][mv_type][method][metric])

    for mv_type, results in mv_type_results.items():
        # Calculate the average over the datasets
        methods = []
        avg_values = []

        for method, values in results.items():
            methods.append(method)
            avg_values.append(np.mean(values))

        # Sort by avg_values
        methods = [method for _, method in sorted(zip(avg_values, methods))]
        avg_values = sorted(avg_values)

        plt.figure(figsize=(10, 6))

        # Use different color for each bar
        colors = [color_dict[method] for method in methods]
        bars = plt.barh(methods, avg_values, color=colors)

        # Add the exact values at the end of each bar
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.3f}', 
                        va='center', ha='left')

        plt.xlabel(metric, fontdict=fontdict)
        plt.title(f'Combined results - {mv_type}', fontdict=fontdict)
        plt.xticks(large_error_ticks)
        plt.show()


        
################################# MISSING VALUE PROPORTIONS - METHODS #################################

def generate_mv_proportion_bar_plots(results, dataset_name, metric):

    # Get unique imputers and percentages from the results
    imputers = list(set(imputer for percentage in results[dataset_name] for imputer in results[dataset_name][percentage]))
    
    percentages = sorted(list(results[dataset_name].keys()))

    # Create a DataFrame to hold results
    df = pd.DataFrame(index=imputers, columns=percentages)
    
    # Fill DataFrame with results
    for percentage in percentages:
        for imputer in imputers:
            df.loc[imputer, percentage] = results[dataset_name][percentage].get(imputer, {}).get(metric, None)
    
    # Plotting each imputer
    for imputer_name in df.index:
        imputer_data = df.loc[imputer_name].tolist()
        plt.plot(df.columns, imputer_data, 
                 label = imputer_name, 
                 linestyle=line_style_dict[imputer_name],  
                 marker=marker_style_dict[imputer_name], 
                 linewidth=3, 
                 color=color_dict[imputer_name])

    max_value = df.max().max()
    error_ticks = large_error_ticks
    if max_value < max(medium_error_ticks): error_ticks = medium_error_ticks
    if max_value < max(small_error_ticks): error_ticks = small_error_ticks

    plt.xlabel('Missing Value Proportion', fontdict=fontdict)
    plt.ylabel(metric, fontdict=fontdict)
    plt.legend(loc='upper left')
    plt.yticks(error_ticks)
    plt.xticks(percentages)
    plt.title(dataset_name, fontdict=fontdict)
    plt.grid(visible=True)
    plt.show()



def generate_mv_proportion_combined_bar_plots(results, metric, figsize):

    combined_df = None
    dataset_names = list(results.keys())
    
    # Get unique imputers and percentages from the results
    imputers = list(set(imputer for dataset in results for percentage in results[dataset] for imputer in results[dataset][percentage]))

    percentages = sorted(list(results[dataset_names[0]].keys()))

    for dataset_name in dataset_names:
        # Create a DataFrame to hold results
        df = pd.DataFrame(index=imputers, columns=percentages)

        # Fill DataFrame with results
        for percentage in percentages:
            for imputer in imputers:
                df.loc[imputer, percentage] = results[dataset_name][percentage].get(imputer, {}).get(metric, None)

        if combined_df is None:
            combined_df = df
        else:
            combined_df += df

    # Calculate average
    combined_df = combined_df / len(dataset_names)
    percentageList = combined_df.columns.tolist()

    plt.figure(figsize=figsize)

    for imputer_name in combined_df.index:
        imputer_data = combined_df.loc[imputer_name].tolist()
        plt.plot(percentageList, imputer_data, label=imputer_name, linewidth=3, 
        color=color_dict[imputer_name], linestyle=line_style_dict[imputer_name], marker=marker_style_dict[imputer_name], )

    plt.xlabel('Missing Value Proportion', fontdict=fontdict)
    plt.ylabel(metric, fontdict=fontdict)
    plt.legend(loc='upper left')
    plt.xticks(percentages)
    plt.title('Combined results', fontdict=fontdict)
    plt.grid(visible=True)
    plt.show()

