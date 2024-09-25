import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gmean

# Replace 'your_file.csv' with the path to your actual CSV file
file_path = 'results_p23.txt'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# df = df[df['Epoch'] == 5000]

# Group by 'q' and 'n', and calculate the median or geometric mean of L2 error
def agg_func(x):
    return gmean(x)  # Change to gmean(x) for geometric mean

grouped_df = df.groupby(['q', 'n']).agg({'L2 error': agg_func, 'h': 'first'}).reset_index()

# Get the unique 'q' values
q_values = grouped_df['q'].unique()

# Choose a colormap
colormap = plt.get_cmap('plasma', len(q_values))  # Change 'tab20' to another colormap if needed
colors = [colormap(i) for i in np.linspace(0, 1, len(q_values))]

# Create a PDF file to save the plots
with PdfPages('error_vs_h_grouped_by_q_with_aggregation_and_regression.pdf') as pdf:
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Iterate through each 'q' value and plot them
    for i, q_value in enumerate(q_values):
        # Filter the DataFrame for the given 'q' value
        filtered_df = grouped_df[grouped_df['q'] == q_value]

        # Scatter plot of the aggregated L2 error
        x = filtered_df['h']
        y = filtered_df['L2 error']
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'q = {q_value}')

        # Fit a regression line in log-log space
        log_x = np.log10(x)
        log_y = np.log10(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        poly = np.poly1d(coeffs)

        # Plot the regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = 10**poly(np.log10(x_fit))
        plt.plot(x_fit, y_fit, linestyle='--', color=colors[i % len(colors)])

    # Set logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('h')
    plt.ylabel('Aggregated L2 error')
    plt.title('Aggregated L2 error vs h for all q values')
    plt.legend()

    # Save the current figure to the PDF
    pdf.savefig()
    plt.close()

    grouped_df = df.groupby(['q', 'n']).agg({'L2 error': np.mean, 'h': 'first'}).reset_index()
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Iterate through each 'q' value and plot them
    for i, q_value in enumerate(q_values):
        # Filter the DataFrame for the given 'q' value
        filtered_df = grouped_df[grouped_df['q'] == q_value]

        # Scatter plot of the aggregated L2 error
        x = filtered_df['h']
        y = filtered_df['L2 error']
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'q = {q_value}')

        # Fit a regression line in log-log space
        log_x = np.log10(x)
        log_y = np.log10(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        poly = np.poly1d(coeffs)

        # Plot the regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = 10**poly(np.log10(x_fit))
        plt.plot(x_fit, y_fit, linestyle='--', color=colors[i % len(colors)])

    # Set logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('h')
    plt.ylabel('Aggregated L2 error')
    plt.title('Aggregated L2 error vs h for all q values')
    plt.legend()

    # Save the current figure to the PDF
    pdf.savefig()
    plt.close()

    grouped_df = df.groupby(['q', 'n']).agg({'L2 error': np.median, 'h': 'first'}).reset_index()
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Iterate through each 'q' value and plot them
    for i, q_value in enumerate(q_values):
        # Filter the DataFrame for the given 'q' value
        filtered_df = grouped_df[grouped_df['q'] == q_value]

        # Scatter plot of the aggregated L2 error
        x = filtered_df['h']
        y = filtered_df['L2 error']
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'q = {q_value}')

        # Fit a regression line in log-log space
        log_x = np.log10(x)
        log_y = np.log10(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        poly = np.poly1d(coeffs)

        # Plot the regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = 10**poly(np.log10(x_fit))
        plt.plot(x_fit, y_fit, linestyle='--', color=colors[i % len(colors)])

    # Set logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('h')
    plt.ylabel('Aggregated L2 error')
    plt.title('Aggregated L2 error vs h for all q values')
    plt.legend()

    # Save the current figure to the PDF
    pdf.savefig()
    plt.close()

print("Plots with aggregation and regression lines have been saved to error_vs_h_grouped_by_q_with_aggregation_and_regression.pdf")

