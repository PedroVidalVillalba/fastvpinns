import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


"""
Fits and plots a regression line in log-log space for the given data
:param x: x component of the data
:type x: list
:param y: y component of the data
:type y: list
:param color: color used to plot the regression line
:type color: color
:param start: starting index of the data to fit the regression line
:type start: int
:param end: last index (not included) of the data to fit the regression line
:type end: int
"""
def fitLogRegression(x: list, y: list, color, start=0, end=None):
    log_x = np.log(x[start:end])
    log_y = np.log(y[start:end])

    # Fit the regression line
    coeffs = np.polyfit(log_x, log_y, 1)
    poly = np.poly1d(coeffs)

    # Points used to draw the line
    x_fit = np.linspace(x.min(), x.max(), 100)
    # Line has equation: log(y) = a log(x) + b = poly(log(x)),
    # hence y = exp(poly(log(x)))
    y_fit = np.exp(poly(np.log(x_fit)))
    plt.plot(x_fit, y_fit, linestyle = "-", color = color)

    # Return slope of the regression line
    return coeffs[0]


def plotSlopes():
    x_values = np.linspace(0.5, 1, 100)
    log_x_values = np.log(x_values)

    y0 = 1.0e-4

    for slope in range(8):
        y_values = np.exp( np.log(y0) + slope * (log_x_values - log_x_values[0]) )

        plt.plot(x_values, y_values, linestyle = "--", color = "grey")
        plt.text(x_values[99] + 0.05, y_values[99], f"{slope}")


"""
Plots the data in input_file to output_file.
Groups the data by f value in different plots and by s value in the same plot.
:param input_file: Name of the file containing the data
:type input_file: str
:param output_file: Name of the file in which to plot the results (should be PDF)
:type output_file: str
:param f: First field of the data to group the results in different plots
:type f: str
:param s: Second field of the data to group the results within the same plot
:type s: str
:param error: Error type to plot
:type error: str
:param epoch: Value of the 'Epoch' field for which to plot the results
:type epoch: int
"""
def plotResultsGrouped(input_file: str, output_file: str, f, s, error = 'L2 error', epoch = 10000):
    print(f"{error} vs h, grouped by ({f}, {s}) at epoch {epoch}")
    print(f"Data read from: {input_file}")

    # Read data from input file
    data = pd.read_csv(input_file)

    # Filter to use only data from the given epoch
    data = data[data['Epoch'] == epoch]
    data['q'] = 2 * data['q'] - 3
    data['p'] = data['p'] + 1

    # Aggregate the errors by (p, q, n) values
    data = data.groupby(['p', 'q', 'n']).agg(error_mean=(error, lambda x : np.exp(np.mean(np.log(x)))),
            error_q1=(error, lambda x : np.quantile(x, 0.25)), error_q3=(error, lambda x : np.quantile(x, 0.75)),
            h=('h', "first")).reset_index()

    # Create different plots for every f value
    f_values = data[f].unique()
    with PdfPages(output_file) as pdf:
        for f_value in f_values:
            data_f = data[data[f] == f_value]

            plt.figure(figsize=(10, 6))

            s_values = data_f[s].unique()
            print(f"{f}={f_value}, {s}:{s_values}")
            colormap = plt.get_cmap("tab10")
            colors = [colormap(i) for i in range(len(s_values))]
            # Group data by s in the same plot
            for i, s_value in enumerate(s_values):
                # Get data with the given (f, s) values
                data_fs = data_f[data_f[s] == s_value]

                # Get x, y values for the points
                x = data_fs['h']
                y = data_fs['error_mean']
                y_q1 = data_fs['error_q1']
                y_q3 = data_fs['error_q3']

                plt.scatter(x, y, label = f"{s} = {s_value}", color = colors[i])

                plt.fill_between(x, y_q1, y_q3, color = colors[i], alpha = 0.3)

                slope = fitLogRegression(x, y, colors[i], start=-16)
                print(f"({f}={f_value}, {s}={s_value}): {slope}")

            plotSlopes()

            # Use logarithmic scale for the plot
            plt.xscale("log")
            plt.yscale("log")

            plt.xlabel("Grid length (h)")
            plt.ylabel(f"{error}")
            plt.title(f"{error} vs h grouped by {s} ({f} = {f_value})", fontsize=20)
            plt.legend()

            pdf.savefig()
            plt.close()

    print(f"Results plotted to: {output_file}\n")

def plotEOC(input_file: str, output_file: str, f, s, epoch = 10000, sigma = 1):
    print(f"EOC vs h, grouped by ({f}, {s}) at epoch {epoch}; smoothed with gaussian filter (sigma={sigma})")
    print(f"Data read from: {input_file}")

    # Read data from input file
    data = pd.read_csv(input_file)

    # Filter to use only data from the given epoch
    data = data[data['Epoch'] == epoch]

    # Aggregate the errors by (p, q, n) values
    data = data.groupby(['p', 'q', 'n']).agg(error_mean=('L2 error', lambda x : np.exp(np.mean(np.log(x)))),
            h=('h', "first")).reset_index()

    # Create different plots for every f value
    f_values = data[f].unique()
    with PdfPages(output_file) as pdf:
        for f_value in f_values:
            data_f = data[data[f] == f_value]

            plt.figure(figsize=(10, 6))

            s_values = data[s].unique()
            colormap = plt.get_cmap("tab10")
            colors = [colormap(i) for i in range(len(s_values))]
            # Group data by s in the same plot
            for i, s_value in enumerate(s_values):
                # Get data with the given (f, s) values
                data_fs = data_f[data_f[s] == s_value]

                # Get x, y values for the points
                x = data_fs['h']
                y = data_fs['error_mean']

                log_h = np.array(np.log(x))
                log_e = np.array(np.log(y))
                eoc = np.zeros(len(x))
                for j in range(len(x) - 1):
                    eoc[j+1] = (log_e[j] - log_e[j+1]) / (log_h[j] - log_h[j+1])

                # Apply moving average to the EOC to reduce noise
                eoc = gaussian_filter1d(eoc, sigma = sigma)

                plt.scatter(x, eoc, label = f"{s} = {s_value}", color = colors[i])

            # Use logarithmic scale for the plot
            plt.xscale("log")
            # plt.yscale("log")

            plt.xlabel("Grid length (h)")
            plt.ylabel("EOC")
            plt.title(f"EOC vs h grouped by {s} ({f} = {f_value})", fontsize=20)
            plt.legend()

            pdf.savefig()
            plt.close()

    print(f"Results plotted to: {output_file}\n")

def errorVSLoss(input_file: str, output_file: str):
    print(f"Error vs √Loss")
    print(f"Data read from: {input_file}")

    # Read data from input file
    data = pd.read_csv(input_file)

    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(10, 6))

        sqrt_loss = np.sqrt(data['Loss'])
        error = data['L2 error']

        plt.scatter(sqrt_loss, error, s=1)
        corr = np.corrcoef(sqrt_loss, error)
        print(f"Correlation:\n{corr}")
        print(f"Coefficient of determination: R²={corr[0][1]*corr[0][1]}")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(r"$\sqrt{\text{Loss}}$")
        plt.ylabel(r"L2 error")
        plt.title(r"L2 error vs $\sqrt{\text{Loss}}$", fontsize=20)

        pdf.savefig()
        plt.close()

    print(f"Results plotted to: {output_file}\n")

def epochImprovement(input_file: str, output_file: str):
    print("Error improvement from 5000 to 10000 epochs")
    print(f"Data read from: {input_file}")

    # Read data from input file
    data = pd.read_csv(input_file)

    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(10, 6))

        data_size = len(data.index)
        l2_error_index = 6
        improvement = np.zeros(data_size // 2)
        for i in range(0, data_size//2):
            improvement[i] = data.iloc[2 * i, l2_error_index] / data.iloc[2 * i+1, l2_error_index]

        print(f"Average improvement: {np.exp(np.mean(np.log(improvement)))}")
        plt.plot(improvement)

        plt.yscale("log")

        plt.xlabel("Execution")
        plt.ylabel("Improvement")
        plt.title("Improvement in L2 error from 5000 to 10000 epochs", fontsize=20)

        pdf.savefig()
        plt.close()
    
    print(f"Results plotted to: {output_file}\n")



def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_file>", file = sys.stderr)
        exit(1)

    input_file = sys.argv[1]

    plotResultsGrouped(input_file, "out_pq.pdf", 'p', 'q')
    plotResultsGrouped(input_file, "out_qp.pdf", 'q', 'p')
    plotResultsGrouped(input_file, "loss_pq.pdf", 'p', 'q', error = 'Loss')
    plotResultsGrouped(input_file, "loss_qp.pdf", 'q', 'p', error = 'Loss')
    plotEOC(input_file, "eoc_pq.pdf", 'p', 'q', sigma = 3)
    plotEOC(input_file, "eoc_qp.pdf", 'q', 'p', sigma = 3)
    errorVSLoss(input_file, "error_vs_loss.pdf")
    epochImprovement(input_file, "training_improvement.pdf")


if __name__ == "__main__":
    main()
