import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

file_path = "results_p23.txt"

data = pd.read_csv(file_path)

# q_values = data['q'].unique()
# with PdfPages("error_vs_h_grouped_by_p.pdf") as pdf:
#     for q_value in q_values:
#         data_q = data[data['q'] == q_value]

#         plt.figure(figsize=(10, 6))

#         for p_value, group in data_q.groupby('p'):
#             x = group['gridlen']
#             y = group['L2 error']

#             plt.scatter(x, y, label = f"p = {p_value}")

#             # Fit a regression line in log-log space
#             log_x = np.log10(x)
#             log_y = np.log10(y)
#             coeffs = np.polyfit(log_x, log_y, 1)
#             poly = np.poly1d(coeffs)

#             print(f"q = {q_value}, p = {p_value}: \n\tExpected = {q_value - p_value + 2} \t Actual = {coeffs[0]}")

#             x_fit = np.linspace(x.min(), x.max(), 100)
#             y_fit = 10**poly(np.log10(x_fit))
#             plt.plot(x_fit, y_fit, linestyle='-')

#         plt.xscale("log")
#         plt.yscale("log")

#         plt.xlabel("Grid length")
#         plt.ylabel("L2 error")
#         plt.title(f"L2 error vs Grid length (q = {q_value})")
#         plt.legend()

#         pdf.savefig()
#         plt.close()


p_values = data['p'].unique()
with PdfPages("error_vs_h_grouped_by_q.pdf") as pdf:
    for p_value in p_values:
        data_p = data[data['p'] == p_value]

        plt.figure(figsize=(10, 6))

        for q_value, group in data_p.groupby('q'):
            print(group)
            x = group['h']
            y = mean(group['L2 error'])

            plt.scatter(x, y, label = f"q = {q_value}")

            # Fit a regression line in log-log space
            log_x = np.log10(x)
            log_y = np.log10(y)
            coeffs = np.polyfit(log_x, log_y, 1)
            poly = np.poly1d(coeffs)

            print(f"q = {q_value}, p = {p_value}: \n\tExpected = {q_value - p_value + 2} \t Actual = {coeffs[0]}")

            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = 10**poly(np.log10(x_fit))
            plt.plot(x_fit, y_fit, linestyle='-')

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Grid length")
        plt.ylabel("L2 error")
        plt.title(f"L2 error vs Grid length (p = {p_value})")
        plt.legend()

        pdf.savefig()
        plt.close()

# data['q-p'] = data['q'] - data['p']

# grouped_data = data.groupby(['q-p', 'h']).agg({'L2 error': 'mean', 'gridlen': 'first'}).reset_index()
# qp_values = data['q-p'].unique()

# colormap = plt.get_cmap('tab20')
# colors = [colormap(i) for i in range(len(qp_values))]

# plt.figure(figsize=(10, 6))

# for i, qp_value in enumerate(qp_values):
#     if (qp_value not in range(-1, 5)):
#         continue
#     # if (qp_value not in range(-1, 4)):
#     #     continue
#     filtered_data = grouped_data[grouped_data['q-p'] == qp_value]

#     x = filtered_data['gridlen']
#     y = filtered_data['L2 error']
#     # print(list(zip(x, y)))

#     plt.scatter(x, y, label = f"q - p = {qp_value}", color = colors[i])

#     # Fit a regression line in log-log space
#     log_x = np.log10(x[1:])
#     log_y = np.log10(y[1:])
#     coeffs = np.polyfit(log_x, log_y, 1)
#     poly = np.poly1d(coeffs)

#     print(f"q - p = {qp_value}:\tExpected = {qp_value + 2}\tActual = {coeffs[0]}")

#     x_fit = np.linspace(x.min(), x.max(), 100)
#     y_fit = 10**poly(np.log10(x_fit))
#     plt.plot(x_fit, y_fit, linestyle='-', color = colors[i])

# plt.xscale("log")
# plt.yscale("log")

# plt.xlabel("Grid length")
# plt.ylabel("L2 error")
# plt.title(f"L2 error vs Grid length grouped by q - p")
# plt.legend()

# plt.savefig("a.pdf")
# plt.close()

