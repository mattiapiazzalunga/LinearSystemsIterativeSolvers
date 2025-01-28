import eel
import numpy as np
from scipy.io import mmread
import GenericAlgorithm as gA
import UpdateStrategy as uS
import scipy as sc

# import matplotlib.pyplot as plt
# import pandas as pd

eel.init("web")
# THIS INTERFACE IS ONLY A FACILITATOR, IS NOT MADE FOR MULTIPLE REQUEST, ...


@eel.expose
def matrix_by_file(path: str, tolerance: float, method: int, plot: str):
    try:
        matrix = mmread(path)
        matrix = matrix.tocsr()
        print("a")
        eigen_value, _ = sc.sparse.linalg.eigsh(matrix, k=matrix.get_shape()[0]-1)
        print(np.abs(max(eigen_value))/np.abs(min(eigen_value)))
        #apply_method(matrix, float(tolerance), int(method), plot)
        print("d")
    except Exception as exception:
        eel.exception(str(exception))


@eel.expose
def matrix_by_matrix(matrix: np.array, tolerance: float, method: int):
    try:
        apply_method(sc.sparse.csr_matrix(np.array(matrix).astype(float)), float(tolerance), int(method), "False")
    except Exception as exception:
        eel.exception(str(exception))


result = []  # This can be a problem in concurrent request!


def apply_method(matrix: sc.sparse.csr_matrix, tolerance: float, method: int, plot: str):
    try:
        if tolerance == 4:
            tolerance = 1e-4
        elif tolerance == 6:
            tolerance = 1e-6
        elif tolerance == 8:
            tolerance = 1e-8
        else:
            tolerance = 1e-10
        x_solution = np.ones(gA.GenericAlgorithm.get_matrix_rows(matrix))
        algorithm = gA.GenericAlgorithm(matrix, matrix.dot(x_solution), x_solution, tolerance)
        update: uS.UpdateStrategy

        result.clear()
        if method == 5:
            update = uS.JacobyStrategy()
            result.append(algorithm.apply_algorithm(update))
            update = uS.GaubSeidelStrategy()
            result.append(algorithm.apply_algorithm(update))
            update = uS.GradientStrategy()
            result.append(algorithm.apply_algorithm(update))
            update = uS.ConjugateGradientStrategy()
            result.append(algorithm.apply_algorithm(update))
        else:
            if method == 1:
                update = uS.JacobyStrategy()
            elif method == 2:
                update = uS.GaubSeidelStrategy()
            elif method == 3:
                update = uS.GradientStrategy()
            else:
                update = uS.ConjugateGradientStrategy()
            result.append(algorithm.apply_algorithm(update))
        if plot == "False":
            eel.redirect()
    except Exception as exception:
        eel.exception(str(exception))


@eel.expose
def give_me_the_result():
    for element in result:
        method = str(element.get_method())
        method = method[23:len(method) - 10]
        eel.add_row(method, str(element.get_relative_error()), str(element.get_time()),
                    str(element.get_number_of_iterations()))

# This part is for plot generation!
# To avoid problems, since it is a supporting part of the project
# and not meant for online operation, it is commented out.
# To use it, uncomment it, but beware of making concurrent requests and
# watch out for paths to .mtx files!
# Once uncommented, enter the paths to the arrays in the path_to_plot
# array and then invoke the eel.generate_plot()
# command from the browser console.
# @eel.expose
# def generate_plot():
#     try:
#         print("We are generating all plots")
#         path_to_plot = ["C:/Users/matti/Downloads/dati/spa1.mtx",
#                         "C:/Users/matti/Downloads/dati/spa2.mtx",
#                         "C:/Users/matti/Downloads/dati/vem1.mtx",
#                         "C:/Users/matti/Downloads/dati/vem2.mtx"]
#         tolerance = [4, 6, 8, 10]
#         for tol in tolerance:
#             print("Calculating for " + str(tol))
#             m = []
#             e = []
#             t = []
#             i = []
#             n = []
#             for matrix in path_to_plot:
#                 matrix_by_file(matrix, float(tol), 5, "True")
#                 for element in result:
#                     method = str(element.get_method())
#                     method = method[23:len(method) - 10]
#                     m.append(method)
#                     e.append(element.get_relative_error())
#                     t.append(element.get_time())
#                     i.append(element.get_number_of_iterations())
#                     n.append(matrix[30:-4])
#             for what in ["Relative Error", "Time", "Number of iterations"]:
#                 df = pd.DataFrame()
#                 df[what] = (e if what == "Relative Error" else (t if what == "Time" else i))
#                 df["Matrix"] = n
#                 df["Method"] = m
#                 df_pivot = pd.pivot_table(df, values=what, index="Matrix", columns="Method")
#                 print(df_pivot)
#                 ax = df_pivot.plot(kind="bar", color=["#4A6274", "#94ACBF", "#79AEB2", "#FDD8D6"])
#                 fig = ax.get_figure()
#                 ax.set_xlabel("Matrix")
#                 ax.set_ylabel(what)
#                 ax.set_title("Tolerance: 10e-"+str(tol))
#                 fig.show()
#         print("All plots generated!")
#     except Exception as exception:
#         eel.exception(str(exception))


eel.start("index.html", mode='firefox')
