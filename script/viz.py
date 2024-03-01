# Created on 12 Jun 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Visualization
import matplotlib.pyplot as plt

from utility_viz import *


def viz_CAMPD(date, target, n_sample, save_fig):
    path_viz = create_directory(f"../viz/CAMPD_{date}/")
    path_res = f"../model/Bayesian_CAMPD_{date}/"
    path_res_os = f"../model/OneShot_CAMPD_{date}/"
    paths = (path_res, path_viz)

    figure_0(paths, target, n_sample, save_fig)             # parity for modeling
    # figure_0B(paths, target, n_sample, save_fig)            # parity for optimization
    figure_1(paths, target, save_fig)                       # BO performance with different sizes of initial samples
    figure_1os(paths, path_res_os, target, save_fig)        # BO performance with different sizes of initial samples
    if target == "QH":
        figure_2A(paths, target, n_sample, save_fig)        # the change of (y, time cost) with BO iterations
        figure_2B(paths, target, n_sample, save_fig)        # the change of (g1, g2) with BO iterations
        figure_2Asub(paths, target, n_sample, save_fig)     # the change of (y, time cost) with BO iterations
    else:
        figure_2A(paths, target, n_sample, save_fig)        # the change of (y, time cost) with BO iterations
        figure_2B(paths, target, n_sample, save_fig)        # the change of (g1, g2) with BO iterations
        figure_2Asub(paths, target, n_sample, save_fig)     # the change of (y, time cost) with BO iterations
    figure_5(paths, target, save_fig)                       # success/convergence rate of optimization
    figure_5B(paths, target, save_fig)                      # success/convergence rate of simulation
    # figure_4(paths, target, n_sample, save_fig)           # the change of solvent properties with BO iterations
    # figure_3(paths, target, n_sample, save_fig)           # distribution of solvent properties (database, optim, cand)
    # figure_3B(paths, target, n_sample, save_fig)          # distribution of solvent properties (database, optim, cand)


def viz_CAPD(date, solvent, target, n_sample, save_fig):

    path_viz = create_directory(f"../viz/CAPD_{solvent}_{date}/")
    path_res = f"../model/Bayesian_CAPD_{solvent}_{date}/"
    path_res_os = f"../model/OneShot_CAPD_{solvent}_{date}/"
    paths = (path_res, path_viz)

    figure_0(paths, target, n_sample, save_fig)                 # parity for modeling
    figure_1(paths, target, save_fig)                  # BO performance with different sizes of initial samples
    # figure_1os(paths, path_res_os, target, save_fig) # BO performance with different sizes of initial samples
    figure_2A(paths, target, n_sample, save_fig)       # the change of (y, time cost) with BO iterations
    figure_2B(paths, target, n_sample, save_fig)       # the change of (g1, g2) with BO iterations
    # figure_2Asub(paths, target, n_sample, save_fig)  # the change of (y, time cost) with BO iterations
    figure_5(paths, target, save_fig)                  # success/convergence rate of optimization
    figure_5B(paths, target, save_fig)                          # success/convergence rate of simulation


if __name__ == "__main__":
    save_fig = True

    # Case 1: minimization of reboiler heat duty
    viz_CAMPD(date="1221", target="QH", n_sample=384, save_fig=save_fig)
    viz_CAPD(date="1221", solvent="C2H6O2", target="QH", n_sample=1024, save_fig=save_fig)

    # Case 2: minimization of total annual cost
    viz_CAMPD(date="1221", target="TAC", n_sample=384, save_fig=save_fig)
    viz_CAPD(date="1221", solvent="C2H6O2", target="TAC", n_sample=128, save_fig=save_fig)

    # Benchmark
    # viz_CAPD(date="1221", solvent="C5H9NO-D2", target="QH", n_sample=256, save_fig=save_fig)
    # viz_CAPD(date="1221", solvent="C5H9NO-D2", target="TAC", n_sample=256, save_fig=save_fig)
