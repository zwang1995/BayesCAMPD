# Created on 12 Jun 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Visualization

from utility_viz import *


def viz_CAMPD(date, target, n_sample, save_fig):
    path_res = f"../model/Bayesian_CAMPD_{date}/"
    path_viz = create_directory(f"../viz/CAMPD_{date}/")
    path_res_os = f"../model/OneShot_CAMPD_{date}/"
    paths = (path_res, path_viz)

    # figure_0(paths, target, n_sample, save_fig)             # parity for modeling
    # figure_0B(paths, target, n_sample, save_fig)            # parity for optimization
    figure_1(paths, target, save_fig)                       # BO performance with different sizes of initial samples
    figure_1os(paths, path_res_os, target, save_fig)        # OS performance with different sizes of initial samples
    if target == "QH":
        figure_2A(paths, target, n_sample, save_fig)        # the change of (y, time cost) with BO iterations
        figure_2B(paths, target, n_sample, save_fig)        # the change of (g1, g2) with BO iterations
        # figure_2Asub(paths, target, n_sample, save_fig)     # the change of (y, time cost) with BO iterations
    else:
        figure_2A(paths, target, n_sample, save_fig)        # the change of (y, time cost) with BO iterations
        figure_2B(paths, target, n_sample, save_fig)        # the change of (g1, g2) with BO iterations
        # figure_2Asub(paths, target, n_sample, save_fig)     # the change of (y, time cost) with BO iterations
    figure_5(paths, target, save_fig)                       # success/convergence rate of optimization
    # figure_5B(paths, target, save_fig)                      # success/convergence rate of initial simulation
    # figure_4(paths, target, n_sample, save_fig)             # the change of solvent properties with BO iterations
    # figure_3(paths, target, n_sample, save_fig)             # distribution of solvent props (database, optim, cand)
    # figure_3B(paths, target, n_sample, save_fig)            # distribution of solvent props (database, optim, cand)


def viz_CAMPD_sup(date, target, save_fig):
    path_res = f"../model/Bayesian_CAMPD_{date}/"
    path_viz = create_directory(f"../viz/CAMPD_{date}/")
    paths = (path_res, path_viz)

    figure_1_sup(paths, target, save_fig)                   # BO performance with different sizes of initial samples


def viz_CAPD(date, solvent, target, n_sample, save_fig):
    path_viz = create_directory(f"../viz/CAPD_{solvent}_{date}/")
    path_res = f"../model/Bayesian_CAPD_{solvent}_{date}/"
    # path_res_os = f"../model/OneShot_CAPD_{solvent}_{date}/"
    paths = (path_res, path_viz)

    # figure_0(paths, target, n_sample, save_fig)             # parity for modeling
    figure_1(paths, target, save_fig)                       # BO performance with different sizes of initial samples
    # figure_1os(paths, path_res_os, target, save_fig)        # OS performance with different sizes of initial samples
    figure_2A(paths, target, n_sample, save_fig)            # the change of (y, time cost) with BO iterations
    figure_2B(paths, target, n_sample, save_fig)            # the change of (g1, g2) with BO iterations
    # figure_2Asub(paths, target, n_sample, save_fig)         # the change of (y, time cost) with BO iterations
    # figure_5(paths, target, save_fig)                       # success/convergence rate of optimization
    # figure_5B(paths, target, save_fig)                      # success/convergence rate of initial simulation


if __name__ == "__main__":
    saveFig = True

    date = "E15_P20"

    # Case 1: minimization of reboiler heat duty
    viz_CAMPD(date=date, target="QH", n_sample=768, save_fig=saveFig)
    viz_CAPD(date=date, solvent="C2H6O2", target="QH", n_sample=512, save_fig=saveFig)

    # Case 2: minimization of total annual cost
    viz_CAMPD(date=date, target="TAC", n_sample=768, save_fig=saveFig)
    viz_CAPD(date=date, solvent="C2H6O2", target="TAC", n_sample=256, save_fig=saveFig)

    # Others
    # viz_CAMPD_sup(date="0506", target="QH", save_fig=saveFig)
