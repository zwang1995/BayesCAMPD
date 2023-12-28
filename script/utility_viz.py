# Created on 12 Jun 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Functions used for visualization

from utility_common import *

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# print(plt.rcParams)

ratio = 1
plt.rcParams["font.size"] = "10.5"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.figsize"] = (3.3 * ratio, 3 * ratio)
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["legend.columnspacing"] = 0.6
plt.rcParams["legend.handletextpad"] = 0.3
plt.rcParams["figure.dpi"] = 600.0
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True

label_size, text_size, legend_size = 11, 10, 8.5
dot_size = 30
lw = 1
alp = 0.7

scale_t, scale_hv = 3600, 1e3
puri_spec = logit(0.995)


def extend_xylim(low, up, onlyLow=False, onlyUp=False):
    percent = 0.05
    range_ = up - low
    if onlyLow:
        return [low - percent * range_, up]
    elif onlyUp:
        return [low, up + percent * range_]
    else:
        return [low - percent * range_, up + percent * range_]


def get_bo_result(bo_record_file, columns):
    df_result = pd.read_csv(bo_record_file)
    df_result = df_result[df_result["optim_success"]]
    print(df_result.columns)

    n_col = len(columns)
    next_x = np.array([list_str2value(x) for x in df_result["next_x"].values])
    next_x = next_x[:, :n_col]
    df_next = pd.DataFrame(next_x, columns=columns)

    x_mol = np.array([list_str2value(next_x)[:3] for next_x in df_result["candidate_props"].values])
    df_cand = pd.DataFrame(x_mol, columns=columns)

    return df_result, df_next, df_cand


def get_data(df, col, result_filter=None, isTuple=True):
    y = df[col].values
    if result_filter == "Error":
        hasErrorNext = [True if i or np.isnan(i) else False for i in df["hasErrorNext"].values]
        no_error_idx = np.where(~np.array(hasErrorNext))[0]
        y = y[no_error_idx]
        if isTuple and "hat" in col:
            y = np.array([tuple_str2value(yi)[1] for yi in y])
        return no_error_idx + 1, y
    elif result_filter == "Fulfill":
        fulfillNext = [True if i and not np.isnan(i) else False for i in df["fulfillNext"].values]
        no_error_idx = np.where(np.array(fulfillNext))[0]
        y = y[no_error_idx]
        if isTuple and "hat" in col:
            y = np.array([tuple_str2value(yi)[1] for yi in y])
        return no_error_idx + 1, y
    else:
        if isTuple and "hat" in col:
            y = np.array([tuple_str2value(yi)[1] for yi in y])
        return y


def figure_1(paths, target, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 1")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if target == "QH":
        y_label = "Q$_H$ (MW)"
        ylim1_dict = {"CAMPD": extend_xylim(0, 50, onlyUp=True), "CAPD": extend_xylim(9, 18)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 12, onlyUp=True), "CAPD": extend_xylim(0, 3, onlyUp=True)}
    else:
        y_label = "TAC (MM$/yr)"
        ylim1_dict = {"CAMPD": extend_xylim(0, 15, onlyUp=True), "CAPD": extend_xylim(2, 6)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 12, onlyUp=True), "CAPD": extend_xylim(0, 3, onlyUp=True)}

    n_s, y_best, y_next = [], [], []
    n_failed, y_best_failed = [], []
    time_ini, time_extra, time_model, time_optim = [], [], [], []

    if task == "CAMPD":
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]
    else:
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]

    if task == "CAMPD":
        file_mol = path_res + "data/solvent_list.csv"
        df_mol = pd.read_csv(file_mol)
        alias_name_dict = {row["Solvent"]: row["cname"] for (idx, row) in df_mol.iterrows()}
    else:
        alias_name_dict = None

    for n_sample in n_samples:
        suffix = f"{target}_{n_sample}"
        file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
        bo_monitor_record = joblib.load(file_bo_monitor)
        # print(bo_monitor_record)

        fulfill_idx, next_x, next_y, next_alias, cand_props, next_g1, next_g2 = [], [], [], [], [], [], []
        for record in bo_monitor_record:
            if record["fulfillNext"]:
                idx_feas = np.where(record["hasErrorNext"] == 0)[0]
                x = np.array(record["next_x"])[0][idx_feas, :]
                y = np.array(record["next_y"])
                alias = np.array(record["next_alias"])
                prop = np.array(record["candidate_props"])
                g1 = np.array(record["next_g1"])
                g2 = np.array(record["next_g2"])
                idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
                idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]

                fulfill_idx.append(record["iteration"])
                next_x.append(x[idx_best])
                next_y.append(y[idx_best])
                next_alias.append(alias[idx_best])
                cand_props.append(prop[idx_best])
                next_g1.append(g1[idx_best])
                next_g2.append(g2[idx_best])

        y_tr_best = bo_monitor_record[0]["y_tr_best"]
        if len(fulfill_idx) > 0:
            n_s.append(n_sample)
            y_best.append(y_tr_best / scale_y)
            y_next.append(next_y[-1] / scale_y)
            if task == "CAMPD":
                print(n_sample, y_tr_best, alias_name_dict[next_alias[-1]], next_y[-1], next_alias[-1],
                      fulfill_idx[-1], [x for x in next_x[-1]], [x for x in cand_props[-1]])
            else:
                print(n_sample, y_tr_best, next_y[-1], next_alias[-1], fulfill_idx[-1],
                      [x for x in next_x[-1]], [x for x in cand_props[-1]])
            print(f"{int(next_x[-1][-7])}, {next_x[-1][-6]:.2f}, {next_x[-1][-5]:.2f}, "
                  f"{int(next_x[-1][-3])}, {next_x[-1][-2]:.2f}, {next_x[-1][-1]:.2f}, "
                  f"{next_x[-1][-4]:.2f}, {next_y[-1] / scale_y:.2f}")
        else:
            n_failed.append(n_sample)
            y_best_failed.append(y_tr_best / scale_y)
            print(n_sample, y_tr_best)

        time_initial_labeling = bo_monitor_record[-1]["time_initial_labeling"]
        time_extra_labeling = bo_monitor_record[-1]["time_extra_labeling"]
        time_modeling = bo_monitor_record[-1]["time_modeling"]
        time_optimization = bo_monitor_record[-1]["time_optimization"]
        time_ini.append(time_initial_labeling / scale_t)
        time_extra.append(time_extra_labeling / scale_t)
        time_model.append(time_modeling / scale_t)
        time_optim.append(time_optimization / scale_t)

    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])
    plt.scatter(n_s, y_best, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    plt.scatter(n_s, y_next, dot_size, "tab:red", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    plt.scatter(n_failed, y_best_failed, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    for xi, y_besti, y_nexti in zip(n_s, y_best, y_next):
        plt.plot([xi, xi], [y_besti, y_nexti], "k-", zorder=2)

    fake_x, fake_y = 1, -1
    plt.scatter(fake_x, fake_y, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, label="Start")
    plt.scatter(fake_x, fake_y, dot_size, "tab:red", marker="s", edgecolors="k", linewidth=lw, label="End")

    plt.xticks(n_samples)
    plt.xlim(extend_xylim(n_samples[0], n_samples[-1]))
    plt.ylim(ylim1_dict[task])
    plt.ylabel(y_label, size=label_size)
    # if task == "CAMPD":
    plt.legend(prop={"size": legend_size}, ncol=2, loc="upper left")
        # plt.legend(prop={"size": legend_size}, ncol=2, loc="upper left", bbox_to_anchor=(0.07, 1))
    # else:
    #     plt.legend(prop={"size": legend_size}, ncol=2, loc="upper center")
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)

    plt.subplot(gs[1])
    time_by_step = {"Initial labeling": time_ini, "Extra labeling": time_extra,
                    "Modeling": time_model, "Optimization": time_optim}
    color_map = ["mediumseagreen", "mediumpurple", "darkorange", "dodgerblue"]
    plt.stackplot(n_samples, time_by_step.values(), colors=color_map, labels=time_by_step.keys(), alpha=0.8, zorder=2)
    lw_, c_, alpha_ = 0.7, "w", 0.8
    plt.plot(n_samples, time_ini, lw=lw_, c=c_, alpha=alpha_)
    plt.plot(n_samples, np.array(time_ini) + np.array(time_extra), lw=lw_, c=c_, alpha=alpha_)
    plt.plot(n_samples, np.array(time_ini) + np.array(time_extra) + np.array(time_model), lw=lw_, c=c_, alpha=alpha_)
    plt.xticks(n_samples)
    plt.xlim(extend_xylim(n_samples[0], n_samples[-1]))
    plt.ylim(ylim2_dict[task])
    plt.xlabel("Initial sample size", size=label_size)
    plt.ylabel("Computational time (h)", size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    if task == "CAMPD" and target == "TAC":
        plt.legend(prop={"size": legend_size}, ncol=2, loc="upper left", reverse=True)
    else:
        plt.legend(prop={"size": legend_size}, ncol=1, loc="upper left", reverse=True)
        # plt.legend(prop={"size": legend_size}, ncol=2, loc="upper left", bbox_to_anchor=(0.07, 1), reverse=True)
    # else:
    #     plt.legend(prop={"size": legend_size}, ncol=1, loc="upper center", reverse=True)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_1_n_sample")
    plt.close()


def figure_1os(paths, path_res_os, target, solvent=None):
    path_res_bo, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 1 (oneshot)")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if target == "QH":
        y_label = "Q$_H$ (MW)"
        ylim1_dict = {"CAMPD": extend_xylim(0, 50, onlyUp=True), "CAPD": extend_xylim(12, 18)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 12, onlyUp=True), "CAPD": extend_xylim(0, 3, onlyUp=True)}
    else:
        y_label = "TAC (MM$/yr)"
        ylim1_dict = {"CAMPD": extend_xylim(0, 15, onlyUp=True), "CAPD": extend_xylim(2, 6)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 12, onlyUp=True), "CAPD": extend_xylim(0, 3, onlyUp=True)}

    x, y_best, y_next = [], [], []
    x_failed, y_best_failed = [], []
    time_ini, time_extra, time_model, time_optim = [], [], [], []

    if task == "CAMPD":
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    else:
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 2560, 3072, 3584, 4096]

    if task == "CAMPD":
        file_mol = path_res_bo + "data/solvent_list.csv"
        df_mol = pd.read_csv(file_mol)
        alias_name_dict = {row["Solvent"]: row["cname"] for (idx, row) in df_mol.iterrows()}
    else:
        alias_name_dict = None

    for n_sample in n_samples:
        suffix = f"{target}_{n_sample}"
        file_bo_monitor = path_res_os + f"optim_monitor_{suffix}.pkl"
        bo_monitor_record = joblib.load(file_bo_monitor)

        fulfill_idx, next_y, next_alias, next_g1, next_g2 = [], [], [], [], []
        for record in bo_monitor_record:
            if record["fulfillNext"]:
                y = np.array(record["next_y"])
                alias = np.array(record["next_alias"])
                g1 = np.array(record["next_g1"])
                g2 = np.array(record["next_g2"])
                idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
                idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]

                fulfill_idx.append(record["iteration"])
                next_y.append(y[idx_best])
                next_alias.append(alias[idx_best])
                next_g1.append(g1[idx_best])
                next_g2.append(g2[idx_best])

        y_tr_best = bo_monitor_record[0]["y_tr_best"]
        n_simulation = bo_monitor_record[0]["n_simulation"]
        n_tr = bo_monitor_record[0]["n_tr"]
        if len(fulfill_idx) > 0:
            x.append(n_sample)
            y_best.append(y_tr_best / scale_y)
            y_next.append(next_y[-1] / scale_y)
            if task == "CAMPD":
                print(n_sample, n_simulation, n_tr, y_tr_best, fulfill_idx[-1], next_y[-1], next_alias[-1],
                      alias_name_dict[next_alias[-1]])
            else:
                print(n_sample, n_simulation, n_tr, y_tr_best, fulfill_idx[-1], next_y[-1], next_alias[-1])
        else:
            x_failed.append(n_sample)
            y_best_failed.append(y_tr_best / scale_y)
            print(n_sample, n_simulation, n_tr, y_tr_best)

        time_initial_labeling = bo_monitor_record[-1]["time_initial_labeling"]
        time_extra_labeling = bo_monitor_record[-1]["time_extra_labeling"]
        time_modeling = bo_monitor_record[-1]["time_modeling"]
        time_optimization = bo_monitor_record[-1]["time_optimization"]
        time_ini.append(time_initial_labeling / scale_t)
        time_extra.append(time_extra_labeling / scale_t)
        time_model.append(time_modeling / scale_t)
        time_optim.append(time_optimization / scale_t)

    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])
    plt.scatter(x, y_best, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    plt.scatter(x, y_next, dot_size, "tab:red", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    plt.scatter(x_failed, y_best_failed, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, zorder=3)
    for xi, y_besti, y_nexti in zip(x, y_best, y_next):
        plt.plot([xi, xi], [y_besti, y_nexti], "k-", zorder=2)

    fake_x, fake_y = 10, -10
    plt.scatter(fake_x, fake_y, dot_size, "tab:green", marker="s", edgecolors="k", linewidth=lw, label="Start")
    plt.scatter(fake_x, fake_y, dot_size, "tab:red", marker="s", edgecolors="k", linewidth=lw, label="End")

    plt.xticks([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    plt.xlim(extend_xylim(n_samples[0], n_samples[-1]))
    plt.ylim(ylim1_dict[task])
    plt.ylabel(y_label, size=label_size)
    plt.legend(prop={"size": legend_size}, ncol=2, loc="upper left")
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)

    plt.subplot(gs[1])
    time_by_step = {"Initial labeling": time_ini, "Extra labeling": time_extra,
                    "Modeling": time_model, "Optimization": time_optim}
    color_map = ["mediumseagreen", "mediumpurple", "darkorange", "dodgerblue"]
    plt.stackplot(n_samples, time_by_step.values(), colors=color_map, labels=time_by_step.keys(), alpha=0.8, zorder=2)
    lw_, c_, alpha_ = 0.7, "w", 0.8
    plt.plot(n_samples, time_ini, lw=lw_, c=c_, alpha=alpha_)
    plt.plot(n_samples, np.array(time_ini) + np.array(time_extra), lw=lw_, c=c_, alpha=alpha_)
    plt.plot(n_samples, np.array(time_ini) + np.array(time_extra) + np.array(time_model), lw=lw_, c=c_, alpha=alpha_)
    plt.xticks([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    plt.xlim(extend_xylim(n_samples[0], n_samples[-1]))
    plt.ylim(ylim2_dict[task])
    plt.xlabel("Initial sample size", size=label_size)
    plt.ylabel("Computational time (h)", size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    plt.legend(prop={"size": legend_size}, ncol=1, loc="upper left", reverse=True)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_1os_n_sample")
    plt.close()


def figure_2A(paths, target, n_sample, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 2a")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if target == "QH":
        y_label = "Q$_H$ (MW)"
        ylim1_dict = {"CAMPD": extend_xylim(9, 21), "CAPD": extend_xylim(9, 12)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 5, onlyUp=True), "CAPD": extend_xylim(0, 2, onlyUp=True)}
    else:
        y_label = "TAC (MM$/yr)"
        ylim1_dict = {"CAMPD": extend_xylim(2, 10), "CAPD": extend_xylim(3, 5)}
        ylim2_dict = {"CAMPD": extend_xylim(0, 5, onlyUp=True), "CAPD": extend_xylim(0, 1.5, onlyUp=True)}

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(len(bo_monitor_record))

    no_error_idx, next_y, next_yhat, next_g1, next_g1hat, next_g2, next_g2hat = [], [], [], [], [], [], []
    for record in bo_monitor_record:
        if np.any(record["hasErrorNext"] == 0):
            y = np.array(record["next_y"]) / scale_y
            yhat = np.array(record["next_yhat"])[1] / scale_y
            g1 = np.array(record["next_g1"])
            g1hat = np.array(record["next_g1hat"])[1]
            g2 = np.array(record["next_g2"])
            g2hat = np.array(record["next_g2hat"])[1]
            # print(y, yhat, g1, g1hat, g2, g2hat)

            no_error_idx.append(record["iteration"])
            next_y.append(y)
            next_yhat.append(yhat)
            next_g1.append(g1)
            next_g1hat.append(g1hat)
            next_g2.append(g2)
            next_g2hat.append(g2hat)

            if record["fulfillNext"]:
                idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
                idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]
                print(record["iteration"], record["next_alias"][idx_best])

    idx = [record["iteration"] for record in bo_monitor_record]
    y_tr_best = [record["y_tr_best"] / scale_y for record in bo_monitor_record]

    next_g1, next_g2 = [logit_inverse(g1) for g1 in next_g1], [logit_inverse(g2) for g2 in next_g2]

    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])
    for idx_, g1, g2, y, yhat in zip(no_error_idx, next_g1, next_g2, next_y, next_yhat):
        print(idx_, y, yhat)
        for (yi, yhati, g1i, g2i) in zip(y, yhat, g1, g2):
            if (g1i <= 0.995) or (g2i <= 0.995):
                plt.scatter(idx_, yi, dot_size, "lightgray", edgecolors="k", linewidth=lw, zorder=3)
            else:
                plt.scatter(idx_, yi, dot_size, "tab:red", edgecolors="k", linewidth=lw, zorder=4)

    fake_x, fake_y = 10, -10
    plt.scatter(fake_x, fake_y, dot_size, "tab:red", edgecolors="k", linewidth=lw, label="Viable solution")
    plt.scatter(fake_x, fake_y, dot_size, "lightgray", edgecolors="k", linewidth=lw, label="Inviable solution")

    plt.step(idx, y_tr_best, c="tab:green", lw=1.2, zorder=5)
    plt.xlim(extend_xylim(idx[0], idx[-1]))
    plt.ylim(ylim1_dict[task])
    plt.ylabel(y_label, size=label_size)
    if target == "QH":
        plt.legend(prop={"size": legend_size}, ncol=1, loc="upper right")
    else:
        plt.legend(prop={"size": legend_size}, ncol=1, loc="lower left")
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)

    plt.subplot(gs[1])
    time_by_step = {
        "Initial labeling": np.array([record["time_initial_labeling"] for record in bo_monitor_record]) / scale_t,
        "Extra labeling": np.array([record["time_extra_labeling"] for record in bo_monitor_record]) / scale_t,
        "Modeling": np.array([record["time_modeling"] for record in bo_monitor_record]) / scale_t,
        "Optimization": np.array([record["time_optimization"] for record in bo_monitor_record]) / scale_t,
    }
    color_map = ["mediumseagreen", "mediumpurple", "darkorange", "dodgerblue"]
    plt.stackplot(idx, time_by_step.values(), colors=color_map, labels=time_by_step.keys(), alpha=0.8, zorder=2)
    lw_, c_, alpha_ = 0.7, "w", 0.8
    plt.plot(idx, time_by_step["Initial labeling"], lw=lw_, c=c_, alpha=alpha_)
    plt.plot(idx, time_by_step["Initial labeling"] + time_by_step["Extra labeling"], lw=lw_, c=c_, alpha=alpha_)
    plt.plot(idx, time_by_step["Initial labeling"] + time_by_step["Extra labeling"] + time_by_step["Modeling"],
             lw=lw_, c=c_, alpha=alpha_)
    plt.xlim(extend_xylim(idx[0], idx[-1]))
    plt.ylim(ylim2_dict[task])
    plt.xlabel("Iteration", size=label_size)
    plt.ylabel("Computational time (h)", size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    plt.legend(prop={"size": legend_size}, ncol=1, loc="upper left", reverse=True)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_2A_performance_{n_sample}")
    plt.close()


def figure_2Asub(paths, target, n_sample, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 2a")
    plt.clf()

    if target == "QH":
        y_label = "Q$_H$ (MW)"
        ylim_dict = {"CAMPD": extend_xylim(0, 90, onlyUp=True), "CAPD": extend_xylim(9, 12)}

    else:
        y_label = "TAC (MM$/yr)"
        ylim_dict = {"CAMPD": extend_xylim(0, 25, onlyUp=True), "CAPD": extend_xylim(3, 5)}


    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(len(bo_monitor_record))

    no_error_idx, next_y, next_yhat, next_g1, next_g1hat, next_g2, next_g2hat = [], [], [], [], [], [], []
    for record in bo_monitor_record:
        if np.any(record["hasErrorNext"] == 0):
            y = np.array(record["next_y"]) / scale_y
            yhat = np.array(record["next_yhat"])[1] / scale_y
            g1 = np.array(record["next_g1"])
            g1hat = np.array(record["next_g1hat"])[1]
            g2 = np.array(record["next_g2"])
            g2hat = np.array(record["next_g2hat"])[1]
            # print(y, yhat, g1, g1hat, g2, g2hat)

            no_error_idx.append(record["iteration"])
            next_y.append(y)
            next_yhat.append(yhat)
            next_g1.append(g1)
            next_g1hat.append(g1hat)
            next_g2.append(g2)
            next_g2hat.append(g2hat)

            if record["fulfillNext"]:
                idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
                idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]
                print(record["iteration"], record["next_alias"][idx_best])

    idx = [record["iteration"] for record in bo_monitor_record]
    y_tr_best = [record["y_tr_best"] / scale_y for record in bo_monitor_record]

    next_g1, next_g2 = [logit_inverse(g1) for g1 in next_g1], [logit_inverse(g2) for g2 in next_g2]

    for idx_, g1, g2, y, yhat in zip(no_error_idx, next_g1, next_g2, next_y, next_yhat):
        print(idx_, y, yhat)
        for (yi, yhati, g1i, g2i) in zip(y, yhat, g1, g2):
            if (g1i <= 0.995) or (g2i <= 0.995):
                plt.scatter(idx_, yi, dot_size, "lightgray", edgecolors="k", linewidth=lw, zorder=3)
            else:
                plt.scatter(idx_, yi, dot_size, "tab:red", edgecolors="k", linewidth=lw, zorder=4)

    fake_x, fake_y = 10, -10
    plt.scatter(fake_x, fake_y, dot_size, "tab:red", edgecolors="k", linewidth=lw, label="Viable solution")
    plt.scatter(fake_x, fake_y, dot_size, "lightgray", edgecolors="k", linewidth=lw, label="Inviable solution")

    plt.step(idx, y_tr_best, c="tab:green", lw=1.2, zorder=5)
    plt.xlim(extend_xylim(idx[0], idx[-1]))
    plt.ylim(ylim_dict[task])
    plt.xlabel("Iteration", size=label_size)
    plt.ylabel(y_label, size=label_size)
    plt.legend(prop={"size": legend_size}, ncol=1)
    plt.grid(linestyle="dotted", zorder=1)

    plt.savefig(path_viz + f"{target}_2Asub_performance")
    plt.close()


def figure_2B(paths, target, n_sample, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    print(f"\n{task} Figure 2b")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if target == "QH":
        ylim1_dict = {"CAMPD": extend_xylim(0.6, 1, onlyLow=True), "CAPD": extend_xylim(0.8, 1, onlyLow=True)}
        ylim2_dict = {"CAMPD": extend_xylim(0.6, 1, onlyLow=True), "CAPD": extend_xylim(0.8, 1, onlyLow=True)}
    else:
        ylim1_dict = {"CAMPD": extend_xylim(0.6, 1, onlyLow=True), "CAPD": extend_xylim(0.96, 1, onlyLow=True)}
        ylim2_dict = {"CAMPD": extend_xylim(0.6, 1, onlyLow=True), "CAPD": extend_xylim(0.96, 1, onlyLow=True)}

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)

    no_error_idx, next_y, next_yhat, next_g1, next_g1hat, next_g2, next_g2hat = [], [], [], [], [], [], []
    for record in bo_monitor_record:
        if np.any(record["hasErrorNext"] == 0):
            # y = np.array(record["next_y"]) / scale_y
            # yhat = np.array(record["next_yhat"])[1] / scale_y
            g1 = np.array(record["next_g1"])
            g1hat = np.array(record["next_g1hat"])[1]
            g2 = np.array(record["next_g2"])
            g2hat = np.array(record["next_g2hat"])[1]
            # print(y, yhat, g1, g1hat, g2, g2hat)

            # idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
            # idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]

            no_error_idx.append(record["iteration"])
            # next_y.append(y)
            # next_yhat.append(yhat)
            next_g1.append(g1)
            next_g1hat.append(g1hat)
            next_g2.append(g2)
            next_g2hat.append(g2hat)

    next_g1, next_g2 = [logit_inverse(g1) for g1 in next_g1], [logit_inverse(g2) for g2 in next_g2]
    next_g1hat, next_g2hat = [logit_inverse(g1hat) for g1hat in next_g1hat], \
                             [logit_inverse(g2hat) for g2hat in next_g2hat]

    idx = [record["iteration"] for record in bo_monitor_record]

    # comparison between predictions and simulations
    xlim_dict = {"g1": extend_xylim(idx[0], idx[-1]), "g2": extend_xylim(idx[0], idx[-1])}
    ylabel_dict = {"g1": "C$_4$H$_8$ purity", "g2": "C$_4$H$_6$ purity"}
    next_y_dict = {"g1": next_g1, "g2": next_g2}
    next_yhat_dict = {"g1": next_g1hat, "g2": next_g2hat}

    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])
    output = "g1"
    for idx_, g1, g2, y, yhat in zip(no_error_idx, next_g1, next_g2, next_y_dict[output], next_yhat_dict[output]):
        print(idx_, y, yhat)
        for (yi, yhati, g1i, g2i) in zip(y, yhat, g1, g2):
            if (g1i <= 0.995) or (g2i <= 0.995):
                plt.scatter(idx_, yi, dot_size, "lightgray", edgecolors="k", linewidth=lw, zorder=3)
            else:
                plt.scatter(idx_, yi, dot_size, "tab:red", edgecolors="k", linewidth=lw, zorder=4)
    fake_x, fake_y = 1, -1
    plt.scatter(fake_x, fake_y, dot_size, "tab:red", edgecolors="k", linewidth=lw, label="Viable solution")
    plt.scatter(fake_x, fake_y, dot_size, "lightgray", edgecolors="k", linewidth=lw, label="Inviable solution")
    plt.xlim(xlim_dict[output])
    plt.ylim(ylim1_dict[task])
    plt.ylabel(ylabel_dict[output], size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)
    plt.legend(prop={"size": legend_size}, ncol=1, loc="lower right")

    plt.subplot(gs[1])
    output = "g2"
    for idx_, g1, g2, y, yhat in zip(no_error_idx, next_g1, next_g2, next_y_dict[output], next_yhat_dict[output]):
        print(idx_, y, yhat)
        for (yi, yhati, g1i, g2i) in zip(y, yhat, g1, g2):
            if (g1i <= 0.995) or (g2i <= 0.995):
                plt.scatter(idx_, yi, dot_size, "lightgray", edgecolors="k", linewidth=lw, zorder=3)
            else:
                plt.scatter(idx_, yi, dot_size, "tab:red", edgecolors="k", linewidth=lw, zorder=4)
    plt.xlim(xlim_dict[output])
    plt.ylim(ylim2_dict[task])
    plt.xlabel("Iteration", size=label_size)
    plt.ylabel(ylabel_dict[output], size=label_size)
    plt.grid(linestyle="dotted", zorder=1)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_2B_specification_{n_sample}")
    plt.close()


def figure_3(paths, target, n_sample):
    path_res, path_viz = paths
    print("\nFigure_3")
    plt.clf()
    plt.rcParams["font.size"] = "14"
    label_size_ = 14.5
    legend_size_ = 12
    dot_size_ = 50

    xylim_dict = [extend_xylim(0, 11), extend_xylim(50, 400), extend_xylim(20, 120)]
    xystick_dict = [[0, 2, 4, 6, 8, 10], None, None]
    bins_dict = [np.arange(0, 12, 1), np.arange(50, 400, 35), np.arange(20, 120, 10)]

    with open(path_res + f"params_{target}.json", "r") as fp:
        params = json.load(fp)
    columns = params["col_mol"]
    column_names = ["Selectivity", "Heat capacity (J/mol K)", "Heat of vaporization (kJ/mol)"]
    column_names = ["S$^∞$", "C$_P$ (J/mol K)", "ΔH$_{evp}$ (kJ/mol)"]
    n_col = len(columns)

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(bo_monitor_record[0])

    next_x, cand_x = np.empty((0, n_col)), np.empty((0, n_col))
    for record in bo_monitor_record:
        if record["next_x"] is not None:
            x = np.array(record["next_x"])[0][:, :3]
            next_x = np.vstack((next_x, x))
        if record["fulfillNext"] is not None:
            prop = np.array(record["candidate_props"])
            g1 = np.array(record["next_g1"])
            g2 = np.array(record["next_g2"])

            idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
            cand_x = np.vstack((cand_x, np.array(prop[idx_satisfy])))
    next_x[:, -1] = next_x[:, -1] / scale_hv
    cand_x[:, -1] = cand_x[:, -1] / scale_hv
    df_next = pd.DataFrame(next_x, columns=columns)
    df_cand = pd.DataFrame(cand_x, columns=columns)

    file_mol = path_res + "data/solvent_list.csv"
    df_all = pd.read_csv(file_mol, usecols=columns)
    df_all = df_all.reindex(columns=columns)
    df_all["HV"] = df_all["HV"] / scale_hv
    print(df_all.min())
    print(df_all.max())

    fig, axes = plt.subplots(n_col, n_col, figsize=(3.5 * n_col, 3 * n_col), tight_layout=True)

    def dist(df, color, label):
        for i_ in range(n_col):
            for j_ in range(n_col):
                # If this is the lower-triangule, add a scatterlpot for each group.
                if (i_ >= j_) and (i_ != n_col - 1):
                    axes[i_, j_].scatter(columns[j_], columns[i_ + 1], c=color, s=dot_size_, alpha=0.7, linewidths=0,
                                         data=df, label=label)
                    axes[i_, j_].set_xlim(xylim_dict[j_])
                    axes[i_, j_].set_ylim(xylim_dict[i_ + 1])
                    if xystick_dict[j_] is not None:
                        axes[i_, j_].set_xticks(xystick_dict[j_])
                    # if (xystick_dict[i + 1] is not None) and (i != n_col - 1):
                    #     axes[i, j].set_yticks(xystick_dict[i + 1])

                if i_ == j_ == 0:
                    axes[i_, j_].legend(prop={"size": legend_size_}, loc="upper right")

                if i_ == n_col - 1:
                    axes[i_, j_].hist(columns[j_], bins=bins_dict[j_], density=True, color=color, alpha=0.7, data=df)
                    axes[i_, j_].set_xlim(xylim_dict[j_])
                    if xystick_dict[j_] is not None:
                        axes[i_, j_].set_xticks(xystick_dict[j_])

                if i_ == n_col - 1:
                    axes[i_, j_].set_xlabel(column_names[j_], size=label_size_)
                if (j_ == 0) and (i_ != n_col - 1):
                    axes[i_, j_].set_ylabel(column_names[i_ + 1], size=label_size_)
                if (j_ == 0) and (i_ == n_col - 1):
                    axes[i_, j_].set_ylabel("Probability density")

    dist(df_all, "silver", "Database")
    dist(df_next, "tab:blue", "Hypothesis")
    dist(df_cand, "tab:red", "Candidate")

    for i in range(n_col):
        for j in range(n_col):
            if i < j:
                axes[i, j].remove()
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_3_sol_dist")
    plt.close()


def figure_3B(paths, target, n_sample):
    path_res, path_viz = paths
    print("\nFigure_3b")
    plt.clf()
    plt.rcParams["font.size"] = "14"
    label_size_ = 14.5
    legend_size_ = 12
    dot_size_ = 50

    xylim_dict = [extend_xylim(0, 11), extend_xylim(50, 400), extend_xylim(20, 120)]
    xystick_dict = [[0, 2, 4, 6, 8, 10], None, None]
    bins_dict = [np.arange(0, 12, 1), np.arange(50, 400, 35), np.arange(20, 120, 10)]

    with open(path_res + f"params_{target}.json", "r") as fp:
        params = json.load(fp)
    columns = params["col_mol"]
    column_names = ["Selectivity", "Heat capacity (J/mol K)", "Heat of vaporization (kJ/mol)"]
    column_names = ["S$^∞$", "C$_P$ (J/mol K)", "ΔH$_{evp}$ (kJ/mol)"]
    n_col = len(columns)

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(bo_monitor_record[0])

    file_mol = path_res + "data/solvent_list.csv"
    df_all = pd.read_csv(file_mol, usecols=columns)
    df_all = df_all.reindex(columns=columns)
    df_all["HV"] = df_all["HV"] / scale_hv

    df_low_S = df_all[df_all["S"] < 4]
    df_high_S = df_all[df_all["S"] >= 4]

    print(df_all.min())
    print(df_all.max())

    fig, axes = plt.subplots(n_col, n_col, figsize=(3.5 * n_col, 3 * n_col), tight_layout=True)

    def dist(df, color, label):
        for i_ in range(n_col):
            for j_ in range(n_col):
                # If this is the lower-triangule, add a scatterlpot for each group.
                if (i_ >= j_) and (i_ != n_col - 1):
                    axes[i_, j_].scatter(columns[j_], columns[i_ + 1], c=color, s=dot_size_, alpha=0.7, linewidths=0,
                                         data=df, label=label)
                    axes[i_, j_].set_xlim(xylim_dict[j_])
                    axes[i_, j_].set_ylim(xylim_dict[i_ + 1])
                    if xystick_dict[j_] is not None:
                        axes[i_, j_].set_xticks(xystick_dict[j_])
                    # if (xystick_dict[i + 1] is not None) and (i != n_col - 1):
                    #     axes[i, j].set_yticks(xystick_dict[i + 1])

                if i_ == j_ == 0:
                    axes[i_, j_].legend(prop={"size": legend_size_}, loc="upper right")

                if i_ == n_col - 1:
                    axes[i_, j_].hist(columns[j_], bins=bins_dict[j_], density=True, color=color, alpha=0.7, data=df)
                    axes[i_, j_].set_xlim(xylim_dict[j_])
                    if xystick_dict[j_] is not None:
                        axes[i_, j_].set_xticks(xystick_dict[j_])

                if i_ == n_col - 1:
                    axes[i_, j_].set_xlabel(column_names[j_], size=label_size_)
                if (j_ == 0) and (i_ != n_col - 1):
                    axes[i_, j_].set_ylabel(column_names[i_ + 1], size=label_size_)
                if (j_ == 0) and (i_ == n_col - 1):
                    axes[i_, j_].set_ylabel("Probability density")

    dist(df_low_S, "silver", "Database")
    dist(df_high_S, "tab:blue", "Hypothesis")
    # dist(df_next, "tab:blue", "Hypothesis")
    # dist(df_cand, "tab:red", "Candidate")

    for i in range(n_col):
        for j in range(n_col):
            if i < j:
                axes[i, j].remove()
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_3B_sol_dist")
    plt.close()


def figure_4(paths, target, n_sample):
    path_res, path_viz = paths
    print("\nFigure 4")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    ylim_dict = {"S": extend_xylim(0, 17, onlyUp=True), "CP": extend_xylim(100, 270),
                 "HV": extend_xylim(35, 100), "BP": extend_xylim(80, 320)}
    ylabel_dict = {"S": "S$^∞$", "CP": "C$_P$ (J/mol K)",
                   "HV": "ΔH$_{evp}$ (kJ/mol)", "BP": "T$_b$ (°C)"}

    with open(path_res + f"params_{target}.json", "r") as fp:
        params = json.load(fp)
    columns = params["col_mol"]
    n_col = len(columns)

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(bo_monitor_record[0])

    next_idx, cand_idx, next_x, cand_x = [], [], np.empty((0, n_col)), np.empty((0, n_col))
    for record in bo_monitor_record:
        if record["next_x"] is not None:
            x = np.array(record["next_x"])[0][:, :3]
            next_x = np.vstack((next_x, x))
            next_idx += [record["iteration"]] * x.shape[0]
        if record["fulfillNext"] is not None:
            prop = np.array(record["candidate_props"])
            g1 = np.array(record["next_g1"])
            g2 = np.array(record["next_g2"])

            idx_satisfy = np.where((g1 >= puri_spec) & (g2 >= puri_spec))[0]
            cand_x = np.vstack((cand_x, np.array(prop[idx_satisfy])))
            cand_idx += [record["iteration"]] * len(idx_satisfy)

    next_x[:, -1] = next_x[:, -1] / scale_hv
    cand_x[:, -1] = cand_x[:, -1] / scale_hv
    df_next = pd.DataFrame(next_x, columns=columns)
    df_cand = pd.DataFrame(cand_x, columns=columns)

    # 1: separate
    # for col in columns:
    #     plt.clf()
    #     next_prop, cand_prop = df_next[col].values, df_cand[col].values
    #     markerlinestyle = (0, (2, 1))
    #     for i, prop in zip(next_idx, next_prop):
    #         plt.scatter(i, prop, dot_size, "tab:blue", linewidth=0, zorder=3)
    #     for i, prop in zip(cand_idx, cand_prop):
    #         plt.scatter(i, prop, 1.2 * dot_size, "tab:red", marker="^", linewidth=0, zorder=4)
    #     fake_x, fake_y = 1, -1
    #     plt.scatter(fake_x, fake_y, dot_size, "tab:blue", linewidth=0, label="Hypothesis")
    #     plt.scatter(fake_x, fake_y, 1.2 * dot_size, "tab:red", marker="^", linewidth=0, label="Candidate")
    #
    #     plt.ylim(ylim_dict[col])
    #     plt.xlabel("Iteration", size=label_size)
    #     plt.ylabel(ylabel_dict[col], size=label_size)
    #     plt.legend(prop={"size": legend_size}, ncol=2)
    #     plt.grid(linestyle="dotted")
    #     plt.savefig(path_viz + f"4_prop_track_{col}")

    # 2: combine
    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure()
    for fi, col in enumerate(columns):
        plt.subplot(gs[fi])
        next_prop, cand_prop = df_next[col].values, df_cand[col].values
        for i, prop in zip(next_idx, next_prop):
            plt.scatter(i, prop, dot_size, "tab:blue", linewidth=0, alpha=0.7, zorder=3)
        for i, prop in zip(cand_idx, cand_prop):
            plt.scatter(i, prop, 1.2 * dot_size, "tab:red", marker="^", linewidth=0, alpha=0.7, zorder=4)
        fake_x, fake_y = 1, -1
        plt.scatter(fake_x, fake_y, dot_size, "tab:blue", linewidth=0, label="Hypothesis")
        plt.scatter(fake_x, fake_y, 1.2 * dot_size, "tab:red", marker="^", linewidth=0, label="Candidate")

        plt.xlim(extend_xylim(0, len(bo_monitor_record)))
        plt.ylim(ylim_dict[col])
        plt.ylabel(ylabel_dict[col], size=label_size)
        if fi == 0:
            plt.legend(prop={"size": legend_size})
        plt.grid(linestyle="dotted")
        if fi == n_col - 1:
            plt.xlabel("Iteration", size=label_size)
        else:
            plt.tick_params(axis="x", labelbottom=False)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_4_prop_track")
    plt.close()


def figure_5(paths, target, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    print(f"\n{task} Figure 5")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if task == "CAMPD":
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]
    else:
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]

    if target == "QH":
        yup_dict = {"CAMPD": 400, "CAPD": 200}
    else:
        yup_dict = {"CAMPD": 400, "CAPD": 500}

    n_successes, n_optimizations, success_rates, n_convergences, convergence_rates, y_tr_best_ini = \
        [], [], [], [], [], []
    for n_sample in n_samples:
        suffix = f"{target}_{n_sample}"
        file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
        bo_monitor_record = joblib.load(file_bo_monitor)
        n_success, n_convergence = 0, 0
        for record in bo_monitor_record:
            if record["hasErrorNext"] is not None:
                n_success += len(record["hasErrorNext"])
                n_convergence += len(np.where(record["hasErrorNext"] == 0)[0])
        n_optim = 5 * len(bo_monitor_record)
        success_rate = n_success / n_optim
        n_successes.append(n_success)
        n_optimizations.append(n_optim)
        success_rates.append(success_rate)
        convergence_rate = n_convergence / n_success
        n_convergences.append(n_convergence)
        convergence_rates.append(convergence_rate)
        y_tr_best_ini.append(bo_monitor_record[0]["y_tr_best"])
        print(n_sample, n_optim, n_success, n_convergence, success_rate, convergence_rate)

    bar_width = 26
    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])
    # plt.plot(n_samples, n_optimizations, "o-", ms=1.1*np.sqrt(dot_size), color="tab:orange", zorder=3,
    #          label="Optimization attempt")
    # plt.plot(n_samples, n_successes, "^-", ms=np.sqrt(dot_size), color="tab:blue", zorder=3,
    #          label="Optimization success")
    # plt.plot(n_samples, n_convergences, "s-", ms=0.9 * np.sqrt(dot_size), color="tab:green", zorder=3,
    #          label="Simulation success")
    if task == "CAMPD":
        dot_size_o = dot_size
    else:
        dot_size_o = 1.1 * dot_size
    dot_size_v = 1.2 * dot_size
    dot_size_s = 0.9 * dot_size
    plt.plot(n_samples, n_optimizations, "tab:orange", zorder=3)
    plt.scatter(n_samples, n_optimizations, dot_size_o, "w", edgecolors="tab:orange", marker="o", linewidth=lw,
                zorder=3, label="Optimization attempt")
    plt.plot(n_samples, n_successes, "tab:blue", zorder=3)
    plt.scatter(n_samples, n_successes, dot_size_v, "w", edgecolors="tab:blue", marker="^", linewidth=lw,
                zorder=3, label="Optimization success")
    plt.plot(n_samples, n_convergences, "tab:green", zorder=3)
    plt.scatter(n_samples, n_convergences, dot_size_s, "w", edgecolors="tab:green", marker="s", linewidth=lw,
                zorder=3, label="Simulation success")

    plt.xticks(n_samples)
    plt.xlim(n_samples[0] - 3 * bar_width, n_samples[-1] + 3 * bar_width)
    plt.ylim(extend_xylim(0, yup_dict[task], onlyUp=True))
    plt.ylabel("Frequency", size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)
    # if task == "CAMPD":
    plt.legend(prop={"size": legend_size}, loc="upper right")
    # else:
    #     plt.legend(prop={"size": legend_size}, loc="upper right")

    plt.subplot(gs[1])
    scale_per = 100
    success_rates = np.array(success_rates) * scale_per
    convergence_rates = np.array(convergence_rates) * scale_per
    plt.bar(np.array(n_samples) - bar_width, success_rates, 2 * bar_width, color="tab:blue", alpha=0.8,
            zorder=2, label="Optimization success rate")
    plt.bar(np.array(n_samples) + bar_width, convergence_rates, 2 * bar_width, color="tab:green", alpha=0.8,
            zorder=2, label="Simulation success rate")
    # plt.bar(np.array(n_samples) - bar_width, success_rates, 2 * bar_width, color="tab:blue", ec="tab:blue",
    #         alpha=0.2, zorder=2, label="Optimization success rate")
    # plt.bar(np.array(n_samples) + bar_width, convergence_rates, 2 * bar_width, color="tab:green", ec="tab:green",
    #         alpha=0.2, zorder=2, label="Simulation success rate")
    # plt.bar(np.array(n_samples) - bar_width, success_rates, 2 * bar_width, color="None", ec="tab:blue",
    #         zorder=2)
    # plt.bar(np.array(n_samples) + bar_width, convergence_rates, 2 * bar_width, color="None", ec="tab:green",
    #         zorder=2)
    plt.xticks(n_samples)
    plt.xlim(n_samples[0] - 3 * bar_width, n_samples[-1] + 3 * bar_width)
    plt.ylim(0, 1 * scale_per)
    plt.xlabel("Initial sample size", size=label_size)
    plt.ylabel("Percentage", size=label_size)
    plt.grid(True, linestyle="dotted", zorder=1)
    # if task == "CAMPD":
    plt.legend(prop={"size": legend_size}, loc="upper right")
    # elif task == "CAMPD" and target == "TAC":
    #     plt.legend(prop={"size": legend_size}, loc="upper right")

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_5_success_rate")
    plt.close()


def figure_5B(paths, target, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    print(f"\n{task} Figure 5b")
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 4.5)

    if task == "CAMPD":
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]
    else:
        n_samples = [128, 256, 384, 512, 640, 768, 896, 1024]

    yup_dict = {"CAMPD": 2000, "CAPD": 2000}

    n_simulations, n_convergences, convergence_rates = [], [], []
    for n_sample in n_samples:
        suffix = f"{target}_{n_sample}"
        file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
        bo_monitor_record = joblib.load(file_bo_monitor)
        n_success, n_convergence = 0, 0
        # for record in bo_monitor_record:
        #     if record["hasErrorNext"] is not None:
        #         n_success += len(record["hasErrorNext"])
        #         n_convergence += len(np.where(record["hasErrorNext"] == 0)[0])
        n_simulation = bo_monitor_record[0]["n_simulation"]
        n_convergence = bo_monitor_record[0]["n_tr"]
        convergence_rate = n_convergence / n_simulation
        n_simulations.append(n_simulation)
        n_convergences.append(n_convergence)
        convergence_rates.append(convergence_rate)
        print(n_sample, n_convergence, n_success, convergence_rates)

    bar_width = 32
    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()
    plt.subplot(gs[0])

    plt.plot(n_samples, n_simulations, "k", zorder=3)
    plt.scatter(n_samples, n_simulations, dot_size, "w", edgecolors="k", marker="o", linewidth=lw,
                zorder=3, label="Simulation attempt")
    plt.plot(n_samples, n_convergences, "tab:red", zorder=3)
    plt.scatter(n_samples, n_convergences, 1.2 * dot_size, "w", edgecolors="tab:red", marker="^", linewidth=lw,
                zorder=3, label="Simulation success")
    plt.xticks(n_samples)
    plt.xlim(extend_xylim(n_samples[0] - bar_width, n_samples[-1] + bar_width))
    plt.ylim(extend_xylim(0, yup_dict[task], onlyUp=True))
    plt.ylabel("Frequency", size=label_size)
    plt.grid(linestyle="dotted", zorder=1)
    plt.tick_params(axis="x", labelbottom=False)
    plt.legend(prop={"size": legend_size}, loc="upper left")

    plt.subplot(gs[1])
    plt.bar(n_samples, convergence_rates, 3 * bar_width, color="tab:red", alpha=0.8, zorder=2)
    plt.xticks(n_samples)
    plt.xlim(extend_xylim(n_samples[0] - bar_width, n_samples[-1] + bar_width))
    plt.ylim(0, 1)
    plt.xlabel("Initial sample size", size=label_size)
    plt.ylabel("Convergence rate", size=label_size)
    plt.grid(True, linestyle="dotted", zorder=1)

    plt.subplots_adjust(hspace=0.1)
    fig.align_labels()
    plt.savefig(path_viz + f"{target}_5B_convergence_rate")
    plt.close()


def figure_0(paths, target, n_sample, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 0")
    plt.clf()

    xytick_dict = {"y": None, "g1": None, "g2": None}
    xylim_dict = {"g1": [0, 1], "g2": [0, 1]}
    xlabel_dict = {"g1": "C$_4$H$_8$ purity", "g2": "C$_4$H$_6$ purity"}
    ylabel_dict = {"g1": "Estimated C$_4$H$_8$ purity", "g2": "Estimated C$_4$H$_6$ purity"}
    if target == "QH":
        xlabel_dict["y"] = "Q$_H$ (MW)"
        ylabel_dict["y"] = "Estimated Q$_H$ (MW)"
        if task == "CAMPD":
            xytick_dict["y"] = [0, 50, 100, 150, 200, 250, 300]
            xylim_dict["y"] = [0, 300]
        else:
            xylim_dict["y"] = [0, 80]
    else:
        xlabel_dict["y"] = "TAC (MM$/yr)"
        ylabel_dict["y"] = "Estimated TAC (MM$/yr)"
        if task == "CAMPD":
            xylim_dict["y"] = [0, 80]
        else:
            xylim_dict["y"] = [0, 30]

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(len(bo_monitor_record))

    data_file = path_res + f"data/train_data_{target}_{n_sample}.csv"
    df_data = pd.read_csv(data_file)

    y, yhat = df_data["y"].values / scale_y, df_data["yhat"].values / scale_y
    g1, g1hat = logit_inverse(df_data["g1"].values), logit_inverse(df_data["g1hat"].values)
    g2, g2hat = logit_inverse(df_data["g2"].values), logit_inverse(df_data["g2hat"].values)

    def parity_without_uncertainty(label_, y_list_, yhat_list_):
        plt.clf()
        plt.scatter(y_list_, yhat_list_, c="tab:blue", alpha=1, linewidths=0, zorder=3)
        plt.plot(xylim_dict[label_], xylim_dict[label_], c="k", zorder=2)
        plt.grid(linestyle="dotted", zorder=1)
        plt.xlim(xylim_dict[label_])
        plt.ylim(xylim_dict[label_])
        if xytick_dict[label_]:
            plt.xticks(xytick_dict[label_])
            plt.yticks(xytick_dict[label_])
        plt.xlabel(xlabel_dict[label_], size=label_size)
        plt.ylabel(ylabel_dict[label_], size=label_size)
        plt.savefig(path_viz + f"{target}_6_parity_{label_}")

    for label, y_list, yhat_list in zip(["y", "g1", "g2"],
                                        [y, g1, g2],
                                        [yhat, g1hat, g2hat]):
        parity_without_uncertainty(label, y_list, yhat_list)

    plt.close()


def figure_0B(paths, target, n_sample, solvent=None):
    path_res, path_viz = paths
    if "CAMPD" in path_viz:
        task = "CAMPD"
    else:
        task = "CAPD"
    scale_y = 1e3 if target == "QH" else 1e6
    print(f"\n{task} Figure 6B")
    plt.clf()

    if target == "QH":
        xylim_dict = {"y": extend_xylim(0, 80, onlyUp=True),
                      "g1": extend_xylim(0, 12, onlyUp=True), "g2": extend_xylim(0, 8, onlyUp=True)}
        xlabel_dict = {"y": "Q$_H$ (MW)", "g1": "C$_4$H$_8$ purity", "g2": "C$_4$H$_6$ purity"}
        ylabel_dict = {"y": "Estimated Q$_H$ (MW)",
                       "g1": "Estimated C$_4$H$_8$ purity", "g2": "Estimated C$_4$H$_6$ purity"}
    else:
        xylim_dict = {"CAMPD": extend_xylim(0, 30, onlyUp=True)}
        xlabel_dict = {"y": "TAC (MM$/yr)", "CAPD": extend_xylim(2, 6)}
        ylabel_dict = {"y": "Estimated TAC (MM$/yr)", "CAPD": extend_xylim(2, 6)}

    suffix = f"{target}_{n_sample}"
    file_bo_monitor = path_res + f"optim_monitor_{suffix}.pkl"
    bo_monitor_record = joblib.load(file_bo_monitor)
    print(list(bo_monitor_record[0].keys()))
    print(len(bo_monitor_record))

    next_y, next_yhat, next_yhat_std, next_g1, next_g1hat, next_g1hat_std, next_g2, next_g2hat, next_g2hat_std = \
        [], [], [], [], [], [], [], [], []
    for record in bo_monitor_record:
        if np.any(record["hasErrorNext"] == 0):
            y = np.array(record["next_y"]) / scale_y
            yhat = np.array(record["next_yhat"])[1] / scale_y
            yhat_std = np.array(record["next_yhat_std"])[1] / scale_y
            g1 = np.array(record["next_g1"])
            g1hat = np.array(record["next_g1hat"])[1]
            g1hat_std = np.array(record["next_g1hat_std"])[1]
            g2 = np.array(record["next_g2"])
            g2hat = np.array(record["next_g2hat"])[1]
            g2hat_std = np.array(record["next_g2hat_std"])[1]
            # print(y, yhat, g1, g1hat, g2, g2hat)

            # idx_satisfy = np.where((g1 >= puri_spec) & (g1 >= puri_spec))[0]
            # idx_best = idx_satisfy[np.argmin(y[idx_satisfy])]

            next_y = np.append(next_y, y)
            next_yhat = np.append(next_yhat, yhat)
            next_yhat_std = np.append(next_yhat_std, yhat_std)
            next_g1 = np.append(next_g1, g1)
            next_g1hat = np.append(next_g1hat, g1hat)
            next_g1hat_std = np.append(next_g1hat_std, g1hat_std)
            next_g2 = np.append(next_g2, g2)
            next_g2hat = np.append(next_g2hat, g2hat)
            next_g2hat_std = np.append(next_g2hat_std, g2hat_std)

    print(next_y, next_yhat)
    print(r2_score(next_y, next_yhat))

    def parity_with_uncertainty(label_, y_list_, yhat_list_, yhat_std_list_):
        plt.clf()
        n_total, n_within1, n_within2, n_within3 = 0, 0, 0, 0
        for y_, yhat_, yhat_std_ in zip(y_list_, yhat_list_, yhat_std_list_):
            n_total += 1
            if (yhat_ - yhat_std_) <= y_ <= (yhat_ + yhat_std_):
                n_within1 += 1
                ci, zi = "tab:red", 5
            elif (yhat_ - 2 * yhat_std_) <= y_ <= (yhat_ + 2 * yhat_std_):
                n_within2 += 1
                ci, zi = "tab:green", 4
            elif (yhat_ - 3 * yhat_std_) <= y_ <= (yhat_ + 3 * yhat_std_):
                n_within3 += 1
                ci, zi = "tab:blue", 3
            else:
                ci, zi = "tab:gray", 2
            plt.scatter(y_, yhat_, c=ci, alpha=1, linewidths=0, zorder=zi)
            plt.errorbar(y_, yhat_, yerr=yhat_std_, capsize=2, c=ci, linestyle="", zorder=zi)

        plt.plot(xylim_dict[label_], xylim_dict[label_], c="k", zorder=2)
        plt.grid(linestyle="dotted", zorder=1)
        plt.xlim(xylim_dict[label_])
        plt.ylim(xylim_dict[label_])
        plt.xlabel(xlabel_dict[label_], size=label_size)
        plt.ylabel(ylabel_dict[label_], size=label_size)

        print(n_within1, n_within2, n_within3, n_total)
        plt.savefig(path_viz + f"{target}_6B_parity_{label_}")

    for label, y_list, yhat_list, yhat_std_list in zip(["y", "g1", "g2"],
                                                       [next_y, next_g1, next_g2],
                                                       [next_yhat, next_g1hat, next_g2hat],
                                                       [next_yhat_std, next_g1hat_std, next_g2hat_std]):
        parity_with_uncertainty(label, y_list, yhat_list, yhat_std_list)

    plt.close()
