# Created on 25 May 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Computer-Aided Molecular and Process Design using Bayesian Optimization
# Gaussian process regression + Bayesian optimization + Automated process simulation


from utility_params import *
from utility_aspen import *
from utility_model import *
from utility_bayesopt import *
from utility_common import *
from process_simulation_CAPD import run_simulation_CAPD
from process_simulation_CAMPD import run_simulation_CAMPD

import time
import json
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, NonlinearConstraint

warnings.filterwarnings("ignore")


def main(params, obj="MIN", constr=("DIST_C4H8_T1", "DIST_C4H6_T2")):
    # <editor-fold desc="logger setup">
    file_log = params["file_log"]
    with open(file_log, "w") as f:
        f.write("")
    sys.stdout = Logger(file_log)
    sys.stderr = Logger(file_log)
    # </editor-fold>

    timestamp()
    time_total_start = time.time()

    # <editor-fold desc="script backup">
    path_script = params["path_script"]
    if os.path.exists(path_script):
        shutil.rmtree(path_script)
    shutil.copytree("../script", path_script)
    # </editor-fold>

    # <editor-fold desc="load solvent list and save parameters">
    n_mol = len(params["col_mol"])
    params, mol_list, mol_property, mol_property_dict = get_mol(params)
    with open(params["file_params"], "w") as fp:
        json.dump(params, fp, indent=4)
    # </editor-fold>

    # <editor-fold desc="configuration">
    print(params)
    useLogit = params["useLogit"]
    puri_lb, puri_ub, puri_spec = params["puri_lb"], params["puri_ub"], params["puri_spec"]
    print(f"* purity specification: {puri_spec}")
    if useLogit:
        puri_lb_, puri_ub_, puri_spec_ = logit(puri_lb), logit(puri_ub), logit(puri_spec)
        print(f"* logit of purity specification: {puri_spec_}")
    else:
        puri_lb_, puri_ub_, puri_spec_ = puri_spec, puri_lb, puri_ub

    target, strategy, considerSolvent = params["target"], params["strategy"], params["considerSolvent"]
    saveInitialData = params["saveInitialData"]
    random_seed = params["random_seed"]
    sampling_method = params["sampling_method"]
    n_iter, ac_mode = params["n_iter"], params["ac_mode"]
    file_bkp = params["file_bkp"]
    path_out, path_data = params["path_out"], params["path_data"]
    list_n_sample = params["list_n_sample"]
    active_list_n_sample = params["active_list_n_sample"]
    n_reconnect = params["n_reconnect"]
    # </editor-fold>

    for n_sample in active_list_n_sample:
        time_initial_labeling, time_extra_labeling, time_modeling, time_optimization, n_simulation = 0, 0, 0, 0, 0
        np.random.seed(random_seed)
        print(f"\n# initial samples: {n_sample}")

        # <editor-fold desc="data storage preparation">
        data_suffix = f"{n_sample}"
        suffix = f"{target}_{n_sample}"
        file_simu = path_out + f"simulation_{suffix}.txt"
        file_optim_monitor = path_out + f"optim_monitor_{suffix}.pkl"
        with open(file_simu, "w") as f:
            f.write("")
        file_optim_record = path_out + f"optim_record_{suffix}.csv"
        columns = ["n_sample", "iteration", "n_simulation", "n_tr",
                   "time_initial_labeling", "time_extra_labeling", "time_modeling", "time_optimization",
                   "r2_g1", "r2_g2", "r2_y",
                   "x_tr_best", "y_tr_best", "gotBetterSample", "optimSuccess",
                   "ac_mode", "next_af", "x_mean", "x_std", "next_x",
                   "next_y1", "next_y2",
                   "next_g1hat", "next_g1hat_std", "next_g1",
                   "next_g2hat", "next_g2hat_std", "next_g2",
                   "next_yhat", "next_yhat_std", "next_y", "next_error",
                   "hasErrorNext", "fulfillNext",
                   "next_alias", "candidate_props", "optim_counter"]
        write_csv(file_optim_record, columns)
        # </editor-fold>

        if saveInitialData:
            # <editor-fold desc="initial data generation">
            time_ini_start = time.time()
            data_suffix_ = None
            if (sampling_method == "random") and (n_sample != list_n_sample[0]):
                idx_n_sample = np.where((np.array(list_n_sample) == n_sample))[0].item()
                n_sample_prev = list_n_sample[idx_n_sample - 1]
                data_suffix_ = f"{n_sample_prev}"
                random_state, n_simulation, time_initial_labeling = joblib.load(path_data + f"state_{data_suffix_}.pkl")
                np.random.set_state(random_state)
                print(f"\t-> success(attempt): {n_sample_prev}({n_simulation})")
                n_sample_ = n_sample - n_sample_prev
                mol_alias, x_pro_ini = sampling(params, n_sample_, mol_list, sampling_method)
                n_simulation += n_sample_
            else:
                mol_alias, x_pro_ini = sampling(params, n_sample, mol_list, sampling_method)
                n_simulation = n_sample

            # load Aspen simulation
            aspen_plus = AspenPlusInterface(file_bkp)
            aspen_plus.load_bkp()
            if considerSolvent:
                (g1_ini, y1_ini, g2_ini, y2_ini, hasError_ini, _, _, tac_ini), time_cost = \
                    run_simulation_CAMPD(aspen_plus, mol_alias, x_pro_ini, file_simu, "0_INI", n_reconnect)
            else:
                (g1_ini, y1_ini, g2_ini, y2_ini, hasError_ini, _, _, tac_ini), time_cost = \
                    run_simulation_CAPD(aspen_plus, mol_alias, x_pro_ini, file_simu, "0_INI")
            aspen_plus.close_bkp()
            time_initial_labeling += time_cost

            if useLogit:
                g1_ini, g2_ini = logit(g1_ini), logit(g2_ini)
            idx_valid = np.where((hasError_ini == 0) &
                                 ((puri_lb_ <= g1_ini) & (g1_ini <= puri_ub_)) &
                                 ((puri_lb_ <= g2_ini) & (g2_ini <= puri_ub_)))[0]
            mol_alias_ini = [mol_alias[idx] for idx in idx_valid]
            x_pro_ini = x_pro_ini[idx_valid, :]
            g1_ini = g1_ini[idx_valid]
            g2_ini = g2_ini[idx_valid]
            y1_ini = y1_ini[idx_valid]
            y2_ini = y2_ini[idx_valid]
            duty_ini = y1_ini + y2_ini
            tac_ini = tac_ini[idx_valid]

            if (sampling_method == "random") and (n_sample != list_n_sample[0]):
                inputs_prev = np.load(path_data + f"initial_{data_suffix_}.npz", allow_pickle=True)
                mol_alias_prev = list(inputs_prev["mol_alias"])
                x_pro_prev = inputs_prev["x_pro"]
                g1_prev = inputs_prev["g1"]
                g2_prev = inputs_prev["g2"]
                duty_prev = inputs_prev["duty"]
                tac_prev = inputs_prev["tac"]

                mol_alias_ini = mol_alias_prev + mol_alias_ini
                x_pro_ini = np.vstack((x_pro_prev, x_pro_ini))
                g1_ini = np.append(g1_prev, g1_ini)
                g2_ini = np.append(g2_prev, g2_ini)
                duty_ini = np.append(duty_prev, duty_ini)
                tac_ini = np.append(tac_prev, tac_ini)

            print(f"\t-> success(attempt)/demand: {x_pro_ini.shape[0]}({n_simulation})/{n_sample}")
            # </editor-fold>

            # <editor-fold desc="data supplementation">
            sampleMore = True  # sampling more to find satisfactory samples in the training data
            aspen_plus = AspenPlusInterface(file_bkp)
            aspen_plus.load_bkp()
            while (x_pro_ini.shape[0] < n_sample) or sampleMore:
                inp = (mol_alias_ini, x_pro_ini, g1_ini, g2_ini, duty_ini, tac_ini)
                outp, time_sampling, last_alias = \
                    valid_sampling(params, aspen_plus, mol_list, inp, file_simu, "0_INI")
                mol_alias_ini, x_pro_ini, g1_ini, g2_ini, duty_ini, tac_ini = outp
                n_simulation += 1
                time_initial_labeling += time_sampling
                idx_satisfy = np.where((puri_spec_ <= g1_ini) & (puri_spec_ <= g2_ini))[0]
                if len(idx_satisfy) > 0:
                    sampleMore = False
                print(f"\t-> success(attempt)/demand: {x_pro_ini.shape[0]}({n_simulation})/{n_sample}")
                if n_simulation % n_reconnect == 0:
                    aspen_plus.close_bkp()
                    aspen_plus = AspenPlusInterface(file_bkp)
                    aspen_plus.load_bkp()
            time_ini_end = time.time()
            aspen_plus.close_bkp()
            print(f"-> Time Cost of Initial Labeling: {time_ini_end - time_ini_start:.1f} s")
            random_state = np.random.get_state()
            # </editor-fold>

            # <editor-fold desc="save and load initial data">
            np.savez(path_data + f"initial_{data_suffix}.npz",
                     mol_alias=mol_alias_ini,
                     x_pro=x_pro_ini,
                     g1=g1_ini,
                     g2=g2_ini,
                     duty=duty_ini,
                     tac=tac_ini)
            joblib.dump((random_state, n_simulation, time_initial_labeling), path_data + f"state_{data_suffix}.pkl")
            # </editor-fold>

        else:
            inputs = np.load(path_data + f"initial_{data_suffix}.npz", allow_pickle=True)
            mol_alias_ini = list(inputs["mol_alias"])
            x_pro_ini = inputs["x_pro"]
            g1_ini = inputs["g1"]
            g2_ini = inputs["g2"]
            duty_ini = inputs["duty"]
            tac_ini = inputs["tac"]

            random_state, n_simulation, time_initial_labeling = joblib.load(path_data + f"state_{data_suffix}.pkl")
            np.random.set_state(random_state)

        # start Bayesian optimization
        time_bo_start = time.time()

        # model definition
        model_y = GPRegressorSklearn(params, target, obj=obj)
        model_g1 = GPRegressorSklearn(params, constr[0])
        model_g2 = GPRegressorSklearn(params, constr[1])

        # optimization
        addNext = False
        mol_alias_tr = x_tr = x_pro_tr = g1_tr = g2_tr = duty_tr = tac_tr = None
        x_tr_best = y_tr_best = None
        next_alias = next_x_pro = next_g1 = next_g2 = next_duty = next_tac = None

        bo_converge = BOConvergence()
        bo_monitor = BOMonitor()
        idx_iter = 0
        while idx_iter < n_iter:
            gotBetterSample = None
            idx_iter += 1
            print(f"\n# Iteration {idx_iter}")
            # <editor-fold desc="data preparation">
            if idx_iter == 1:
                mol_alias_tr = mol_alias_ini
                x_pro_tr = x_pro_ini
                if considerSolvent:
                    x_mol_tr = np.array([mol_property_dict[mol] for mol in mol_alias_tr])
                    x_tr = np.hstack((x_mol_tr, x_pro_tr))
                else:
                    x_tr = x_pro_tr
                duty_tr, tac_tr = duty_ini, tac_ini
                g1_tr, g2_tr = g1_ini, g2_ini
            else:
                if addNext:
                    mol_alias_tr = mol_alias_tr + next_alias
                    x_pro_tr = np.vstack((x_pro_tr, next_x_pro))
                    if considerSolvent:
                        x_mol_tr = np.array([mol_property_dict[mol] for mol in mol_alias_tr])
                        x_tr = np.hstack((x_mol_tr, x_pro_tr))
                    else:
                        x_tr = x_pro_tr
                    g1_tr = np.append(g1_tr, next_g1)
                    g2_tr = np.append(g2_tr, next_g2)
                    duty_tr = np.append(duty_tr, next_duty)
                    tac_tr = np.append(tac_tr, next_tac)
                    print(f"-> Previous suggested sample added")
                else:
                    print("-> Extra sampling needed")
                    aspen_plus = AspenPlusInterface(file_bkp)
                    aspen_plus.load_bkp()
                    n_tr_ = x_tr.shape[0]
                    n_simulation_ = n_simulation
                    while x_tr.shape[0] < n_tr_ + 10:
                        inp = (mol_alias_tr, x_pro_tr, g1_tr, g2_tr, duty_tr, tac_tr)
                        outp, time_sampling, last_alias = \
                            valid_sampling(params, aspen_plus, mol_list, inp, file_simu, f"2_EXT_{idx_iter}")
                        mol_alias_tr, x_pro_tr, g1_tr, g2_tr, duty_tr, tac_tr = outp
                        if considerSolvent:
                            x_mol_tr = np.array([mol_property_dict[mol] for mol in mol_alias_tr])
                            x_tr = np.hstack((x_mol_tr, x_pro_tr))
                        else:
                            x_tr = x_pro_tr
                        n_simulation += 1
                        print(f"\t-> success(attempt): {x_tr.shape[0]}({n_simulation})")
                        time_extra_labeling += time_sampling
                        if (n_simulation - n_simulation_) % n_reconnect == 0:
                            aspen_plus.close_bkp()
                            aspen_plus = AspenPlusInterface(file_bkp)
                            aspen_plus.load_bkp()
                    aspen_plus.close_bkp()

            y_tr = None
            if target == "QH":
                y_tr = duty_tr
            elif target == "TAC":
                y_tr = tac_tr

            if (idx_iter != 1) and (not addNext):
                isBetter = None
                if target == "QH":
                    isBetter = (puri_spec_ <= g1_tr) & (puri_spec_ <= g2_tr) & (duty_tr < bo_converge.best)
                elif target == "TAC":
                    isBetter = (puri_spec_ <= g1_tr) & (puri_spec_ <= g2_tr) & (tac_tr < bo_converge.best)
                if np.any(isBetter):
                    print(f"\t-> A better solution found by extra sampling")
                    gotBetterSample = True
                else:
                    gotBetterSample = False

            if obj == "MIN":
                idx_satisfy = np.where((puri_spec_ <= g1_tr) & (puri_spec_ <= g2_tr))[0]
                idx_best = idx_satisfy[np.argmin(y_tr[idx_satisfy])]
                x_tr_best, g1_tr_best, g2_tr_best, y_tr_best = \
                    x_tr[idx_best], g1_tr[idx_best], g2_tr[idx_best], y_tr[idx_best]
                model_g1.y_tr_best = y_tr_best
                x_tr_best = x_tr_best.tolist()
                print(f"\t* the best for now: {y_tr_best:.4f}, "
                      f"g1: {g1_tr_best:.4f}/{puri_spec_:.4f}, g2: {g2_tr_best:.4f}/{puri_spec_:.4f}")
                print(f"\t  with the solvent {mol_alias_tr[idx_best]}")
                print(f"\t  and the corresponding variables: {x_tr_best}")
                bo_converge.update_best(y_tr_best)
            elif obj == "MAX":
                print("Warning: updates required ...")
                quit()
            # </editor-fold>

            # <editor-fold desc="modeling">
            n_tr = x_tr.shape[0]
            r2_g1, g1hat_tr, time_fit_g1 = model_g1.fit(x_tr, g1_tr)
            r2_g2, g2hat_tr, time_fit_g2 = model_g2.fit(x_tr, g2_tr)
            r2_y, yhat_tr, time_fit_y = model_y.fit(x_tr, y_tr)
            print(f"\t* with {n_tr}({n_simulation}) samples, model R2: {r2_y:.4f}, {r2_g1:.4f}, {r2_g2:.4f}")
            time_modeling += (time_fit_g1 + time_fit_g2 + time_fit_y)
            # </editor-fold>

            # <editor-fold desc="optimization">
            bound = params["bound"]
            integrality = params["integrality"]

            def constr_1(x_):
                x_ = x_.reshape(1, -1)
                ghat, ghat_std = model_g1.predict(x_, return_std=True)
                return ghat

            def constr_2(x_):
                x_ = x_.reshape(1, -1)
                ghat, ghat_std = model_g2.predict(x_, return_std=True)
                return ghat

            nlc1 = NonlinearConstraint(constr_1, puri_spec_, np.inf)
            nlc2 = NonlinearConstraint(constr_2, puri_spec_, np.inf)
            constraints = (nlc1, nlc2)

            candidate, acq_value = [], []
            for idx_trial in range(params["n_trial"]):
                seed_ = random_seed + idx_trial
                idx_trial += 1
                time_opt_start = time.time()
                result = differential_evolution(acquisition_func, bound,
                                                args=(params, model_y, y_tr_best),
                                                seed=seed_, constraints=constraints, integrality=integrality)
                time_opt_end = time.time()
                time_optimization += (time_opt_end - time_opt_start)
                trailSuccess = result.success
                print(f"\t# Trial {idx_trial}: Success {trailSuccess} / {result.message}")
                if trailSuccess:
                    candidate.append(result.x)
                    next_af = None
                    if (ac_mode == "PI") or (ac_mode == "EI"):
                        next_af = -result.fun
                    elif ac_mode == "UCB":
                        if obj == "MAX":
                            next_af = -result.fun
                        elif obj == "MIN":
                            next_af = result.fun
                    elif ac_mode == "MIN":
                        next_af = result.fun
                    acq_value.append(next_af)
            candidate = np.array(candidate)
            if len(candidate) == 0:
                optimSuccess = False
            else:
                optimSuccess = True
            # </editor-fold>

            # <editor-fold desc="validation">
            idx_fulfill_best = fulfillNext = None
            output = next_alias = next_x_pro = next_duty = next_tac = None
            next_g1hat = next_g1hat_ = next_g1hat_std = next_g1hat_std_ = np.array([None])
            next_g2hat = next_g2hat_ = next_g2hat_std = next_g2hat_std_ = np.array([None])
            next_yhat = next_yhat_ = next_yhat_std = next_yhat_std_ = np.array([None])
            next_g1 = next_g2 = next_y1 = next_y2 = next_y = next_error = np.array([None])
            if not optimSuccess:
                addNext = False
                if strategy == "Bayesian":
                    bo_converge(np.inf)
                    output_1 = [n_sample, idx_iter, n_simulation, n_tr,
                                time_initial_labeling, time_extra_labeling, time_modeling, time_optimization,
                                r2_g1, r2_g2, r2_y,
                                x_tr_best, y_tr_best, gotBetterSample, optimSuccess]
                    output_2 = [bo_converge.counter]
                    output = output_1 + [None] * (len(columns) - len(output_1) - len(output_2)) + output_2
            else:
                next_x = candidate
                print(f"\t* next point at {next_x.tolist()}")

                if considerSolvent:
                    next_alias, candidate_props = [], []
                    next_x_pro = next_x[:, n_mol:]
                    mol_mean, mol_std = model_g1.x_mean[:n_mol], model_g1.x_std[:n_mol]
                    for x in next_x:
                        next_x_mol = x[:n_mol].reshape(1, -1)
                        next_x_mol_ = (next_x_mol - mol_mean) / mol_std
                        distance_dict = {}
                        for idx, (alias, prop) in enumerate(zip(mol_list, mol_property)):
                            prop_ = (prop.reshape(1, -1) - mol_mean) / mol_std
                            dist = euclidean_distance(next_x_mol_ - prop_)
                            distance_dict[idx] = dist
                        sort_dist_dict = {k: v for (k, v) in sorted(distance_dict.items(), key=lambda x_: x_[1])}
                        idx_optimal = list(sort_dist_dict.keys())[0]
                        next_alias.append(mol_list[idx_optimal])
                        candidate_props.append(mol_property[idx_optimal])
                    candidate_props = np.array(candidate_props)
                    print(f"\t* next candidate: {next_alias}, {[prop.tolist() for prop in candidate_props]}")
                    print(f"\t- property mean & std: {mol_mean.tolist()}, {mol_std.tolist()}")
                    if strategy == "OneShot":
                        n_reconn = 5
                    else:
                        n_reconn = n_reconnect
                    aspen_plus = AspenPlusInterface(file_bkp)
                    aspen_plus.load_bkp()
                    (puri1, duty1, puri2, duty2, hasError, _, _, tac), time_cost = \
                        run_simulation_CAMPD(aspen_plus, next_alias, next_x_pro,
                                             file_simu, f"1_VAL_{idx_iter}", n_reconn)
                    next_x_ = np.hstack((candidate_props, next_x_pro))
                else:
                    next_x_pro = next_x_ = next_x
                    next_alias = [params["solvent"]] * next_x.shape[0]
                    aspen_plus = AspenPlusInterface(file_bkp)
                    aspen_plus.load_bkp()
                    (puri1, duty1, puri2, duty2, hasError, _, _, tac), time_cost = \
                        run_simulation_CAPD(aspen_plus, next_alias, next_x_pro, file_simu, f"1_VAL_{idx_iter}")
                    candidate_props = np.empty((next_x.shape[0], n_mol))
                n_simulation += 1
                time_extra_labeling += time_cost
                aspen_plus.close_bkp()

                hasErrorNext = hasError
                idx_feasible = np.where((hasErrorNext == 0) & ((puri_lb <= puri1) & (puri1 <= puri_ub)) & (
                        (puri_lb <= puri2) & (puri2 <= puri_ub)))[0]
                if len(idx_feasible) == 0:
                    addNext = False
                    bo_converge(np.inf)
                else:
                    addNext = True
                    next_x_pro = next_x_pro[idx_feasible, :]
                    next_y1, next_y2 = duty1[idx_feasible], duty2[idx_feasible]
                    next_duty = next_y1 + next_y2
                    next_tac = tac[idx_feasible]
                    next_g1, next_g2 = puri1[idx_feasible], puri2[idx_feasible]
                    next_alias = [next_alias[idx] for idx in idx_feasible]
                    candidate_props = candidate_props[idx_feasible, :]
                    if useLogit:
                        next_g1, next_g2 = logit(next_g1), logit(next_g2)
                    if target == "QH":
                        next_y = next_duty
                    elif target == "TAC":
                        next_y = next_tac
                    idx_fulfill = np.where((next_y <= y_tr_best) & (puri_spec_ <= next_g1) & (puri_spec_ <= next_g2))[0]
                    fulfillNext = True if len(idx_fulfill) > 0 else False

                    next_g1hat, next_g1hat_std = model_g1.predict(next_x[idx_feasible, :], return_std=True)
                    next_g2hat, next_g2hat_std = model_g2.predict(next_x[idx_feasible, :], return_std=True)
                    next_yhat, next_yhat_std = model_y.predict(next_x[idx_feasible, :], return_std=True)

                    next_g1hat_, next_g1hat_std_ = model_g1.predict(next_x_[idx_feasible, :], return_std=True)
                    next_g2hat_, next_g2hat_std_ = model_g2.predict(next_x_[idx_feasible, :], return_std=True)
                    next_yhat_, next_yhat_std_ = model_y.predict(next_x_[idx_feasible, :], return_std=True)
                    next_error = np.abs(next_y - next_yhat)
                    print(f"\t* {ac_mode}: {acq_value}, "
                          f"y_hat(std) / y: {next_yhat.tolist()}({next_yhat_std.tolist()}) / {next_y.tolist()}, "
                          f"error: {next_error.tolist()}, "
                          f"g1_hat: {next_g1hat.tolist()}, g2_hat:{next_g2hat.tolist()}, fulfill: {fulfillNext}")

                    if fulfillNext:
                        idx_fulfill_best = idx_fulfill[np.argmin(next_y[idx_fulfill])]
                        bo_converge(next_y[idx_fulfill_best])
                    else:
                        bo_converge(np.inf)

                output = [n_sample, idx_iter, n_simulation, n_tr,
                          time_initial_labeling, time_extra_labeling, time_modeling, time_optimization,
                          r2_g1, r2_g2, r2_y,
                          x_tr_best, y_tr_best, gotBetterSample, optimSuccess,
                          ac_mode, acq_value,
                          model_g1.x_mean.tolist(), model_g1.x_std.tolist(), (next_x.tolist(), next_x_.tolist()),
                          next_y1.tolist(), next_y2.tolist(),
                          (next_g1hat.tolist(), next_g1hat_.tolist()),
                          (next_g1hat_std.tolist(), next_g1hat_std_.tolist()), next_g1.tolist(),
                          (next_g2hat.tolist(), next_g2hat_.tolist()),
                          (next_g2hat_std.tolist(), next_g2hat_std_.tolist()), next_g2.tolist(),
                          (next_yhat.tolist(), next_yhat_.tolist()),
                          (next_yhat_std.tolist(), next_yhat_std_.tolist()), next_y.tolist(), next_error.tolist(),
                          hasErrorNext, fulfillNext,
                          next_alias, candidate_props, bo_converge.counter]

            print(f"\t-> Time Cost of Initial Labeling, Extra Labeling, Modeling, and Optimization: "
                  f"{time_initial_labeling:.1f} s, {time_extra_labeling:.1f} s, "
                  f"{time_modeling:.1f} s, {time_optimization:.1f} s")
            # </editor-fold>

            file_tr_data = path_data + f"train_data_{suffix}.csv"
            if bo_converge.counter == 0:
                bo_converge.optimum = (idx_iter, next_alias[idx_fulfill_best], next_x_pro[idx_fulfill_best].tolist(),
                                       next_y[idx_fulfill_best], next_g1[idx_fulfill_best], next_g2[idx_fulfill_best])
                df = pd.DataFrame(list(zip(mol_alias_tr, x_tr, g1_tr, g2_tr, y_tr,
                                           g1hat_tr, g2hat_tr, yhat_tr)),
                                  columns=["alias", "x", "g1", "g2", "y",
                                           "g1hat", "g2hat", "yhat"])
                df.to_csv(file_tr_data)
            bo_monitor.update(columns, output)
            write_csv(file_optim_record, output, "a")
            if bo_converge.stopBO:
                break

        print(f"* optimal solution: {bo_converge.optimum}")
        time_bo_end = time.time()
        print(f"-> Time Cost of {strategy} optimization: {time_bo_end - time_bo_start:.1f} s")
        bo_monitor.save(file_optim_monitor)
        timestamp()

    time_total_end = time.time()
    print(f"-> Total Time Cost: {time_total_end - time_total_start:.1f} s")
    timestamp()

    time.sleep(10)
    kill_aspen_hard()


if __name__ == "__main__":
    # <editor-fold desc="CAMPD towards low TAC">
    # main(get_params("QH", "OneShot", considerSolvent=True, saveInitialData=True))
    # time.sleep(10)
    # main(get_params("QH", "Bayesian", considerSolvent=True, saveInitialData=False))
    # time.sleep(10)
    # main(get_params("TAC", "Bayesian", considerSolvent=True, saveInitialData=False))
    # time.sleep(10)
    # main(get_params("TAC", "OneShot", considerSolvent=True, saveInitialData=False))
    # </editor-fold>

    # <editor-fold desc="CAPD using the identified solvent">
    # solvents = ["CH2I2", "C2H4O2-D1", "C2H6O2", "C5H9NO-D2", "C7H7NO3"]  # TAC
    # solvents = ["CH2I2", "C2H4O2-D1", "C2H6OS", "C4H4N2", "C5H9NO-D2", "C7H7NO2-D1", "C7H7NO3"]  # QH

    # solvent = "CH2I2"
    # main(get_params("QH", "Bayesian", considerSolvent=False, saveInitialData=True, solvent=solvent))
    # time.sleep(10)
    # main(get_params("QH", "OneShot", considerSolvent=False, saveInitialData=False, solvent=solvent))
    # time.sleep(10)

    # solvent = "C2H6O2"
    # main(get_params("TAC", "Bayesian", considerSolvent=False, saveInitialData=True, solvent=solvent))
    # time.sleep(10)

    # solvent = "C5H9NO-D2"
    # main(get_params("QH", "Bayesian", considerSolvent=False, saveInitialData=True, solvent=solvent))
    # time.sleep(10)
    # main(get_params("TAC", "Bayesian", considerSolvent=False, saveInitialData=False, solvent=solvent))
    # time.sleep(10)
    # </editor-fold>

    # <editor-fold desc="Additional verification">
    # params_CAMPD = get_params("QH", "Bayesian", considerSolvent=True, saveInitialData=False)
    # params_CAMPD["list_n_sample"] = params_CAMPD["active_list_n_sample"] = [2560]
    # params_CAMPD["ac_mode"] = "MIN"
    # main(params_CAMPD)
    # time.sleep(10)
    #
    # solvent = "C7H7NO3"
    # params_CAPD = get_params("QH", "Bayesian", considerSolvent=False, saveInitialData=True, solvent=solvent)
    # params_CAPD["ac_mode"] = "MIN"
    # main(params_CAPD)

    solvent = "C2H6O2"
    params_CAPD = get_params("TAC", "OneShot", considerSolvent=False, saveInitialData=True, solvent=solvent)
    params_CAPD["bound_pro"] = [(40, 80), (0.2, 2), (3.5, 6), (0.5, 4), (8, 20), (0.4, 4), (3.5, 6)]
    main(params_CAPD)
    # time.sleep(10)
    # </editor-fold>
