# Created on 25 Jan 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Functions used for Bayesian optimization

from utility_common import logit
from process_simulation_CAPD import run_simulation_CAPD
from process_simulation_CAMPD import run_simulation_CAMPD

import joblib
import numpy as np
import scipy.stats as st
from smt.sampling_methods import Random, LHS
from smt.applications.mixed_integer import MixedIntegerContext, FLOAT, ORD


def sampling(params, n_sample, solvent_list, sampling_method):
    half = 0.5
    considerSolvent = params["considerSolvent"]
    s_ub = None
    if considerSolvent:
        s_ub = len(solvent_list) - 1
        xtypes = [ORD, ORD, FLOAT, FLOAT, FLOAT, ORD, FLOAT, FLOAT]
        xlimits = [(0, s_ub)] + params["bound_pro"]
    else:
        xtypes = [ORD, FLOAT, FLOAT, FLOAT, ORD, FLOAT, FLOAT]
        xlimits = params["bound_pro"]
    xlimits = [xlimit if xtype == FLOAT else [xlimit[0] - half, xlimit[1] + half] for (xlimit, xtype) in
               zip(xlimits, xtypes)]

    mixint = MixedIntegerContext(xtypes, xlimits)
    samp = None
    if sampling_method == "random":
        samp = mixint.build_sampling_method(Random)
    elif sampling_method == "latin_hypercube":
        samp = mixint.build_sampling_method(LHS, random_state=params["random_seed"])

    xt = samp(n_sample)
    solvent_alias = None
    if considerSolvent:
        x_pro = xt[:, 1:]
        if sampling_method == "random":
            solvent_alias = [solvent_list[i] for i in np.array(xt[:, 0], dtype=int)]
        elif sampling_method == "latin_hypercube":  # solvent selection should be random
            mixint_ = MixedIntegerContext([ORD], [[0 - half, s_ub + half]])
            samp_ = mixint_.build_sampling_method(Random)
            xt_ = samp_(n_sample)
            solvent_alias = [solvent_list[i] for i in np.array(xt_.flatten(), dtype=int)]
    else:
        solvent_alias = [params["solvent"]] * n_sample
        x_pro = xt
    return solvent_alias, x_pro


def valid_sampling(params, aspen_plus_, mol_list, input_, file_txt, label):
    alias = None
    considerSolvent = params["considerSolvent"]
    useLogit = params["useLogit"]
    n_reconnect = params["n_reconnect"]
    puri_lb_, puri_ub_, puri_spec_ = params["puri_lb"], params["puri_ub"], params["puri_spec"]
    if useLogit:
        puri_lb_, puri_ub_, puri_spec_ = logit(puri_lb_), logit(puri_ub_), logit(puri_spec_)

    mol_alias_in, x_pro_in, g1_in, g2_in, duty_in, tac_in = input_
    mol, x_pro = sampling(params, 1, mol_list, sampling_method="random")
    if considerSolvent:
        (g1, y1, g2, y2, run_error, _, _, tac), time_cost = \
            run_simulation_CAMPD(aspen_plus_, mol, x_pro, file_txt, label, n_reconnect)
    else:
        (g1, y1, g2, y2, run_error, _, _, tac), time_cost = \
            run_simulation_CAPD(aspen_plus_, mol, x_pro, file_txt, label)
    if g1 is not None:
        if useLogit:
            g1, g2 = logit(g1), logit(g2)
        if (not run_error) & ((puri_lb_ <= g1) & (g1 <= puri_ub_)) & ((puri_lb_ <= g2) & (g2 <= puri_ub_)):
            if considerSolvent:
                alias = mol[0]
            else:
                alias = params["solvent"]
            mol_alias_in.append(alias)
            x_pro_in = np.vstack((x_pro_in, x_pro))
            g1_in = np.append(g1_in, g1)
            g2_in = np.append(g2_in, g2)
            duty_in = np.append(duty_in, y1 + y2)
            tac_in = np.append(tac_in, tac)
    output = (mol_alias_in, x_pro_in, g1_in, g2_in, duty_in, tac_in)
    return output, time_cost, alias


class BOConvergence:
    def __init__(self, es_patience=10, bo_patience=20):
        self.es_patience, self.bo_patience = es_patience, bo_patience
        self.best = None
        self.counter = 0
        self.stopBO = False
        self.getExtraSample = False
        self.optimum = None

    def update_best(self, best_known):
        if best_known is not None:
            if self.best is None:
                self.best = best_known
            else:
                if best_known < self.best:
                    self.best = best_known
                    self.counter = 0

    def __call__(self, new):
        if self.best is None:
            self.best = new
        if new < self.best:
            self.best = new
            self.counter = 0
            self.getExtraSample = False
        else:
            self.counter += 1
            if self.counter >= self.bo_patience:
                self.stopBO = True
                print(f"\t- terminated as no improvements were observed in {self.bo_patience} iterations")


class BOMonitor:
    def __init__(self):
        self.record = []

    def update(self, keys, values):
        result = {key: value for (key, value) in zip(keys, values)}
        self.record.append(result)

    def save(self, path):
        joblib.dump(self.record, path)


def expected_improvement(mu, sigma, mu_opt, obj="MAX"):
    EI = None
    if obj == "MAX":
        EI = (mu - mu_opt) * st.norm.cdf((mu - mu_opt) / sigma) + sigma * st.norm.pdf((mu - mu_opt) / sigma)
    elif obj == "MIN":
        EI = (mu_opt - mu) * st.norm.cdf((mu_opt - mu) / sigma) + sigma * st.norm.pdf((mu_opt - mu) / sigma)
    return EI


def probability_of_improvement(mu, sigma, mu_opt, obj="MAX"):
    PI = None
    if obj == "MAX":
        PI = st.norm.cdf((mu - mu_opt) / sigma)
    elif obj == "MIN":
        PI = st.norm.cdf((mu_opt - mu) / sigma)
    return PI


def upper_confidence_bound(mu, sigma, lamb=1.0, obj="MAX"):
    UCB = None
    if obj == "MAX":
        UCB = mu + lamb * sigma
    elif obj == "MIN":
        UCB = mu - lamb * sigma
    return UCB


def acquisition_func(x, params, model, y_tr_best):
    ac_mode = params["ac_mode"]
    x = x.reshape(1, -1)

    obj = model.obj
    f, f_std = model.predict(x, return_std=True)

    af_value = None
    if ac_mode == "PI":
        PI = probability_of_improvement(f, f_std, y_tr_best, obj=obj)
        af_value = -PI
    elif ac_mode == "EI":
        EI = expected_improvement(f, f_std, y_tr_best, obj=obj)
        af_value = -EI
    elif ac_mode == "UCB":
        UCB = upper_confidence_bound(f, f_std, 1, obj=obj)
        if model.obj == "MAX":
            af_value = -UCB
        elif model.obj == "MIN":
            af_value = UCB
    elif ac_mode == "MIN":
        af_value = f
    return af_value
