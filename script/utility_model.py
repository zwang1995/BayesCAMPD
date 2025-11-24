# Created on 17 Sep 2023 by Zihao Wang
# Functions used for modeling

import time
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.metrics import r2_score

import torch
from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

seed = 42
torch.set_default_dtype(torch.double)


class GPRegressorSklearn:
    def __init__(self, params, target, obj=None, kernel=None):
        self.x_mean = self.x_std = self.y_mean = self.y_std = None
        self.target = target
        self.obj = obj
        self.normInput = params["normInput"]
        self.normOutput = params["normOutput"]
        self.n_x = None
        if kernel is None:
            kernel = gp.kernels.RBF()
        else:
            kernel = kernel
        self.model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10, random_state=seed)

    def fit(self, x, y):
        time_fit_start = time.time()
        self.n_x = x.shape[-1]
        self.x_mean, self.x_std = x.mean(0), x.std(0)
        if self.normInput:
            x_tr = (x - self.x_mean) / self.x_std
        else:
            x_tr = x
        if self.normOutput:
            self.y_mean, self.y_std = y.mean(), y.std()
            y_tr = (y - self.y_mean) / self.y_std
        else:
            y_tr = y
        self.model.fit(x_tr, y_tr)
        time_fit_end = time.time()
        time_fit = time_fit_end - time_fit_start

        yhat, yhat_std = self.predict(x, return_std=True)
        r2 = r2_score(y, yhat)

        return r2, yhat, time_fit

    def predict(self, x, return_std=False):
        x = x.reshape(-1, self.n_x)
        if self.normInput:
            x = (x - self.x_mean) / self.x_std

        f, f_std = self.model.predict(x, return_std=True)
        if self.normOutput:
            f = f * self.y_std + self.y_mean
            f_std = f_std * self.y_std
        if return_std:
            return f, f_std
        else:
            return f


class GPRegressorTorch:
    def __init__(self, params, target, negative=False):
        self.x_mean = self.x_std = self.y_mean = self.y_std = None
        self.target = target
        self.negative = negative
        self.normInput = params["normInput"]
        self.normOutput = params["normOutput"]
        self.n_x = None
        self.model = None

    def fit(self, x, y):
        time_fit_start = time.time()
        self.n_x = x.shape[-1]
        x_tr, y_tr = torch.Tensor(x), torch.Tensor(y.reshape(-1, 1))
        self.x_mean, self.x_std = x.mean(0), x.std(0)
        if self.negative:
            y_tr = -y_tr  # for minimization purpose
        else:
            y_tr = y_tr
        if self.normInput:
            x_tr = (x_tr - self.x_mean) / self.x_std
        else:
            x_tr = x_tr
        if self.normOutput:
            self.y_mean, self.y_std = y_tr.mean(0), y_tr.std(0)
            y_tr = (y_tr - self.y_mean) / self.y_std
        else:
            y_tr = y_tr
        self.model = MixedSingleTaskGP(x_tr, y_tr, cat_dims=[0, 4])
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        time_fit_end = time.time()
        time_fit = time_fit_end - time_fit_start

        yhat, yhat_std = self.predict(x_tr, return_std=True)
        r2 = r2_score(y, yhat)

        return r2, yhat, time_fit

    def predict(self, x, return_std=False):
        x = x.reshape(-1, self.n_x)
        if self.normInput:
            x = (x - self.x_mean) / self.x_std

        posterior = self.model.posterior(torch.Tensor(x))
        mean = posterior.mean.squeeze(-1)  # .squeeze(-1)  # removing redundant dimensions
        if self.normOutput:
            mean = mean * self.y_std + self.y_mean
        if self.negative:
            mean = -mean
        if not return_std:
            return mean.detach().numpy()
        else:
            sigma = posterior.variance.sqrt().view(mean.shape)
            if self.normOutput:
                sigma = sigma * self.y_std
            return mean.detach().numpy(), sigma.detach().numpy()


def get_mol(params):
    col_text = ["Solvent", "cname", "CAS Number", "Compound Class", "Alternate Name"]
    col_mol = params["col_mol"]
    path_data = params["path_data"]
    if params["considerSolvent"]:
        file_mol = params["file_mol"]
        df_mol = pd.read_csv(file_mol)
        df_mol = df_mol.dropna()
        for mol_pro in col_mol:
            df_mol = df_mol[df_mol[mol_pro] != "None"]
            df_mol = df_mol[df_mol[mol_pro] != "ERROR"]
            df_mol = df_mol.reset_index(drop=True)
        for col in df_mol.columns:
            if col not in col_text:
                df_mol[col] = df_mol[col].astype(float)
        mol_list = df_mol["Solvent"].values
        mol_property = df_mol[col_mol].values
        mol_property_dict = {mol: prop for mol, prop in zip(mol_list, mol_property)}
        df_mol.to_csv(path_data + "solvent_list.csv")
        lb, ub = mol_property.min(0), mol_property.max(0)
        params["bound"] = [(lb_, ub_) for (lb_, ub_) in zip(lb, ub)] + params["bound_pro"]
    else:
        params["bound"] = params["bound_pro"]
        mol_list, mol_property, mol_property_dict = None, None, None
    return params, mol_list, mol_property, mol_property_dict
