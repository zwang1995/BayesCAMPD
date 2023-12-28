# Created on 04 Sep 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Parameters and configurations for Computer-Aided Molecular and Process Design

from utility_common import create_directory


def get_params(target, strategy, considerSolvent, saveInitialData=True, solvent=None):
    params = {
        "target": target,
        "strategy": strategy,
        "considerSolvent": considerSolvent,
        "saveInitialData": saveInitialData,
        "random_seed": 42,
        "sampling_method": "latin_hypercube",  # ["random", "latin_hypercube"]
        "normInput": False,
        "normOutput": True,
        "useLogit": True,

        "file_mol": "../data/Bayesian/0_Property_UNIFAC.csv",
        "file_bkp": "../simulation/ExtractiveDistillation_T1T2_TAC.bkp",

        "puri_lb": 0.,  # [0., 0.5]
        "puri_ub": 1.,
        "puri_spec": 0.995,

        "col_mol": ["S", "CP", "HV"],
        "col_pro": ["NStage_T1", "RR_T1", "TopPres_T1", "StoF", "NStage_T2", "RR_T2", "TopPres_T2"],

        "solvent": solvent,
    }

    if considerSolvent:
        params["list_n_sample"] = [128, 256, 384, 512, 640, 768, 896, 1024]
        if strategy == "OneShot":
            params["list_n_sample"] = [128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    else:
        params["list_n_sample"] = [128, 256, 384, 512]
        if strategy == "OneShot":
            params["list_n_sample"] = [128, 256, 384, 512]
    params["active_list_n_sample"] = params["list_n_sample"]

    if strategy == "Bayesian":
        params["n_reconnect"] = 32
        params["n_iter"] = 1000
        params["n_trial"] = 5
        params["ac_mode"] = "EI"
    elif strategy == "OneShot":
        params["n_reconnect"] = 5
        params["n_iter"] = 1
        params["n_trial"] = 20
        params["ac_mode"] = "MIN"

    params["bound_pro"] = [(40, 80), (1, 10), (3.5, 6), (1, 8), (8, 20), (0.2, 2), (3.5, 6)]
    params["bound"] = None
    mol_integrality = [0] * len(params["col_mol"])
    pro_integrality = [1, 0, 0, 0, 1, 0, 0]
    if considerSolvent:
        params["integrality"] = mol_integrality + pro_integrality
    else:
        params["integrality"] = pro_integrality

    str_task = "CAMPD" if considerSolvent else "CAPD"
    str_sol = f"_{solvent}" if solvent is not None else ""
    params["path_out"] = create_directory(f"../model/{strategy}_{str_task}{str_sol}/")
    params["path_data"] = create_directory(f"../model/Bayesian_{str_task}{str_sol}/data/")
    params["path_script"] = params["path_out"] + f"script_{target}{str_sol}"
    params["file_log"] = params["path_out"] + f"history_{target}{str_sol}.log"
    params["file_params"] = params["path_out"] + f"params_{target}{str_sol}.json"

    return params
