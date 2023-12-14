# Created on 18 May 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de

from utility_aspen import *
import pandas as pd

# collection of solvents
df = pd.read_csv("../data/0_AspenPlusComponentsList.csv", usecols=["alias"], keep_default_na=False)
solvent_list = df["alias"].to_list()
print(len(solvent_list))
print(solvent_list)

txt_file = "../data/0_Property_UNIFAC.txt"
with open(txt_file, "w") as f:
    f.write(" ".join(["Index", "Solvent", "MW", "RHO", "CP", "HV", "MU", "BP",
                      "GAMMA1", "GAMMA2", "VP", "GAMMAS", "\n"]))

# preparation for Aspen automation
Aspen_Plus = AspenPlusInterface(r"../simulation/0_PropertyExtraction.bkp")
Aspen_Plus.load_bkp()
time.sleep(5)
# Aspen_Plus.Application.Tree.FindNode("\Data\Components\Specifications\Input\ANAME1\SOLUTE").Value = "C4H6-4"

# iteration among solvent list
ERROR_num = 0
for (i, solvent) in enumerate(solvent_list):
    if i >= 315:
        print(f"\nSolvent: {i} - {solvent}")

        try:
            Aspen_Plus.re_initialization()
            if solvent == "C5H9NO-D2":
                Aspen_Plus.close_bkp()
                Aspen_Plus = AspenPlusInterface(r"../simulation/0_PropertyExtraction.bkp")
                Aspen_Plus.load_bkp()
                time.sleep(5)

            else:
                if solvent == "C4H6-4":
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C5H8O2-D1"
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\COM2").Value = "C5H9NO-D2"
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C4H6-4"
                elif solvent == "C4H8-1":
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C5H8O2-D1"
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\COM1").Value = "C5H9NO-D2"
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C4H8-1"
                else:  # if solvent not in ["C4H6-4", "C4H8-1"]:
                    Aspen_Plus.Application.Tree.FindNode(
                        r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = solvent

            Aspen_Plus.run_simulation()
            Aspen_Plus.check_run_completion()

            MW = Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\OUTPUT1\Output\MW").Value  # kg/kmol
            RHO = Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\OUTPUT1\Output\RHOMX_MASS\MIXED").Value  # kg/m^3
            CP = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\9").Value  # kJ/kmol-K (25 C & 1 bar)
            HV = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\10").Value  # kJ/kmol (25 C & 1 bar)
            MU = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\11").Value  # cP (25 C)
            BP = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\12").Value  # C (1 bar)
            GAMMA1 = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\13").Value  # (25 C, 1e-5 / 1e-5 / 1-2e-5)
            GAMMA2 = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\14").Value  # (25 C, 1e-5 / 1e-5 / 1-2e-5)
            VP = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\15").Value  # kPa (25 C & VF 1)
            GAMMAS = Aspen_Plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\16").Value  # (25 C, 1e-5 / 1-1e-5)
            print(f"molecular weight: {MW}, "
                  f"density: {RHO}, "
                  f"heat capacity: {CP}, "
                  f"heat of vaporization: {HV}, "
                  f"viscosity: {MU}, "
                  f"boiling point: {BP}, "
                  f"gamma_C4H8: {GAMMA1}, "
                  f"gamma_C4H6: {GAMMA2}, "
                  f"vapor pressure: {VP}, "
                  f"gamma_sol: {GAMMAS}")

        except Exception as e:
            MW = RHO = CP = HV = MU = BP = GAMMA1 = GAMMA2 = VP = GAMMAS = "ERROR"
            ERROR_num += 1
            print(f"Error: {e}; Error_number = {ERROR_num}")

        if solvent in ["C4H6-4", "C4H8-1"]:
            GAMMA1 = GAMMA2 = GAMMAS = "None"
            Aspen_Plus.close_bkp()
            time.sleep(5)
            Aspen_Plus = AspenPlusInterface(r"../simulation/0_PropertyExtraction.bkp")
            Aspen_Plus.load_bkp()
            time.sleep(5)

        with open(txt_file, "a") as f:
            f.write(" ".join([str(i), solvent, str(MW), str(RHO), str(CP), str(HV), str(MU), str(BP),
                              str(GAMMA1), str(GAMMA2), str(VP), str(GAMMAS), "\n"]))

print("\nTerminate simulation ...")
Aspen_Plus.close_bkp()
