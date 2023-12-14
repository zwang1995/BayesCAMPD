# Created on 08 Jun 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Automated process simulation for the extractive distillation process

from utility_aspen import *


def run_simulation(solvent_list, combine_loop, txt_file=None, label=None):
    FeedFlowrate = 500
    Output_array = []
    for run_Index, (solvent, Input) in enumerate(zip(solvent_list, combine_loop)):
        if run_Index % 10 == 0:
            aspen_plus = AspenPlusInterface()
            aspen_plus.load_bkp(r"../simulation/ExtractiveDistillation_T1T2_recycle_TAC.bkp", 1, 1)
            time.sleep(1)

        print("\t\t#", solvent, [x for x in Input])

        aspen_plus.Application.Tree.FindNode(r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = solvent

        TimeStart = time.time()
        aspen_plus.re_initialization()

        # assign values to each operating variable
        Input = list(Input)
        NStage_T1, RR_T1, TopPres_T1, StoF, NStage_T2, RR_T2, TopPres_T2 = Input
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\NSTAGE").Value = NStage_T1
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\BASIS_RR").Value = RR_T1
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\PRES1").Value = TopPres_T1
        aspen_plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = StoF * FeedFlowrate
        aspen_plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\PRES\MIXED").Value = TopPres_T1 + 0.5
        aspen_plus.Application.Tree.FindNode(r"\Data\Streams\FEED\Input\PRES\MIXED").Value = TopPres_T1 + 0.5
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\FEED_STAGE\FEED").Value = \
            np.ceil(0.5 * NStage_T1)

        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\NSTAGE").Value = NStage_T2
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\BASIS_RR").Value = RR_T2
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\PRES1").Value = TopPres_T2
        # aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\COMP\Input\PRES").Value = TopPres_T2 + 0.5
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\FEED_STAGE\BOTTOM1").Value = \
            np.ceil(0.5 * NStage_T2)

        # # For TAC calculation only
        aspen_plus.Application.Tree.FindNode(
            r"\Data\Blocks\ED\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T1 - 1
        aspen_plus.Application.Tree.FindNode(
            "\Data\Blocks\RD\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T2 - 1
        # # For TAC calculation only

        # run the process simulation
        aspen_plus.run_simulation()
        aspen_plus.check_run_completion()

        # collect results
        # try:
        DIST_C4H8_T1 = aspen_plus.Application.Tree.FindNode(
            r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H8").Value
        RebDuty_T1 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\REB_DUTY").Value
        DIST_C4H6_T2 = aspen_plus.Application.Tree.FindNode(
            r"\Data\Streams\DIST2\Output\MOLEFRAC\MIXED\C4H6").Value
        RebDuty_T2 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Output\REB_DUTY").Value
        hasERROR = aspen_plus.check_convergence()

        CAPEX = aspen_plus.Application.Tree.FindNode(
            r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\1").Value
        OPEX = aspen_plus.Application.Tree.FindNode(
            r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\2").Value
        TAC = aspen_plus.Application.Tree.FindNode(
            r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\3").Value

        Output = [DIST_C4H8_T1, RebDuty_T1, DIST_C4H6_T2, RebDuty_T2, hasERROR, CAPEX, OPEX, TAC]
        print(f"\t\t\t* T1T2 simulation result: {Output}")
        Output_array.append(Output)

        TimeEnd = time.time()
        TimeCost = TimeEnd - TimeStart

        if txt_file is not None:
            with open(txt_file, "a") as f:
                f.write(" ".join([label, solvent] +
                                 list_value2str(Input) +
                                 list_value2str(Output) +
                                 list_value2str([TimeCost]) +
                                 ["\n"]))
        # except:
        #     pass

        if run_Index % 10 == 9:
            aspen_plus.close_bkp()
    if run_Index % 10 != 9:
        aspen_plus.close_bkp()
    if len(Output_array) == 1:
        return DIST_C4H8_T1, RebDuty_T1, DIST_C4H6_T2, RebDuty_T2, hasERROR, CAPEX, OPEX, TAC
    elif len(Output_array) == 0:
        return None, None, None, None, None, None, None, None
    else:
        Output_array = np.array(Output_array)
        return [Output_array[:, i] for i in range(Output_array.shape[1])]


def main():
    solvent_list = ["C5H9NO-D2"]
    combine_loop = [
        [80, 5.829299968910926, 3.500699446263485, 2.5178639696102234, 11, 0.4709906486367485, 3.500388025764173]]

    solvent_list = ["C8H6O4-D3"]
    combine_loop = [
        [73.0, 1.9057373067118, 4.099685281889862, 1.9938039283269573, 12.0, 1.0106317620297427, 5.372065842408266]]

    total_run = len(combine_loop)

    solvent_Index, run_Index = 0, 0
    while solvent_Index < len(solvent_list):
        while run_Index < total_run:
            _ = run_Simulation_T1T2(solvent_list, combine_loop)

            kill_aspen()
            time.sleep(10)

            if solvent_Index == len(solvent_list) - 1 and run_Index == total_run - 1:
                break
        if solvent_Index == len(solvent_list) - 1 and run_Index == total_run - 1:
            break


if __name__ == "__main__":
    main()
