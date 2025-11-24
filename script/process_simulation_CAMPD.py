# Created on 08 Jun 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Automated process simulation for the extractive distillation process considering different solvents

from utility_aspen import *


def run_simulation_CAMPD(aspen_plus, solvent_list, combine_loop, txt_file=None, label=None, n_reconnect=32):
    file_bkp = aspen_plus.file_bkp
    FeedFlowrate = 500
    Output_array = []
    TotalTimeCost = 0

    for run_Index, (solvent, Input) in enumerate(zip(solvent_list, combine_loop)):
        if (run_Index != 0) and (run_Index % n_reconnect == 0):
            aspen_plus.close_bkp()
            aspen_plus = AspenPlusInterface(file_bkp)
            aspen_plus.load_bkp()
        print("\t#", solvent, [x for x in Input])

        TimeStart = time.time()
        aspen_plus.re_initialization()
        aspen_plus.Application.Tree.FindNode(r"\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = solvent

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
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\COMP\Input\PRES").Value = TopPres_T2 + 0.5
        aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\FEED_STAGE\FEED2").Value = \
            np.ceil(0.5 * NStage_T2)

        # For TAC calculation only
        aspen_plus.Application.Tree.FindNode(
            r"\Data\Blocks\ED\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T1 - 1
        aspen_plus.Application.Tree.FindNode(
            r"\Data\Blocks\RD\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T2 - 1
        # For TAC calculation only

        # run the process simulation
        aspen_plus.run_simulation()

        # collect results
        try:
            DIST_C4H8_T1 = aspen_plus.Application.Tree.FindNode(
                r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H8").Value
            RebDuty_T1 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\REB_DUTY").Value
            DIST_C4H6_T2 = aspen_plus.Application.Tree.FindNode(
                r"\Data\Streams\DIST2\Output\MOLEFRAC\MIXED\C4H6").Value
            RebDuty_T2 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Output\REB_DUTY").Value
            hasERROR_T1 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\BLKSTAT").Value
            hasERROR_T2 = aspen_plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Output\BLKSTAT").Value
            hasERROR_blo = \
                aspen_plus.Application.Tree.FindNode(r"\Data\Results Summary\Run-Status\Output\UOSSTAT").Value
            hasERROR_cal = \
                aspen_plus.Application.Tree.FindNode(r"\Data\Flowsheeting Options\Calculator\C-1\Output\BLKSTAT").Value
            hasERROR = np.any(hasERROR_blo) or np.any(hasERROR_cal)

            CAPEX = aspen_plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\1").Value
            OPEX = aspen_plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\2").Value
            TAC = aspen_plus.Application.Tree.FindNode(
                r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\3").Value

            Output = [DIST_C4H8_T1, RebDuty_T1, DIST_C4H6_T2, RebDuty_T2, hasERROR, CAPEX, OPEX, TAC]
            print(f"\t\t* simulation result: {Output}"
                  f" | {hasERROR_blo}({hasERROR_T1}&{hasERROR_T2}) & {hasERROR_cal}")
        except Exception as e:
            DIST_C4H8_T1, RebDuty_T1, DIST_C4H6_T2, RebDuty_T2, hasERROR, CAPEX, OPEX, TAC = 0, 0, 0, 0, 1, 0, 0, 0
            hasERROR_T1, hasERROR_T2, hasERROR_blo, hasERROR_cal = None, None, None, None
            Output = [DIST_C4H8_T1, RebDuty_T1, DIST_C4H6_T2, RebDuty_T2, hasERROR, CAPEX, OPEX, TAC]
            print(f"\t\t* simulation error: {e}")
        Output_array.append(Output)

        TimeEnd = time.time()
        TimeCost = TimeEnd - TimeStart
        TotalTimeCost += TimeCost

        if txt_file is not None:
            with open(txt_file, "a") as f:
                f.write(" ".join([label, solvent] +
                                 list_value2str(Input) +
                                 list_value2str(Output) +
                                 list_value2str([hasERROR_T1, hasERROR_T2, hasERROR_blo, hasERROR_cal]) +
                                 list_value2str([TimeCost]) +
                                 ["\n"]))

    Output_array = np.array(Output_array)
    return (Output_array[:, i] for i in range(Output_array.shape[1])), TotalTimeCost
