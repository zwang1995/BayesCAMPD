# Created on 25 Aug 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de

from utility_aspen import *


def Aspen_Monitor(tolerance=3):
    while 1 < 2:
        print("--- Status Monitoring ---")
        try_idx, patience = 0, 0
        while try_idx < tolerance:
            sub_try_idx, sub_patience = 0, 0
            while sub_try_idx < tolerance:
                try:
                    for process in [psutil.Process(pid) for pid in psutil.pids()]:
                        name = process.name()
                        if name == "AspenPlus.exe":
                            cpu_percent = process.cpu_percent(interval=0.5)
                            memory_percent = process.memory_percent()
                            print("CPU% and MEM%:", cpu_percent, "%.2f" % memory_percent)
                            if cpu_percent == 0:
                                sub_patience += 1

                            break
                    sub_try_idx += 1
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(1)
            if sub_patience == tolerance:
                patience += 1
            try_idx += 1
            print(f"#{try_idx}: {patience} {sub_patience}")
            if patience == tolerance:
                kill_aspen()
                # aspen_plus = Aspen_Plus_Interface()
                # aspen_plus.load_bkp(r"../simulation/ExtractiveDistillation_T1T2_TAC.bkp", 0, 1)
            time.sleep(60)


if __name__ == "__main__":
    Aspen_Monitor()
