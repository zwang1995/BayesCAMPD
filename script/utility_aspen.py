# Created on 12 May 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Functions used for process simulation in Aspen Plus

import os
import signal
import time
import psutil
import numpy as np
from win32com.client import GetObject, Dispatch


class AspenPlusInterface:
    def __init__(self, file_bkp):
        self.Application = Dispatch("Apwn.Document.38.0")  # {"V10.0": "36.0", V11.0": "37.0", V12.0": "38.0"}
        self.file_bkp = file_bkp
        print(f"- Connection: {self.Application}")

    def load_bkp(self, visible_state=0, dialog_state=1):
        time.sleep(0.5)
        self.Application.InitFromArchive2(os.path.abspath(self.file_bkp))
        self.Application.Visible = visible_state
        self.Application.SuppressDialogs = dialog_state
        time.sleep(0.5)

    def re_initialization(self):
        self.Application.Reinit()

    def run_simulation(self):
        self.Application.Engine.Run()

    def check_run_completion(self, time_limit=60):
        times = 0
        while self.Application.Engine.IsRunning == 1:
            time.sleep(1)
            times += 1
            print(times)
            if times >= time_limit:
                print("Violate time limitation")
                self.Application.Engine.Stop()
                break

    def check_convergence(self):
        runID = self.Application.Tree.FindNode(r"\Data\Results Summary\Run-Status\Output\RUNID").Value
        his_file = "../simulation/" + runID + ".his"
        with open(his_file, "r") as f:
            hasERROR = np.any(np.array([line.find("ERROR") for line in f.readlines()]) >= 0)
        return hasERROR

    def close_bkp(self):
        try:
            self.Application.Quit()
            print(f"- Connection terminated")
        except Exception as e:
            print(f"Warning: {e}")
        time.sleep(0.5)

    def collect_stream(self):
        streams = []
        node = self.Application.Tree.FindNode(r"\Data\Streams")
        for item in node.Elements:
            streams.append(item.Name)
        return tuple(streams)

    def collect_block(self):
        blocks = []
        node = self.Application.Tree.FindNode(r"\Data\Blocks")
        for item in node.Elements:
            blocks.append(item.Name)
        return tuple(blocks)


def kill_aspen():
    WMI = GetObject("winmgmts:")
    for p in WMI.ExecQuery("select * from Win32_Process where Name='AspenPlus.exe'"):
        os.system("taskkill /pid " + str(p.ProcessId))


def kill_aspen_hard():
    for process in [psutil.Process(pid) for pid in psutil.pids()]:
        name = process.name()
        if name == "AspenPlus.exe":
            os.kill(process.pid, signal.SIGILL)


def spaced_value_with_endpoint(start, stop, step=1):
    return np.arange(start, stop + step, step)


def list_value2str(alist):
    return list(map(str, alist))
