import os
from pathlib import Path
import subprocess


'''base = '/home/brett/PycharmProjects/vgcharm/vgts/runs/cass0221/lp_train_logs/'
entries = Path('/home/brett/PycharmProjects/vgcharm/vgts/runs/cass0221/lp_train_logs/')
out_dir = '/home/brett/PycharmProjects/vgcharm/vgts/runs/cass0221/eval2/lpTrain_csv/'''
base = '/home/brett/PycharmProjects/vgcharm/vgts/runs/checkers_eval_0603/noLP/'
entries = Path('/home/brett/PycharmProjects/vgcharm/vgts/runs/checkers_eval_0603/noLP/')
out_dir = '/home/brett/PycharmProjects/vgcharm/vgts/runs/checkersOUT0603/noLP/'
i = 0


for entry in entries.iterdir():
    # then it is an event

    with os.scandir(base + entry.name) as inner:
        for inn in inner:

            if inn.name[0] == 'e':

                stream = subprocess.Popen(["python", "evaluation/exportTensorFlowLog.py", base + entry.name, out_dir, "scalars,tr" + str(i)],
                                        stdin =subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        bufsize=0)

                i+=1

        # Fetch output
        for line in stream.stdout:
            print(line.strip())


            ##### in order to merge all of these logs together for analysis...use merge-csv.com

