import subprocess
import pandas as pd
import re
import numpy as np
import pickle
from datetime import datetime


def pack2size(text_input):
    packer_groups = text_input.split(" ")[::2]
    result = 0
    for s in packer_groups:
        if s[-1] == "M":
            multiplier = 1
        elif s[-1] == "K":
            multiplier = 10 ** -3
        elif s[-1] == "B":
            multiplier = 10 ** -3
        elif s[-1] == "G":
            multiplier = 10 ** 3
        result += (float(s[:-1]) * multiplier)
    return result

def timeparse(text_input):
    time_groups = re.search(
        r"real\s([0-9]+)m([0-9]+.[0-9]+)s\suser\s([0-9]+)m([0-9]+.[0-9]+)s\ssys\s([0-9]+)m([0-9]+.[0-9]+)s", text_input)

    real_t = float(time_groups.group(1)) * 60 + float(time_groups.group(2))
    user_t = float(time_groups.group(3)) * 60 + float(time_groups.group(4))
    sys_t = float(time_groups.group(5)) * 60 + float(time_groups.group(6))

    return real_t, user_t, sys_t

with open("dep_list.txt", "rb") as fp:   # Unpickling
    dep_list = pickle.load(fp)

timings = pd.DataFrame(columns=["library", "real_time_full", "user_time_full", "sys_time_full",
                                "real_time_solo", "user_time_solo", "sys_time_solo",
                                "real_time_constant", "user_time_constant", "sys_time_constant"])
dependencies_sizes = pd.DataFrame(columns=["library", "dependencies"])

## 5190
start = 0
end = 1000
set_num = 1

timer_index = 1
t_list = []
a = datetime.now()

for i in dep_list[start:end]:

    if i == "fmriprep":
        continue

    if set_num == 1:
        rc = subprocess.call("yes | ./benchmarker.sh %s %s" % (i, 1), shell=True)
        dep = open("dependencies.txt", "r").read()
        try:
            if not ("Requires" in dep):
                dep_groups = "ERROR"
            else:
                dep_groups = re.search(r"Requires: (.+) Required-by", dep).group(1).split(", ")
        except:
            dep_groups = []

        pack_before = open("packsize_before.txt", "r").read()
        sum_before = pack2size(pack_before)

        pack_after = open("packsize_after.txt", "r").read()
        sum_after = pack2size(pack_after)


        dependencies_sizes = dependencies_sizes.append({"library": i, "dependencies": dep_groups, "size": sum_after - sum_before, 'size_full': pack_after}, ignore_index=True)


    else:
        rc = subprocess.call("yes | ./benchmarker.sh %s %s" % (i, 0), shell=True)

    tt_real, tt_user, tt_sys = timeparse(open("timetaken.txt", "r").read())
    ts_real, ts_user, ts_sys = timeparse(open("timesolo.txt", "r").read())
    tc_real, tc_user, tc_sys = timeparse(open("timeconstant.txt", "r").read())

    timings = timings.append({"library": i, "real_time_full": tt_real, "user_time_full": tt_user, "sys_time_full": tt_sys,
                              "real_time_solo": ts_real, "user_time_solo": ts_user, "sys_time_solo": ts_sys,
                              "real_time_constant": tc_real, "user_time_constant": tc_user, "sys_time_constant": tc_sys,
                              }, ignore_index=True)

    if timer_index % 50 == 0:
        t_list.append(str(timer_index) + " " + str(datetime.now() - a))
    timer_index += 1

print(timings)
timings.to_csv("ppip_benchmarks_" + str(start) + "_" + str(end - 1) + "_" + str(set_num) + ".csv")

if set_num == 1:
    print(dependencies_sizes)
    dependencies_sizes.to_csv("ppip_dependencies_sizes_" + str(start) + "_" + str(end - 1) + "_" + str(set_num) + ".csv")

print("t_list")
print(t_list)