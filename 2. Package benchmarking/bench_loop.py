import subprocess
import pandas as pd
import re
import numpy as np
import pickle
from datetime import datetime

with open("dep_list.txt", "rb") as fp:   # Unpickling
    dep_list = pickle.load(fp)

timings = pd.DataFrame(columns=["library", "real", "user", "sys"])
dependencies_sizes = pd.DataFrame(columns=["library", "dependencies"])

start = 0
end = 4
set_num = 1

timer_index = 1
t_list = []
a = datetime.now()

for i in dep_list[start:end]:

    if set_num == 1:
        rc = subprocess.call("./benchmarker.sh %s %s" % (i, 1), shell=True)
        dep = open("dependencies.txt", "r").read()
        try:
            dep_groups = re.search(r"Requires: (.+) Required-by", dep).group(1).split(", ")
        except:
            dep_groups = []

        pack = open("packsize.txt", "r").read()
        try:
            packer_groups = pack.split(i)
            assert len(packer_groups) > 1
            package_size = packer_groups[0].split(" ")[-2]
        except:
            package_size = np.nan

        dependencies_sizes = dependencies_sizes.append({"library": i, "dependencies": dep_groups, "size": package_size}, ignore_index=True)


    else:
        rc = subprocess.call("./benchmarker.sh %s %s" % (i, 0), shell=True)

    timetaken = open("timetaken.txt", "r").read()
    time_groups = re.search(r"real\s([0-9]+)m([0-9]+.[0-9]+)s\suser\s([0-9]+)m([0-9]+.[0-9]+)s\ssys\s([0-9]+)m([0-9]+.[0-9]+)s", timetaken)


    real_t = float(time_groups.group(1)) * 60 + float(time_groups.group(2))
    user_t = float(time_groups.group(3)) * 60 + float(time_groups.group(4))
    sys_t = float(time_groups.group(5)) * 60 + float(time_groups.group(6))

    timings = timings.append({"library": i, "real": real_t, "user": user_t, "sys": sys_t}, ignore_index=True)

    if timer_index % 50 == 0:
        t_list.append(str(timer_index) + " " + str(datetime.now() - a))
    timer_index += 1

print(timings)
timings.to_csv("pip_benchmarks_" + str(start) + "_" + str(end - 1) + "_" + str(set_num) + ".csv")

if set_num == 1:
    print(dependencies_sizes)
    dependencies_sizes.to_csv("pip_dependencies_sizes_" + str(start) + "_" + str(end - 1) + "_" + str(set_num) + ".csv")

print("t_list")
print(t_list)