import subprocess
import pandas as pd
import re
import pickle

with open("dep_list.txt", "rb") as fp:   # Unpickling
    dep_list = pickle.load(fp)

timings = pd.DataFrame(columns=["library", "real", "user", "sys"])

start = 6
end = 10
set_num = 1
for i in dep_list[start:end]:

    rc = subprocess.call("./benchmarker.sh '%s'" % i, shell=True)

    timetaken = open("timetaken.txt", "r").read()
    time_groups = re.search(r"real\s([0-9]+)m([0-9]+.[0-9]+)s\suser\s([0-9]+)m([0-9]+.[0-9]+)s\ssys\s([0-9]+)m([0-9]+.[0-9]+)s", timetaken)


    real_t = float(time_groups.group(1)) * 60 + float(time_groups.group(2))
    user_t = float(time_groups.group(3)) * 60 + float(time_groups.group(4))
    sys_t = float(time_groups.group(5)) * 60 + float(time_groups.group(6))

    timings = timings.append({"library": i, "real": real_t, "user": user_t, "sys": sys_t}, ignore_index=True)

print(timings)
timings.to_csv("pip_benchmarks_" + str(start) + "_" + str(end - 1) + "_" + str(set_num) + ".csv")