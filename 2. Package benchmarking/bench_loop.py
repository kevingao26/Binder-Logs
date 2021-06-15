import subprocess

dependencies = ["pandas"]

rc = subprocess.call("./benchmarker.sh '%s'" % dependencies[0], shell=True)

print(rc)
