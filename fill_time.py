# coding: utf-8

# A simple filling calender scheduler
from ray.lprint import lprint
import os
l = lprint("Running fill calender task")

import argparse
parser = argparse.ArgumentParser(description = 'ip and user mapping graph relation mapping')

parser.add_argument('--start', dest = "start",
                    help = "start date string in format of 'yyyy-mm-dd'")

parser.add_argument('--end', dest = "end",
                    help = "end date string in format of 'yyyy-mm-dd'")

parser.add_argument('--pyfile', dest = "pyfile",
                    help = "path of the python file")

parser.add_argument('--py', dest = "py", default = "/home/dev/anaconda3/bin/python",
                    help = "python command")

parser.add_argument('--datename', dest = "datename", default = "dt",
                    help = "name of the date field, default dt")

args = parser.parse_args()
# args = parser.parse_args("--start=2017-07-01 --end=2017-07-10 --pyfile=py123.py".split())

start = args.start
end = args.end
py = args.py
pyfile = args.pyfile
datename = args.datename

l.p(args)

from datetime import datetime,timedelta

def str2dt(x):
    xlist = list(int(i) for i in x.split("-"))
    return datetime(*xlist)

# start and end date
start_dt = str2dt(start)
end_dt = str2dt(end)

assert end_dt>start_dt, "end should be later than start"

for d in range((end_dt - start_dt).days+1):
    dt_ = (start_dt + timedelta(d)).strftime("%Y-%m-%d")
    l.p("Woking on date",dt_)
    cmd = "%s %s --%s=%s"%(py,pyfile,datename,dt_)
    l.p("running command",cmd)
    
    cmd = "%s |tee .fill_%s.log"%(cmd,dt_)
    os.system(cmd)
    l.p("cmd finished",cmd)

