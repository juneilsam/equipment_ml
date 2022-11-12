#!/bin/bash

nohup /usr/bin/python3 /data2/machine-learning/codes/app.py > /data2/machine-learning/logs/server_on.log & echo $! > /data2/machine-learning/logs/tmp.pid
