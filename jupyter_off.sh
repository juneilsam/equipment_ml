#!/bin/bash

jupyter_pid=`cat /data2/machine-learning/logs/jupyter_id.pid` 

kill "${jupyter_pid}"
