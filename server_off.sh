#!/bin/bash

curl 00.00.0.00:8080/stopServer

server_pid=`cat /data2/machine-learning/logs/tmp.pid` 

kill "${server_pid}"
