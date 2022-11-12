#!/bin/bash

echo -e "\n[[jupyter-server]]"

ps -aux | grep jupyter-server

echo -e "\n[[api-server]]"

ps -aux | grep codes/app.py
