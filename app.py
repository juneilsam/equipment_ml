#!/usr/bin/python
from flask import Flask, jsonify, request, redirect, url_for
import logging
import json

import os
import subprocess
import sys
import signal

import pandas as pd
import numpy as np

# 플라스크 app 선언
app = Flask(__name__)

# 로그 저장을 위한 파라미터 설정
logging.basicConfig(filename='/data2/machine-learning/logs/server_log.log', level=logging.ERROR, format='%(asctime)s %(message)s')

# 학습 API 요청 응답
@app.route('/analysis', methods=['POST'])
def analysis_req():
    # 반환받은 파라미터를 출력
    params = request.get_json(force=True)
    logging.error('is_not_error_insert_params', str(params)+'\n')

    # 요청받은 파일명과 경로를 객체로 선언
    file_name = params['file_name']
    file_path = params['file_path'] + '/' if params['file_path'][-1] != '/' else params['file_path']
    
    # 정상의 경우
    try:
        # 정상 경로 확인
        open(file_path + file_name, 'r')
        
        # 텍스트 파일로 저장 - 백단에서 모델을 돌리기 위함
        with open('/data2/machine-learning/codes/parameters.txt', 'w') as file:
            logging.error('is_not_error_insert_params', str(params)+'\n')
            file.write(str(params))
        
        # 백단에서 모델 파이썬 파일 실행
        subprocess.Popen(['python3', '/data2/machine-learning/codes/ml_code.py'], start_new_session=True)

        # 정상 응답 반환
        return jsonify({'request': 'OK'})

    # 비정상의 경우 에러 반환
    except (Exception, FileNotFoundError) as e:
        logging.error(str(e) + '\n')
        return jsonify({'request': str(e)})

# 서버 정지
@app.route('/stopServer', methods=['GET'])
def stopServer():
    os.kill(os.getppid(), signal.SIGINT)
    return jsonify({"request": "Server is shutting down"})

# 주소 설정
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')
