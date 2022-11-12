# -*- coding: utf-8 -*-
import sys
sys.path.append('/data2/machine-learning/')
import os
import glob

# 필요 라이브러리 import
import pandas as pd
import numpy as np
import datetime as dt
import json
import openpyxl
import requests

# 라벨 인코더
from sklearn.preprocessing import LabelEncoder

# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier as rfc

# 정규화 - 스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 모델 평가 척도
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score

# 모델 저장
import joblib

# XGBoost
from xgboost import XGBClassifier, cv

# CatBoost
from catboost import CatBoostClassifier as cbc

# 경고문이 뜨지 않게 함
import warnings
warnings.filterwarnings('ignore')

# 한양대 피처 엔지니어링 부분이라 일부 코드 삭제하였습니다.
def cal_features(df_temp, req_type):
    
    df_temp.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    
    # 이후 데이터에 피처를 추가하고 싶다는 요청으로, 기존 컬럼 저장
    temp_columns = df_temp.columns.tolist()
    
    df_temp['time'] = df_temp['time'].astype('str')
    
    df_temp['time'] = pd.to_datetime(df_temp['time'])
  
    """
    코드 삭제
    """

    df_temp.reset_index(drop=True, inplace=True)
    
    df_temp = df_temp[df_temp['w'] != 0]

    label_encoder = LabelEncoder()
    
    # 학습의 경우
    if req_type == 'L':
        # 작업공종 인코딩
        label_encoder.fit(df_temp['w'])
        df_temp['w'] = label_encoder.transform(df_temp['w'])
        df_temp['w'] = df_temp['w'].astype('int')
        df_temp.dropna(axis=0, inplace=True)
        df_temp.reset_index(drop=True, inplace=True)
        
        # 기존 컬럼과 엔지니어링으로 생성된 컬럼을 합쳐 데이터프레임을 완성
        df_temp = df_temp[list(set(temp_columns + ['time', 'x', 'y', 'h', '이동거리', '속도', '가속도', '각도(-)', '각도(360)', \
       '각변(-)', '각변(360)', '밑변', 'COS_A', '끼인각', '끼변', 'd1', 'd2', 'd2/d1', 'w']))]
       
    # 모델 적용의 
    else:
        # 기존 컬럼과 엔지니어링으로 생성된 컬럼을 합쳐 데이터프레임을 완성
        df_temp = df_temp[list(set(temp_columns + ['time', 'x', 'y', 'h', '이동거리', '속도', '가속도', '각도(-)', '각도(360)', \
       '각변(-)', '각변(360)', '밑변', 'COS_A', '끼인각', '끼변', 'd1', 'd2', 'd2/d1']))]
        
        # 불필요 컬럼 삭제
        df_temp.drop(columns=['w'], axis=1, inplace=True)
    
    df_temp.dropna(axis=0, inplace=True)
    df_temp.reset_index(drop=True, inplace=True)
    
    return df_temp, label_encoder


# 모델 학습에 적합한 데이터 형태로 전처리
def data_for_model(df_temp):
    # 데이터 섞기
    df_model_ = df_temp.sample(frac=1)
    # 불필요 컬럼 제거 - 후에 피처 추가될 경우를 고려하여 이 방식을 고안
    temp_columns = [c for c in df_temp.columns.tolist() if c not in ['time', 'x', 'y', 'h', 'w']]
    df_model_ = df_temp[temp_columns]
    
    # 타겟 데이터
    df_model_t =  df_temp[['w']].copy().values
    
    # 스케일링의 경우
    # df_model = MinMaxScaler().fit_transform(df_model_.copy())
    
    # 스케일링 하지 않은 경우
    df_model = df_model_.copy().values
    
    # 학습/검증/테스트 비율을 정하여, 데이터를 분리
    train_rate = int(len(df_model_t) * 0.7)
    valid_rate = int(len(df_model_t) * 0.2)

    # x : input, y : target
    train_X, train_y = df_model[:train_rate, :], df_model_t[:train_rate]
    val_X, val_y = df_model[train_rate:-valid_rate, :], df_model_t[train_rate:-valid_rate]
    test_X, test_y = df_model[-valid_rate:, :], df_model_t[-valid_rate:]
   
    return train_X, train_y, val_X, val_y, test_X, test_y

# 모델 학습
def model_apply(train_X, train_y, val_X, val_y, test_X, test_y, num_c):
    # 가장 높은 정확도를 보이는 모델을 골라내기 위함
    acc_dict = {}

    # 모델1 XGBoost에 사용되는 파라미터 - 임의의 숫자입니다.
    params = {'learning_rate' : 0.01, 'max_depth': 8, 'n_estimators': 20,
              'num_class' : num_c, 'subsample': 0.7, 'colsample_bytree': 0.6,
              'min_child_weight': 5, 'reg_lambda': 0.6, 'reg_alpha': 0.6,
              'gamma': 0.3, 'num_parallel_tree' : 10, 'booster': 'gbtree',
              'tree_method': 'hist', 'eval_metric' : 'merror',
              'objective': 'multi:softmax', 'random_state' : 1234, 'verbosity': 0}

    # XGBoost
    model1 = XGBClassifier(**params)

    model1.fit(train_X, train_y,
              eval_set=[(train_X, train_y),(val_X, val_y)],
              early_stopping_rounds=200,
              verbose= False)

    pred1 = model1.predict(test_X)
    accuracy1 = accuracy_score(test_y, pred1)
    # 정확도 저장
    acc_dict[accuracy1] = model1

    # Random Forest
    model2 = rfc(random_state=1234)
    model2.fit(train_X, train_y)   
    pred2 = model2.predict(test_X)
    accuracy2 = accuracy_score(test_y, pred2)
    # 정확도 저장
    acc_dict[accuracy2] = model2

    # CatBoost
    model3 = cbc(iterations=200, random_state=100)
    model3.fit(train_X, train_y, eval_set=(val_X, val_y), verbose=False)
    pred3 = model3.predict(test_X)
    accuracy3 = accuracy_score(test_y, pred3)
    # 정확도 저장
    acc_dict[accuracy3] = model3
   
    # 가장 높은 정확도를 보이는 모델을 반환
    return acc_dict[max(acc_dict.keys())]


# 전체 프로세스 실행
def process_start(work_id, file_path, file_name, asset_type, req_type):
    try:
        # 학습의 경우
        if req_type == 'L':
          
            # 디렉토리가 없는 경우 생성
            if os.path.exists(file_path.replace('data/learning', 'models')+'/') == False:
                os.makedirs(file_path.replace('data/learning', 'models')+'/')
            df = pd.DataFrame()
            
            # 입력받은 디렉토리의 모든 엑셀파일 경로 수집
            for f in glob.glob(file_path + '/*.xlsx'):
                temp = pd.ExcelFile(f)
                
                # 파일 통합 후 하나의 데이터프레임으로 변환
                for s in temp.sheet_names:
                    df_temp = temp.parse(sheet_name=s, header=0, index_col=0)
                    df = df.append(df_temp, ignore_index=True)
            
            # 피처 엔지니어링을 통해 라벨인코더와 데이터 프레임 반환
            full_df, en = cal_features(df.copy(), req_type)
            
            # 라벨인코더 저장
            np.save(file_path.replace('data/learning', 'models') + '/' + 'classes.npy', en.classes_)
            num_c = len(en.classes_)
            
            # 학습을 위한 데이터프레임 변환
            train_X, train_y, val_X, val_y, test_X, test_y = data_for_model(full_df.copy())
            
            # 모델에 적용 및 모델 반환
            final_model = model_apply(train_X, train_y, val_X, val_y, test_X, test_y, num_c)
            
            # 모델 저장
            joblib.dump(final_model, file_path.replace('data/learning', 'models') + '/' + 'model.pkl')

        # 모델 적용의 경우
        elif req_type == 'A':
            # 파라미터 디렉토리 보정
            file_path = file_path + '/' if file_path[-1] != '/' else file_path
            main_dir = '/'.join(file_path.replace('data/request', 'models').split('/')[:5]) + '/'
            last = sorted([i for i in os.listdir(main_dir) if i.isdigit() == True])[-1]
            model_dir = main_dir + last
            
            # 모델 적용 대상 파일 불러와서 데이터프레임으로 변환
            df = pd.DataFrame()
            temp = pd.ExcelFile(file_path+file_name)
            
            for s in temp.sheet_names:
                df_temp = temp.parse(sheet_name=s, header=0, index_col=0)
                df = df.append(df_temp, ignore_index=True)

            # 피처 엔지니어링
            full_df, en = cal_features(df.copy(), req_type)
            
            # 인코더 로딩
            en.classes_ = np.load(model_dir + '/' + 'classes.npy', allow_pickle=True)
            
            # 인풋 데이터
            test_X = full_df.copy()[[c for c in full_df.columns.tolist() if c not in ['time', 'x', 'y', 'h', 'w']]].values
            
            # 모델 로딩
            final_model = joblib.load(model_dir + '/' + 'model.pkl')

            # 모델 적용
            pred_X = en.inverse_transform(final_model.predict(test_X))

            # 결과 데이터 생성
            result = pd.concat((full_df[['time', 'x', 'y', 'h']], pd.DataFrame(pred_X)), axis=1)
            
            result.reset_index(inplace=True, drop=False)
            result.columns = ['seq', 'time', 'x', 'y', 'h', 'w']
            # result['seq'] = result['seq'] + 1
            # result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 디렉토리가 없을 경우 생성
            if os.path.exists(file_path.replace('request', 'result')) == False:
                os.makedirs(file_path.replace('request', 'result'))
            
            # 새로운 디렉토리에 대한 리눅스 권한 부여
            os.system("sudo chmod -R 777 /data2/machine-learning/")
            
            # 결과 저장 대상 경로 보정 후 저장
            file_path = file_path.replace('request', 'result')
            file_name = file_name.split('.')[0] + '_' + str(dt.datetime.now().date()).replace('-', '') + '.xlsx'
            result.to_excel(file_path + file_name, index=False)

            # 학습 완료, 결과 파일 저장 후 API 호출
            try:
                headers={'Content-type':'application/json', 'Accept':'application/json'}
                return requests.post('http://0.0.0.0:9201/api/report/ml/analysis/complete', headers=headers, data=json.dumps({"work_id": work_id, "file_name": file_name, "file_path": file_path, "asset_type": asset_type, "result": "ok"}))
            
            # 실패한 경우 에러 
            except (Exception) as e:
                return {'result': str(e)}

    except (Exception) as e:
        return {'result': str(e)}

# API 파라미터 파일 
params = json.loads(open('/data2/machine-learning/codes/parameters.txt', 'r').read().replace("'", '"'))

# 입력된 API 파라미터로부터 필요한 파라미터 추출
work_id = params['work_id']
file_name = params['file_name']
file_path = params['file_path'] + '/' if params['file_path'][-1] != '/' else params['file_path']
asset_type = params['asset_type']
req_type = params['req_type']

# 프로세스 진행
result = process_start(work_id, file_path, file_name, asset_type, req_type)

# 로그 저장
with open('/data2/machine-learning/logs/result.log', 'a') as f:
    f.write(str(dt.datetime.now()) + '\n')
    f.write(str(result) + '\n')
