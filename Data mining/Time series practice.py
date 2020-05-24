# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:54:39 2019

@author: USER
"""

import pandas as pd
import numpy as np

train_temp = [] 
test_temp = []

#共有18種汙染物
for i in range(18):
    train_temp.append([])
    test_temp.append([])

"""
資料前處理
"""    
temp = pd.read_excel('107年新竹站_20190315.xls')
"""
以下是將特殊符號或是空值用0替代，但本次作業用前後兩筆資料取平均代替
"""
#temp.replace(['[-+]?([0-9]*[\.]?[0-9]+[#*x])'] ,0 , regex=True , inplace=True)
#temp = temp.fillna(0) 
temp = temp.replace("NR",0) 
#NR表示無降雨，以0替代

"""
時間序列shift用
"""        
def shift(l, n):
    return l[n:]
                
#train = temp[temp["日期"].between('2018/10/01','2018/11/30')]                
#test = temp[temp["日期"] >= '2018/12/01']
    
"""
train data 2個月，2018/10/01到2018/11/30
test data 1個月，2018/12/01-2018/12/31
將train及test data中的汙染物數值用時序連接在一起
例如：一列即為10/1-11/30的所有資料
"""
for i in range(len(temp)):
    if temp.iloc[i][0] >= '2018/10/01' and temp.iloc[i][0] <= '2018/11/30':
        #第3-27欄位是汙染物值
        for j in range(3,27):
            train_temp[i%18].append((temp.iloc[i][j]))
    elif temp.iloc[i][0] >= '2018/12/01':
        for j in range(3,27):
            test_temp[i%18].append((temp.iloc[i][j]))

           
"""
train data，只拿PM2.5值
"""

#第9個欄位是PM2.5
#len(train_PM25_temp) 
train_PM25_temp =  train_temp[9]
for i in range(len(train_PM25_temp)):
    #處理異常值及空值，將其轉為前後項的平均
    if str(train_PM25_temp[i]).find("*") != -1 or str(train_PM25_temp[i]).find("#") != -1 or np.isnan(train_PM25_temp[i]) == 1:
        #假如後一項是異常值或是空值
        if str(train_PM25_temp[i+1]).find("*") != -1 or str(train_PM25_temp[i+1]).find("#") != -1 or np.isnan(train_PM25_temp[i+1]) == 1 :
              #假設後面很多項都是異常值或是空值，找到非異常值為止
              for index in range(i+1,len(train_PM25_temp)):
                if str(train_PM25_temp[index]).find("*") == -1 and str(train_PM25_temp[index]).find("#") == -1 and np.isnan(train_PM25_temp[index]) == 0 :
                    train_PM25_temp[i]  = (train_PM25_temp[i-1] + train_PM25_temp[index])/2
                    break
        else:
        #如果後一項非異常值，直接做平均處理
            train_PM25_temp[i]  = (train_PM25_temp[i-1] + train_PM25_temp[i+1])/2

train_X = []
train_y = []
period = 6
#用六小時的資料去預測第七小時的PM2.5
length = len(train_PM25_temp) - period
for i in range(length):
    if len(train_PM25_temp) > period:
        train_X.append(train_PM25_temp[0:period])
        train_y.append(train_PM25_temp[6])
        train_PM25_temp = shift(train_PM25_temp,1)
        #用shift把資料往前補，方便固定前0-5欄位都是x,第6欄是y
"""
#test data，只拿PM2.5值
"""
#第9個欄位是PM2.5
test_PM25_temp =  test_temp[9]
for i in range(len(test_PM25_temp)):
    if str(test_PM25_temp[i]).find("*") != -1 or str(test_PM25_temp[i]).find("#") != -1 or np.isnan(test_PM25_temp[i]) == 1:
        if str(test_PM25_temp[i+1]).find("*") != -1 or str(test_PM25_temp[i+1]).find("#") != -1 or np.isnan(test_PM25_temp[i+1]) == 1 :
              for index in range(i+1,len(test_PM25_temp)):
                if str(test_PM25_temp[index]).find("*") == -1 and str(test_PM25_temp[index]).find("#") == -1 and np.isnan(test_PM25_temp[index]) == 0 :
                    test_PM25_temp[i]  = (test_PM25_temp[i-1] + test_PM25_temp[index])/2
                    break
        else:
            test_PM25_temp[i]  = (test_PM25_temp[i-1] + test_PM25_temp[i+1])/2
test_X = []
test_y = []
period = 6
length = len(test_PM25_temp) - period
for i in range(length):
    if len(test_PM25_temp) > period:
        test_X.append(test_PM25_temp[0:period])
        test_y.append(test_PM25_temp[6])
        test_PM25_temp = shift(test_PM25_temp,1)

