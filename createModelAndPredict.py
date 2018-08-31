
# coding: utf-8

# ### 引入相关包
#     版本说明：
#         python3.6.5 
#         conda 4.5.4  

# In[8]:


import numpy as np
import pandas as pd
# 数据拆分
from sklearn.model_selection import StratifiedKFold
# 评分标准
from sklearn.metrics import f1_score
# 机器学习库
import lightgbm as lgb
# auc评判
from sklearn.metrics import roc_auc_score


# ### 读取特征文件，以及标签文件

# In[2]:


"""
trainDF 读取构造好的训练集特征
testDF  读取构造好的测试集特征
trainLabel 读取标签文件
"""
trainDF = pd.read_csv('./data/train_feature.csv',header=0)
testDF = pd.read_csv('./data/test_feature.csv',header=0)
trainLabel = pd.read_csv('./data/train_labels.csv',header=0)


# In[3]:


# 删除f_id列
trainLabel = trainLabel.drop('f_id',1)


# ### 构造训练集，将特征和标签结合

# In[4]:


"""
train_label_df 训练集特征和标签相结合，数据中，有些标签是没有的，这里使用left Join
features_x 划分训练集特征
features_y 划分测试集标签

"""
train_label_df = pd.merge(trainLabel,trainDF,on='file_name',how='left')
features_x = train_label_df.drop('ret',1).drop('file_name',1).get_values()
features_y = train_label_df['ret'].get_values()


# ### 构造测试集

# In[6]:


"""
test  构造测试集
predID 保存文件ID，提交ID

"""
test = testDF.drop('file_name',1).get_values()
predID = testDF['file_name']


# ### 构造模型，这里使用5折交叉验证

# In[9]:


# N是交叉验证，5 折
N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

# 寻找阈值，不同的阈值，放入不同的list中
xx_cv = []
xx_f1score = []
xx_f1score29 = []
xx_f1score4 = []
xx_f1score35 = []
# 预测结果
xx_pred = []
for train_in,test_in in skf.split(features_x,features_y):
    # 构造训练集和验证集
    X_train,X_test,y_train,y_test = features_x[train_in],features_x[test_in],features_y[train_in],features_y[test_in]
    
    # 创建lightGBM 输入数据，以及验证集
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # lgm输入参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 30,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'verbose': 0,
        'lambda_l2':0.5,
        'lambda_l1':0.2
    }
    params['is_unbalance']='false'
    params['max_bin'] = 100
    params['min_data_in_leaf'] = 200
    print('Start training...')
    # 训练模型，这里使用的是lgm,提升树
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20000,
                    valid_sets=lgb_eval,
                    verbose_eval=500,
                    early_stopping_rounds=50)

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    # 预测测试集，并将验证集的分数存入list
    xx_pred.append(gbm.predict(test, num_iteration=gbm.best_iteration))
    xx_f1score.append(f1_score(y_test, pd.Series(y_pred).map(lambda x: 1 if x>0.3 else 0), average='binary'))
    xx_f1score29.append(f1_score(y_test, pd.Series(y_pred).map(lambda x: 1 if x>0.29 else 0), average='binary'))
    xx_f1score4.append(f1_score(y_test, pd.Series(y_pred).map(lambda x: 1 if x>0.4 else 0), average='binary'))
    xx_f1score35.append(f1_score(y_test, pd.Series(y_pred).map(lambda x: 1 if x>0.35 else 0), average='binary'))
print('xx_cv',np.mean(xx_cv))
print('xx_f1score',np.mean(xx_f1score))
print('xx_f1score29',np.mean(xx_f1score29))
print('xx_f1score4',np.mean(xx_f1score4))
print('xx_f1score35',np.mean(xx_f1score35))


# ### 根据上面选择的阈值分数，选择最优的阈值。0.3。这里使用了5折，相当于构建了5个模型，每个模型都有一个结果，然后将结果进行投票选取最终结果。

# In[10]:


s = 0
for i in xx_pred:
    s += pd.Series(i).map(lambda x : 1 if x>0.3 else 0)
s = s.map(lambda x : 1 if x>3 else 0)


# ### 构造提交结果

# In[11]:


res = pd.DataFrame()
res['id'] = predID
res['ret'] = s


# In[12]:


res.ret.value_counts()


# ### 结果保存为csv文件进行提交

# In[13]:


res.to_csv('submit.csv',header=True,index=False)

