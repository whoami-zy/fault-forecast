
# coding: utf-8

# ### 引入相关包
#     版本说明：
#         python3.6.5 
#         conda 4.5.4  

# In[1]:


import zipfile
import pandas as pd


# ### 路径说明  windows
#     引入文件路径
#         当前文件夹下面的data文件夹存放了训练集和测试集以及标注文件    

# In[2]:


"""
trainZip   使用zipfile,将训练集读入
testZip    使用zipfile,将测试集读入
columnsNames  文件列名
"""
trainZip = zipfile.ZipFile('./data/train.zip')
testZip = zipfile.ZipFile('./data/test.zip')
columnsNames = ['轮毂转速', '轮毂角度', '叶片1角度', '叶片2角度', '叶片3角度', '变桨电机1电流', '变桨电机2电流',
       '变桨电机3电流', '超速传感器转速检测值', '5秒偏航对风平均值', 'x方向振动值', 'y方向振动值', '液压制动压力',
       '机舱气象站风速', '风向绝对值', '大气压力', '无功功率控制状态', '变频器电网侧电流', '变频器电网侧电压',
       '变频器电网侧有功功率', '变频器电网侧无功功率', '变频器发电机侧功率', '发电机运行频率', '发电机电流', '发电机转矩',
       '变频器入口温度', '变频器出口温度', '变频器入口压力', '变频器出口压力', '发电机功率限幅值', '无功功率设定值',
       '额定的轮毂转速', '测风塔环境温度', '发电机定子温度1', '发电机定子温度2', '发电机定子温度3', '发电机定子温度4',
       '发电机定子温度5', '发电机定子温度6', '发电机空气温度1', '发电机空气温度2', '主轴承温度1', '主轴承温度2',
       '轮毂温度', '轮毂控制柜温度', '机舱温度', '机舱控制柜温度', '变频器INU温度', '变频器ISU温度',
       '变频器INU RMIO温度', '变桨电机1功率估算', '变桨电机2功率估算', '变桨电机3功率估算', '风机当前状态值',
       '轮毂当前状态值', '偏航状态值', '偏航要求值', '叶片1电池箱温度', '叶片2电池箱温度', '叶片3电池箱温度',
       '叶片1变桨电机温度', '叶片2变桨电机温度', '叶片3变桨电机温度', '叶片1变频器箱温度', '叶片2变频器箱温度',
       '叶片3变频器箱温度', '叶片1超级电容电压', '叶片2超级电容电压', '叶片3超级电容电压', '驱动1晶闸管温度',
       '驱动2晶闸管温度', '驱动3晶闸管温度', '驱动1输出扭矩', '驱动2输出扭矩', '驱动3输出扭矩']


# In[3]:


"""
获取压缩包里面的文件路径

"""
trainfileNameList = trainZip.namelist()
testfileNameList = testZip.namelist()


# In[4]:


"""
定义两个list  分别存放解析后的train和test的feature
"""
trainDFList = []
testDFList = []


# ### 构造训练集特征

# In[5]:


"""
循环遍历zip包里面的文件路径，解析到csv文件，然后读取里面的内容，构建相关特征
"""
for fname in trainfileNameList:
    if fname.endswith('.csv'):
        dict_f = {}
        dict_f['file_name'] = fname.split('/')[-1]
        f_info = trainZip.getinfo(fname)
        fo = trainZip.open(f_info)
        fDF = pd.read_csv(fo)
        dfmean=fDF.mean()
        dfmin = fDF.min()
        dfmax = fDF.max()
        dfstd = fDF.std()
        fDF_diff = fDF.diff(1).drop(0,0)
        df_diffmean= fDF_diff.mean()
        df_diffmin = fDF_diff.min()
        df_diffmax = fDF_diff.max()
        df_diffstd = fDF_diff.std()
        for col in columnsNames:
            dict_f[col+'_mean'] = dfmean[col]
            dict_f[col+'_diff_mean'] = df_diffmean[col]
            
            dict_f[col+'_min'] = dfmin[col]
            dict_f[col+'_diff_min'] = df_diffmin[col]
            
            dict_f[col+'_max'] = dfmax[col]
            dict_f[col+'_diff_max'] = df_diffmax[col]
            
            dict_f[col+'_std'] = dfstd[col]
            dict_f[col+'_diff_std'] = df_diffstd[col]
        trainDFList.append(dict_f)
    else:
        continue


# ### 构造测试集特征

# In[6]:


for fname in testfileNameList:
    if fname.endswith('.csv'):
        dict_f = {}
        dict_f['file_name'] = fname.split('/')[-1]
        f_info = testZip.getinfo(fname)
        fo = testZip.open(f_info)
        fDF = pd.read_csv(fo)
        dfmean=fDF.mean()
        dfmin = fDF.min()
        dfmax = fDF.max()
        dfstd = fDF.std()
        fDF_diff = fDF.diff(1).drop(0,0)
        df_diffmean= fDF_diff.mean()
        df_diffmin = fDF_diff.min()
        df_diffmax = fDF_diff.max()
        df_diffstd = fDF_diff.std()
        for col in columnsNames:
            dict_f[col+'_mean'] = dfmean[col]
            dict_f[col+'_diff_mean'] = df_diffmean[col]
            
            dict_f[col+'_min'] = dfmin[col]
            dict_f[col+'_diff_min'] = df_diffmin[col]
            
            dict_f[col+'_max'] = dfmax[col]
            dict_f[col+'_diff_max'] = df_diffmax[col]
            
            dict_f[col+'_std'] = dfstd[col]
            dict_f[col+'_diff_std'] = df_diffstd[col]
        testDFList.append(dict_f)
    else:
        continue


# In[7]:


"""
将list转换为DataFrame
"""
testDF = pd.DataFrame(testDFList)
trainDF = pd.DataFrame(trainDFList)


# ### 保存特征集

# In[8]:


trainDF.to_csv('./data/train_feature.csv',header=True,index=False)
testDF.to_csv('./data/test_feature.csv',header=True,index=False)

