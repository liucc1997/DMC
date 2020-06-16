## 北京理工大学2020年数据挖掘课程项目
## 算法2 回归决策树

### 代码实现
[algorithm2_decision_tree.ipynb](algorithm2_decision_tree.ipynb)

### 输入数据

训练集 [data/train_data_v1.csv](https://github.com/Zening-Li/BIT_DataMining_project/blob/master/process_data/test_data_v1.csv) 

测试集 [data/test_data_v1.csv](https://github.com/Zening-Li/BIT_DataMining_project/blob/master/process_data/train_data_v1.csv)  

### 输出数据

[对test_data_v1.csv的预测结果](https://github.com/liucc1997/DMC/blob/master/DataMining_project/output/algo2_predict.csv)


### 说明

- 使用回归决策树模型
- 训练集中，根据creatDates和creatDates计算得到used_time
- 作为特征的属性包括;
    ["bodyType","brand","fuelType","gearbox","kilometer",
    'model', 'notRepairedDamage', 'power', 'regDate',
    'v_0', 'v_1', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14',
    'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 
    'name_count','used_time']
- 训练集80%的数据用于训练，20%用于评价模型，使用AE和决定系数R^2评价模型
- 最后使用模型预测测试集，结果保存在output/algo2_predict.csv文件中


