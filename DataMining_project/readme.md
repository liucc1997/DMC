## 北京理工大学2020年数据挖掘课程项目 二手车交易价格预测项目
## 算法2 随机森林回归器和GBDT模型

### 代码/报告
[algorithm2.ipynb](algorithm2.ipynb)

### 说明

- 输入的train_data_v1数据集80%的数据用于训练，20%用于分析模型
- 使用了回归决策树、随机森林回归器和GBDT模型预测二手车价格，并使用MAE(Mean Absolute Error)对不同模型的表现进行了对比
- 训练集中，根据creatDates和creatDates计算得到used_time
- 作为特征的属性包括:
            ["bodyType","brand","kilometer",
              'model', 'notRepairedDamage', 'power',
              'v_0', 'v_1', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14',
               'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 
              'regDate_year',
              'name_count','used_time']

- 最后使用随机森林回归器和GBDT模型的均值融合预测测试集，结果保存在[output/algo2_predict.csv](https://github.com/liucc1997/DMC/blob/master/DataMining_project/output/submmit_stack.csv)文件中

