"""
需求：
根据学生在游戏中的行为预测其能否正确回答问题

步骤:

"""


import pandas as pd
import sklearn as sk
import matplotlib.pyplot as ply
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# 1. 获取数据
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
train_labels_data = pd.read_csv("train_labels.csv")
sample_data = pd.read_csv("sample_submission.csv")

# 2. 数据预处理
# 2.1 计算缺失值的百分比
missing_values = train_data.isnull().sum()/train_data.count()
print(missing_values)

# 3. 特征工程
# 将训练数据分为三部分
# 字符型数据
categorical = ['event_name', 'fqid', 'room_fqid', 'text']
# 非字符型数据
numerical = ['elapsed_time', 'level', 'page', 'room_coor_x', 'room_coor_y', 'screen_coor_x',
             'screen_coor_y', 'hover_duration']
# event_name
binaryEvs = ['navigate_click', 'person_click', 'cutscene_click', 'object_click',
             'map_hover', 'notification_click', 'map_click', 'observation_click', 'checkpoint']


# 该函数对每组特性进行迭代，并根据特性名称的后缀应用不同的聚合函数
def processFeatures(train):
    dfs = []
    for group in [(categorical, '_nunique'), (numerical, '_mean'), (numerical, '_std'),
                  (binaryEvs + ['elapsed_time'], '_sum')]:
        for c in group[0]:
            # 获取session和levelgroup中的唯一值的数量
            if group[1] == '_nunique':
                tmp = train.groupby(['session_id', 'level_group'])[c].nunique()
            # 获取列均值
            elif group[1] == "_mean":
                tmp = train.groupby(['session_id', 'level_group'])[c].mean()
            # 获取列标准偏差
            elif group[1] == '_std':
                tmp = train.groupby(['session_id', 'level_group'])[c].std()
            # 获取列总和
            elif group[1] == '_sum':
                train[c] = (train.event_name == c).astype('int8')
                tmp = train.groupby(['session_id', 'level_group'])[c].sum()
            tmp.name = tmp.name + group[1]
            dfs.append(tmp)
    # 数据清洗、合并
    train = train.drop(binaryEvs, axis=1)  # 按列删除
    df = pd.concat(dfs, axis=1)  # 将dfs横向连接
    df = df.fillna(-1)  # 将缺失值填充为-1
    df = df.reset_index()
    df = df.set_index('session_id')
    return df


dataP = processFeatures(train_data)
# 遍历dataP中的所有列，并检查列名是不是“level_group”。如果不是，则将列名添加到列表中。
f = [c for c in dataP.columns if c != 'level_group']
# 从dataP中提取所有唯一的用户索引session_id并将其存储在列表中。去重
u = dataP.index.unique()
# 使用从训练数据中提取的特征和“trainlabels.csv”文件中提供的目标来训练决策树分类器
goal = train_labels_data
goal['session'] = goal.session_id.apply(lambda x: int(x.split('_')[0]))  # 从train_labels中的session_id获得session字段
goal['q'] = goal.session_id.apply(lambda x: int(x.split('_')[-1][1:]))  # 获得q字段
# 分为18个问题，为每个问题训练一个单独的模型，使用具有5个分割的GroupKFold交叉验证方法
gkf = KFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(u),18)),index=u)
models = {}
# 使用5倍的GroupKFold策略进行交叉验证
for i, (train_index, test_index) in enumerate(gkf.split(X=dataP, groups=dataP.index)):
    #
    params = {'max_depth': 4, 'min_samples_leaf': 10, 'random_state': 42}
    # 对于每个问题（1到18），在该问题的训练数据上训练决策树分类器，并将其保存为模型
    for question_num in range(1,19):
        if question_num <= 3:
            question_group = '0-4'
        elif question_num <= 13:
            question_group = '5-12'
        elif question_num <= 22:
            question_group = '13-22'
        # 训练集
        train_data = dataP.iloc[train_index]
        train_data = train_data.loc[train_data.level_group == question_group]
        train_users = train_data.index.values
        train_labels = goal.loc[goal.q == question_num].set_index('session').loc[train_users]
        # 验证集
        valid_data = dataP.iloc[test_index]
        valid_data = valid_data.loc[valid_data.level_group == question_group]
        valid_users = valid_data.index.values
        valid_labels = goal.loc[goal.q == question_num].set_index('session').loc[valid_users]
        # 训练模型
        clf = DecisionTreeClassifier(**params)
        clf.fit(train_data[f].astype('float32'), train_labels['correct'])
        # 保存模型并预测验证集
        models[f'{question_group}_{question_num}'] = clf
        oof.loc[valid_users,question_num-1] = clf.predict_proba(valid_data[f].astype('float32'))[:, 1]
        # 评估模型
        valid_loss = log_loss(valid_labels['correct'], clf.predict_proba(valid_data[f].astype('float32')))
# true labels
true = oof.copy()
k = 0
while k < 18:
    tmp = goal.loc[goal.q == k+1].set_index('session').loc[u]
    true[k] = tmp.correct.values
    k += 1
#
scores = []
thresholds = []
best_score = 0
best_threshold = 0
threshold = 0.4  # 阈值
while threshold <= 0.8:
    preds = (oof.values.reshape((-1)) > threshold).astype('int')
    m = f1_score(true.values.reshape((-1)), preds, average='macro')
    scores.append(m)
    thresholds.append(threshold)
    if m > best_score:
        best_score = m
        best_threshold = threshold
    threshold += 0.01
# 计算F1 score
print('\n''Optimal Results for questions:')
f1_scores = []
for k in range(18):
    # 计算每个问题的F1score
    y_true = true[k].values
    y_pred = (oof[k].values > best_threshold).astype('int')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Q{k}:F1 = {f1:.4f}')
    f1_scores.append(f1)
y_true = true.values.reshape((-1))
y_pred = (oof.values.reshape((-1)) > best_threshold).astype('int')
f1_scoree = f1_score(y_true, y_pred, average='weighted')
print(f'Final F1 score = {f1_scoree:.4f}')
# results = pd.DataFrame({'Question':['Q' + str(k) for k in range(18)],' F1 score':f1_score})
# results.loc[18] = ['\nOverall F1 score is ',f1_scoree]
# results.to_csv('...', index=False)