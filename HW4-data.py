#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

# 读数据
cate_reader = pd.read_csv("D-cate.csv")
cate_columns = cate_reader.columns.values
cate_values = cate_reader.values

comment_reader = pd.read_csv("D-comment.csv")
comment_columns = comment_reader.columns.values
comment_values = comment_reader.values

mentor_reader = pd.read_csv("D-mentor.csv")
mentor_columns = mentor_reader.columns.values
mentor_values = mentor_reader.values

topic_reader = pd.read_csv("D-topic.csv")
topic_columns = topic_reader.columns.values
topic_values = topic_reader.values

# 数据处理

# 对mentor表中的行家接受率arating，设置高接受率、中接受率、其他共三个水平
arating_index = np.where(mentor_columns == 'arating')[0][0]
ac = mentor_values[:, arating_index]
high_ac = (ac == '高接受率').astype(np.int)
medium_ac = (ac == '中接受率').astype(np.int)
nan_ac_index = np.where((high_ac + medium_ac) == 0)[0]
ac[nan_ac_index] = '其他'
mentor_values[:, arating_index] = ac

# 对mentor表中的行家回应邀约时间react_tm，设置半天内回应、1天内回应、2天内回应、其他共四个水平
react_tm_index = np.where(mentor_columns == 'react_tm')[0][0]
react_tm = mentor_values[:, react_tm_index]
half_res = (react_tm == '半天内回应').astype(np.int)
one_res = (react_tm == '1天内回应').astype(np.int)
two_res = (react_tm == '2天内回应').astype(np.int)
nan_res_index = np.where((half_res + one_res + two_res) == 0)[0]
react_tm[nan_res_index] = '其他'
mentor_values[:, react_tm_index] = react_tm

# 对topic表中的话题约见城市topic_city，设置北京、上海、深圳、广州、杭州、成都、武汉、西安、宁波、其他共10个水平
topic_city_index = np.where(topic_columns == 'topic_city')[0][0]
topic_city = topic_values[:, topic_city_index]
city_list = ['北京', '上海', '深圳', '广州', '杭州', '成都', '武汉', '西安', '宁波']
topic_city_list = []
for i in city_list:
    topic_city_list.append((topic_city == i).astype(np.int))
nan_topic_city_index = np.where(sum(topic_city_list) == 0)[0]
topic_city[nan_topic_city_index] = '其他'
topic_values[:, topic_city_index] = topic_city

# 在topic表中增加9个0-1变量：d1-d9，分别对应心理、投资理财、职场发展、教育学习、创业和融资、生活服务、互联网+、兴业经验、其他共9个话题
new_topic = np.array(['心理', '投资理财', '职场发展', '教育学习', '创业和融资', '生活服务', '互联网+', '兴业经验', '其他'])
topic_columns = np.concatenate((topic_columns, new_topic), axis=0)
topic_tag_index = np.where(topic_columns == 'topic_tag')[0][0]
topic_tag = topic_values[:, topic_tag_index]
topic_tag_bool_index = []
for i in new_topic:
    topic_tag_bool_index.append((topic_tag == i).astype(np.int))
topic_values = np.concatenate((topic_values, np.array(topic_tag_bool_index).transpose(1, 0).astype(np.int)), axis=1)

# 在topic中增加0-1变量：is_ord，话题是否成交（topic_ordcnt>0）
topic_columns = np.concatenate((topic_columns, np.array(['is_ord'])), axis=0)
topic_ordcnt_index = np.where(topic_columns == 'topic_ordcnt')[0][0]
topic_ordcnt = topic_values[:, topic_ordcnt_index]
topic_ordcnt_posi = topic_ordcnt > 0
topic_values = np.concatenate((topic_values, topic_ordcnt_posi.reshape(-1, 1).astype(np.int)), axis=1)

# 在topic中增加0-1变量：is_fst，话题是否发生在北上深
topic_columns = np.concatenate((topic_columns, np.array(['is_fst'])), axis=0)
topic_city_index = np.where(topic_columns == 'topic_city')[0][0]
topic_city = topic_values[:, topic_city_index]
topic_city_neg_posi = (
        (topic_city == '北京').astype(np.int) + (topic_city == '上海').astype(np.int) + (topic_city == '深圳').astype(
    np.int)).clip(0, 1)
topic_values = np.concatenate((topic_values, topic_city_neg_posi.reshape(-1, 1).astype(np.int)), axis=1)

# topic中增加话题价格带：price_b，分为<=300、(300,600]、(600,1000)、1000+四个水平
# topic表中增加话题描述长度：len_desc，话题描述字符数，划分为(0,200]、(200,400]、(400,600]、(600,800]、(800,1000]、1000+六个水平
# topic表中增加行家是否回复用户评价：is_rpl，行家是否回复用户评价（聚合comment表中is_reply得到reply_cnt，reply_cnt>0）
topic_columns = np.concatenate((topic_columns, np.array(['price_b'])), axis=0)
topic_columns = np.concatenate((topic_columns, np.array(['len_desc'])), axis=0)
topic_price_index = np.where(topic_columns == 'price')[0][0]
topic_desc_index = np.where(topic_columns == 'topic_desc')[0][0]
topic_price = topic_values[:, topic_price_index]
topic_desc = topic_values[:, topic_desc_index]
price_b = []
len_desc = []
topic_columns = np.concatenate((topic_columns, np.array(['is_rpl'])), axis=0)
comment_reply_index = np.where(comment_columns == 'is_reply')[0][0]
comment_topic_id_index = np.where(comment_columns == 'topic_id')[0][0]
topic_id = comment_values[:, comment_topic_id_index]
comment_reply = comment_values[:, comment_reply_index]
diff_index = np.where(np.diff(topic_id, prepend=topic_id[0]))[0]
uniq_topic_id = np.concatenate((np.array([topic_id[0]]), topic_id[diff_index]), axis=0)
reply_pos = np.array([sum(i) for i in np.split(comment_reply, diff_index)])
topic_id_pos = uniq_topic_id[np.where(reply_pos > 0)[0]]
reply_zeros = np.zeros((len(topic_price), 1))
topic_id_index = np.where(topic_columns == 'topic_id')[0][0]
real_topic_id = topic_values[:, topic_id_index]
for i in range(len(topic_price)):
    if topic_price[i] <= 300:
        price_b.append('(0,300]')
    elif 300 < topic_price[i] <= 600:
        price_b.append('(300,600]')
    elif 600 < topic_price[i] < 1000:
        price_b.append('(600,1000)')
    elif topic_price[i] >= 1000:
        price_b.append('1000+')
    else:
        print('error')
        exit()
    if len(topic_desc[i]) <= 200:
        len_desc.append('(0,200]')
    elif 200 < len(topic_desc[i]) <= 400:
        len_desc.append('(200,400]')
    elif 400 < len(topic_desc[i]) <= 600:
        len_desc.append('(400,600]')
    elif 600 < len(topic_desc[i]) <= 800:
        len_desc.append('(600,800]')
    elif 800 < len(topic_desc[i]) <= 1000:
        len_desc.append('(800,100)')
    elif len(topic_desc[i]) > 1000:
        len_desc.append('1000+')
    else:
        print('error')
        exit()
    if real_topic_id[i] in topic_id_pos:
        reply_zeros[i] = 1
topic_values = np.concatenate((topic_values, np.array(price_b).reshape(-1, 1)), axis=1)
topic_values = np.concatenate((topic_values, np.array(len_desc).reshape(-1, 1)), axis=1)
topic_values = np.concatenate((topic_values, reply_zeros.reshape(-1, 1).astype(np.int)), axis=1)

# 绘制导师约见人数分布直方图（可以考虑对mentor$mentor_ordcnt做对数变换），并进行描述分析
mentor_ordcnt_index = np.where(mentor_columns == 'mentor_ordcnt')[0][0]
mentor_ordcnt = mentor_values[:, mentor_ordcnt_index]
plt.hist(mentor_ordcnt, bins=mentor_ordcnt.max() + 1)
plt.savefig('hist.jpg')


def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


# 分领域（心理、投资理财等共9个领域）绘制行家约见人数（topic_ordcnt）的箱线图，按中位数从左到右，从高到低的顺序排列，并进行描述分析
mentor_id_index = np.where(topic_columns == 'mentor_id')[0][0]
mentor_id = topic_values[:, mentor_id_index]
real_mentor_id_index = np.where(mentor_columns == 'mentor_id')[0][0]
real_mentor_id = mentor_values[:, real_mentor_id_index]
topics = np.array(['心理', '投资理财', '职场发展', '教育学习', '创业和融资', '生活服务', '互联网+', '兴业经验', '其他'])
num_eachtopic = [[], [], [], [], [], [], [], [], []]
for i in range(len(topics)):
    topic_columns_index = np.where(topic_columns == topics[i])[0][0]
    topic_columns_value = topic_values[:, topic_columns_index]
    mentor_id_pos_index = mentor_id[np.where(topic_columns_value == 1)[0]]
    for j in range(len(real_mentor_id)):
        if real_mentor_id[j] in mentor_id_pos_index:
            num_eachtopic[i].append(mentor_ordcnt[j])
median_topic = []
for i in num_eachtopic:
    if i == []:
        median_topic.append(0)
    else:
        median_topic.append(get_median(i))
median_topic = np.array(median_topic)
index = np.argsort(median_topic)[::-1]
topics = topics[index]
num_eachtopic = np.array(num_eachtopic)[index]
plt.clf()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.boxplot(num_eachtopic, labels=topics)
plt.legend()
plt.savefig('field_boxplot')

# 绘制不同邀约接受率（arating）、不同响应时长（react_tm）、是否回复评价（is_rp）和话题约见人数（topic_ordcnt）的箱线图，并进行描述分析
mentor_arating_index = np.where(mentor_columns == 'arating')[0][0]
mentor_arating = mentor_values[:, mentor_arating_index]
plt.clf()
plt.hist(mentor_arating)
plt.legend()
plt.savefig('arating_hist.jpg')

mentor_react_tm_index = np.where(mentor_columns == 'react_tm')[0][0]
mentor_react_tm = mentor_values[:, mentor_react_tm_index]
plt.clf()
plt.hist(mentor_react_tm)
plt.legend()
plt.savefig('react_tm_hist.jpg')

topic_is_rpl_index = np.where(topic_columns == 'is_rpl')[0][0]
topic_is_rpl = topic_values[:, topic_is_rpl_index]
for i in range(len(topic_is_rpl)):
    if topic_is_rpl[i] == 1:
        topic_is_rpl[i] = '已回复'
    else:
        topic_is_rpl[i] = '未回复'
plt.clf()
plt.hist(topic_is_rpl)
plt.legend()
plt.savefig('is_rpl_hist.jpg')

topic_ordcnt_index = np.where(topic_columns == 'topic_ordcnt')[0][0]
topic_ordcnt = topic_values[:, topic_ordcnt_index]
plt.clf()
plt.hist(topic_ordcnt, bins=60)
plt.savefig('ordcnt_hist.jpg')

# 根据comment表中信息，绘制各行业用户数量柱状图并进行描述分析
comment_industry_name_index = np.where(comment_columns == 'user_industry_name')[0][0]
comment_industry_name = comment_values[:, comment_industry_name_index]
comment_industry_name_list = []
num_comment_industry_name = []
for i in comment_industry_name:
    try:
        if np.isnan(i):
            continue
    except:
        pass
    if i not in comment_industry_name_list:
        comment_industry_name_list.append(i)
        num_comment_industry_name.append(1)
    else:
        industry_name_index = np.where(np.array(comment_industry_name_list) == i)[0][0]
        num_comment_industry_name[industry_name_index] += 1
plt.clf()
plt.figure(dpi=300, figsize=(48, 18))
plt.bar(x=range(len(comment_industry_name_list)), tick_label=comment_industry_name_list,
        height=num_comment_industry_name)
plt.legend()
# plt.show()
plt.savefig('industry_bar.jpg')

# 使用is_ord作为因变量Y
# 自变量选择 话题价格、话题价格带、想见行家用户数、行家约见人数、话题持续时长、话题所在城市、行家回复邀约的时长、行家接受约见的比率、话题长度、话题是否在北上广以及所属行业进行建模。
# 使用AIC进行模型选择，绘制ROC曲线并计算AUC
is_ord_index = np.where(topic_columns == 'is_ord')[0][0]
is_ord = topic_values[:, is_ord_index]
price_index = np.where(topic_columns == 'price')[0][0]
price = topic_values[:, price_index]
price = 4 * price / max(price)
price_b_index = np.where(topic_columns == 'price_b')[0][0]
price_b = topic_values[:, price_b_index]
duration_index = np.where(topic_columns == 'duration')[0][0]
duration = topic_values[:, duration_index]
topic_city_index = np.where(topic_columns == 'topic_city')[0][0]
topic_city = topic_values[:, topic_city_index]
len_desc_index = np.where(topic_columns == 'len_desc')[0][0]
len_desc = topic_values[:, len_desc_index]
for i in range(len(price_b)):
    if price_b[i] == '(0,300]':
        price_b[i] = 1
    elif price_b[i] == '(300,600]':
        price_b[i] = 2
    elif price_b[i] == '(600,1000)':
        price_b[i] = 3
    else:
        price_b[i] = 4
    if duration[i] == '1小时':
        duration[i] = 1
    elif duration[i] == '2小时':
        duration[i] = 2
    elif duration[i] == '2小时+':
        duration[i] = 3
    else:
        duration[i] = 0
    index = np.where(np.array(city_list) == topic_city[i])[0][0]
    topic_city[i] = index
    if len_desc[i] == '(0,200]':
        len_desc[i] = 1
    elif len_desc[i] == '(200,400]':
        len_desc[i] = 2
    elif len_desc[i] == '(400,600]':
        len_desc[i] = 3
    elif len_desc[i] == '(600,800]':
        len_desc[i] = 4
    elif len_desc[i] == '(800,1000]':
        len_desc[i] = 5
    else:
        len_desc[i] = 6

is_fst_index = np.where(topic_columns == 'is_fst')[0][0]
is_fst = topic_values[:, is_fst_index]

mentor_id_index = np.where(topic_columns == 'mentor_id')[0][0]
mentor_id = topic_values[:, mentor_id_index]
topic_id_index = np.where(topic_columns == 'topic_id')[0][0]
topic_id = topic_values[:, topic_id_index]
mentor_mentor_id_index = np.where(mentor_columns == 'mentor_id')[0][0]
mentor_mentor_id = mentor_values[:, mentor_mentor_id_index]
mentor_heart_index = np.where(mentor_columns == 'heart')[0][0]
mentor_heart = mentor_values[:, mentor_heart_index]
mentor_heart = 4 * mentor_heart / max(mentor_heart)
mentor_mentor_ordcnt_index = np.where(mentor_columns == 'mentor_ordcnt')[0][0]
mentor_mentor_ordcnt = mentor_values[:, mentor_mentor_ordcnt_index]
mentor_react_tm_index = np.where(mentor_columns == 'react_tm')[0][0]
mentor_react_tm = mentor_values[:, mentor_react_tm_index]
mentor_arating_index = np.where(mentor_columns == 'arating')[0][0]
mentor_arating = mentor_values[:, mentor_arating_index]
comment_topic_id_index = np.where(comment_columns == 'topic_id')[0][0]
comment_topic_id = comment_values[:, comment_topic_id_index]
comment_user_industry_name_index = np.where(comment_columns == 'user_industry_name')[0][0]
comment_user_industry_name = comment_values[:, comment_user_industry_name_index]
topic_heart = np.zeros(len(mentor_id))
topic_ordcnt = np.zeros(len(mentor_id))
topic_react_tm = np.zeros(len(mentor_id))
topic_arating = np.zeros(len(mentor_id))
topic_industry_name = np.zeros(len(mentor_id))

for i in range(len(topic_heart)):
    id_index = np.where(mentor_mentor_id == mentor_id[i])[0][0]
    topic_heart[i] = mentor_heart[id_index]
    topic_ordcnt[i] = mentor_mentor_ordcnt[id_index]
    if mentor_react_tm[id_index] == '半天内回应':
        topic_react_tm[i] = 1
    elif mentor_react_tm[id_index] == '1天内回应':
        topic_react_tm[i] = 2
    elif mentor_react_tm[id_index] == '2天内回应':
        topic_react_tm[i] = 3
    else:
        topic_react_tm[i] = 0
    if mentor_arating[id_index] == '中接受率':
        topic_arating[i] = 1
    elif mentor_arating[id_index] == '高接受率':
        topic_arating[i] = 2
    else:
        topic_arating[i] = 0
    try:
        id_index2 = np.where(comment_topic_id == topic_id[i])[0][0]
        index = np.where(np.array(comment_industry_name_list) == comment_user_industry_name[id_index2])[0][0]
        topic_industry_name[i] = np.log(index + 1)
    except:
        topic_industry_name[i] = '0'

data = pd.DataFrame({'话题价格': price, '话题价格带': price_b, '想见行家用户数': topic_heart, '行家约见人数': topic_ordcnt,
                     '话题持续时长': duration, '行家回复邀约的时长': topic_react_tm, '话题所在城市': topic_city, 'is_ord': is_ord,
                     '行家接受约见的比率': topic_arating, '话题长度': len_desc, '话题是否在北上广': is_fst, '所属行业': topic_industry_name})
print(data.values[0])
import statsmodels.formula.api as smf
import statsmodels.api as sm


def get_auc(xy_fpr_tpr):
    auc = 0.0
    pre_x = 0
    for x, y in xy_fpr_tpr:
        if x != pre_x:
            auc += (x - pre_x) * y
            pre_x = x
    return auc


def draw_roc(xy_fpr_tpr):
    x = [item[0] for item in xy_fpr_tpr]
    y = [item[1] for item in xy_fpr_tpr]
    plt.plot(x, y)
    plt.title('ROC curve (AUC = %.4f)' % get_auc(xy_fpr_tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def sort_roc(samples, pos, neg):
    fp, tp = 0, 0  # 假阳，真阳
    xy_fpr_tpr = []
    sample_sort = sorted(samples, key=lambda x: x[0], reverse=True)
    for i in range(len(sample_sort)):
        fp += sample_sort[i][2]
        tp += sample_sort[i][1]
        xy_fpr_tpr.append([fp / neg, tp / pos])
    return xy_fpr_tpr


def stepwise_select(data, response):
    def forward_method(data, response, selected):
        initial_selected_num = len(selected)
        remaining = set(data.columns)
        remaining.remove(response)
        # 去掉已选定的变量
        for a in selected:
            remaining.remove(a)
        current_score, best_new_score = float('inf'), float('inf')
        initial_selected_num = len(selected)
        while remaining:

            aic_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {}".format(
                    response, ' + '.join(selected + [candidate]))
                aic = smf.glm(
                    formula=formula, data=data,
                    family=sm.families.Binomial(sm.families.links.logit)
                ).fit().aic
                aic_with_candidates.append((aic, candidate))
            aic_with_candidates.sort(reverse=True)
            best_new_score, best_candidate = aic_with_candidates.pop()
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
                print('aic is {},continuing!'.format(current_score))
            else:
                print('forward selection over!')
                break
            print(len(selected) - initial_selected_num)
            if (len(selected) - initial_selected_num) >= 2:
                formula = "{} ~ {} ".format(response, ' + '.join(selected))
                print('formula is {}'.format(formula))
                selected = backward_method(data, response, selected)

        formula = "{} ~ {} ".format(response, ' + '.join(selected))
        print('formula is {}'.format(formula))
        model = smf.glm(
            formula=formula, data=data,
            family=sm.families.Binomial(sm.families.links.logit)
        ).fit()
        return selected

    def backward_method(data, response, selected):
        # selected = list(selected)
        print("-" * 100)
        removed = []
        # 初始化赋值
        best_new_score = float('inf')
        # 全部特征的AIC作为初始参数
        formula = "{} ~ {}".format(
            response, ' + '.join(selected))
        current_score = smf.glm(
            formula=formula, data=data,
            family=sm.families.Binomial(sm.families.links.logit)
        ).fit().aic
        print('initial aic is {}!'.format(current_score))
        print('initial formula is {}'.format(formula))
        while selected:
            aic_with_candidates = []
            for candidate in selected:
                select_tmp = selected.copy()
                select_tmp.remove(candidate)
                formula = "{} ~ {}".format(
                    response, ' + '.join(select_tmp))
                aic = smf.glm(
                    formula=formula, data=data,
                    family=sm.families.Binomial(sm.families.links.logit)
                ).fit().aic
                aic_with_candidates.append((aic, candidate))
            aic_with_candidates.sort(reverse=True)
            best_new_score, best_candidate = aic_with_candidates.pop()
            if current_score > best_new_score:
                selected.remove(best_candidate)
                removed.append(best_candidate)
                current_score = best_new_score
                print('aic is {},continuing!'.format(current_score))
            else:
                print('backward selection over!')
                break

        formula = "{} ~ {} ".format(response, ' + '.join(selected))
        print('final formula is {}'.format(formula))
        return selected

    selected = []
    forward_method(data, response, selected)

selected = stepwise_select(data, 'is_ord')
print(selected)
