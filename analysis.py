import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy import stats
import ptitprince as pt

data = pd.read_csv('results3.csv')
print(data[['understood', 'stuck', 'count']].describe())
count_avg = data['count'].mean()
count_sd = data['count'].std()
## normal t test ##
#assumptions - normal distribution
mu, std = norm.fit(data['count']) 
plt.hist(data['count'], bins=9, density=True, alpha=0.6, color='b')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
  
plt.plot(x, p, 'k', linewidth=2)
plt.show()

print(stats.shapiro(data['count']))

ml_data = data[data['type'] == 'ML']
base_data = data[data['type'] == 'BL']

print(ml_data.describe())
print(base_data.describe())

ml_count_avg = ml_data.groupby(['partecipant']).mean()['count'].to_numpy()
ml_count_sd = ml_data.groupby(['partecipant']).std()['count'].to_numpy()

base_count_avg = ml_data.groupby(['partecipant']).mean()['count'].to_numpy()
base_count_sd = ml_data.groupby(['partecipant']).std()['count'].to_numpy()

print(stats.shapiro(ml_data['count']))
print(stats.shapiro(base_data['count']))

print(stats.ttest_rel(base_data['count'], ml_data['count']))

print(base_count_avg)
print(ml_count_avg)

# satisfaction general #
negative_understood_data = data[(data['understood'] == 'disagree') | (data['understood'] == 'strongly disagree')]
neutral_understood_data = data[(data['understood'] == 'neutral') | (data['understood'] == 'neutral')]
positive_understood_data = data[(data['understood'] == 'agree') | (data['understood'] == 'strongly agree')]

print('Negative understood percentage: ', len(negative_understood_data)*100/ len(data))
print('Neutral understood percentage: ', len(neutral_understood_data)*100/ len(data))
print('Positive understood percentage: ', len(positive_understood_data)*100/ len(data))

not_stuck_data = data[(data['stuck'] == 'disagree') | (data['stuck'] == 'strongly disagree')]
neutral_stuck_data = data[(data['stuck'] == 'neutral') | (data['stuck'] == 'neutral')]
was_stuck_data = data[(data['stuck'] == 'agree') | (data['stuck'] == 'strongly agree')]

print('Not stuck percentage: ', len(not_stuck_data)*100/ len(data))
print('Neutral stuck percentage: ', len(neutral_stuck_data)*100/ len(data))
print('Was stuck percentage: ', len(was_stuck_data)*100/ len(data))

#satisfaction ml#
negative_understood_data = ml_data[(ml_data['understood'] == 'disagree') | (ml_data['understood'] == 'strongly disagree')]
neutral_understood_data = ml_data[(ml_data['understood'] == 'neutral') | (ml_data['understood'] == 'neutral')]
positive_understood_data = ml_data[(ml_data['understood'] == 'agree') | (ml_data['understood'] == 'strongly agree')]

print('Negative understood percentage: ', len(negative_understood_data)*100/ len(ml_data))
print('Neutral understood percentage: ', len(neutral_understood_data)*100/ len(ml_data))
print('Positive understood percentage: ', len(positive_understood_data)*100/ len(ml_data))

not_stuck_data = ml_data[(ml_data['stuck'] == 'disagree') | (ml_data['stuck'] == 'strongly disagree')]
neutral_stuck_data = ml_data[(ml_data['stuck'] == 'neutral') | (ml_data['stuck'] == 'neutral')]
was_stuck_data = ml_data[(ml_data['stuck'] == 'agree') | (ml_data['stuck'] == 'strongly agree')]

print('Not stuck percentage: ', len(not_stuck_data)*100/ len(ml_data))
print('Neutral stuck percentage: ', len(neutral_stuck_data)*100/ len(ml_data))
print('Was stuck percentage: ', len(was_stuck_data)*100/ len(ml_data))

#satisfaction base

negative_understood_data = base_data[(base_data['understood'] == 'disagree') | (base_data['understood'] == 'strongly disagree')]
neutral_understood_data = base_data[(base_data['understood'] == 'neutral') | (base_data['understood'] == 'neutral')]
positive_understood_data = base_data[(base_data['understood'] == 'agree') | (base_data['understood'] == 'strongly agree')]

print('Negative understood percentage: ', len(negative_understood_data)*100/ len(base_data))
print('Neutral understood percentage: ', len(neutral_understood_data)*100/ len(base_data))
print('Positive understood percentage: ', len(positive_understood_data)*100/ len(base_data))

not_stuck_data = base_data[(base_data['stuck'] == 'disagree') | (base_data['stuck'] == 'strongly disagree')]
neutral_stuck_data = base_data[(base_data['stuck'] == 'neutral') | (base_data['stuck'] == 'neutral')]
was_stuck_data = base_data[(base_data['stuck'] == 'agree') | (base_data['stuck'] == 'strongly agree')]

print('Not stuck percentage: ', len(not_stuck_data)*100/ len(base_data))
print('Neutral stuck percentage: ', len(neutral_stuck_data)*100/ len(base_data))
print('Was stuck percentage: ', len(was_stuck_data)*100/ len(base_data))

data2 = pd.read_csv('result4.csv')

f, ax = plt.subplots()
ax = pt.half_violinplot( x = "type", y = "count", data =data2, width = .6)
ax = sns.stripplot( x = "type", y = "count", data = data2, jitter=0.05, alpha=0.4)
ax=sns.boxplot(x="type", y="count",data =data2, width = .15)
plt.xlabel("Classifier type")
plt.ylabel("Number of turns")
plt.title("Number of turns distribution between classifier")
plt.savefig('cloudCount.png', format='png', bbox_inches="tight")
plt.show()



f, ax = plt.subplots()
ax = pt.half_violinplot( x = "type", y = "understood", data =data2, width = .6)
ax = sns.stripplot( x = "type", y = "understood", data = data2, jitter=0.05, alpha=0.4)
ax=sns.boxplot(x="type", y="understood",data =data2, width = .15)
plt.xlabel("Classifier type")
plt.ylabel("Understood score")
plt.title("Understood scores distribution between classifier")
plt.savefig('cloudUnderstood.png', format='png', bbox_inches="tight")
plt.show()



f, ax = plt.subplots()
ax = pt.half_violinplot( x = "type", y = "stuck", data =data2, width = .6)
ax = sns.stripplot( x = "type", y = "stuck", data = data2, jitter=0.05, alpha=0.4)
ax=sns.boxplot(x="type", y="stuck",data = data2, width = .15)
plt.xlabel("Classifier type")
plt.ylabel("Stuck score")
plt.title("Stuck scores distribution between classifier")
plt.savefig('cloudStuck.png', format='png', bbox_inches="tight")
plt.show()

f, ax = plt.subplots()
colors = {'ML':'tab:blue', 'BL':'tab:orange'}
ax = pt.half_violinplot( x = "trial", y = "count",data =data, width = .6 )
ax = sns.stripplot( x = "trial", y = "count",hue = 'type',data = data, jitter=0.1, alpha=0.4)
#ax=sns.boxplot(x="trial", y="count",data = data, width = .15)
plt.xlabel("Trial number")
plt.ylabel("Number of turns")
plt.title("Number of turns distribution between trials")
plt.savefig('cloudTrials.png', format='png', bbox_inches="tight")
plt.show()
