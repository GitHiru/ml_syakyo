import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()

tips = sns.load_dataset('tips')


# seaborn
sns.violinplot(x='day', y='total_bill', data=tips)

# matplotlib
x1 = tips[tips['day'] == 'Thur']['total_bill'].values.tolist()
x2 = tips[tips['day'] == 'Fri']['total_bill'].values.tolist()
x3 = tips[tips['day'] == 'Sat']['total_bill'].values.tolist()
x4 = tips[tips['day'] == 'Sun']['total_bill'].values.tolist()
fig = plt.figure()
ax = fig.add_subplot()
ax.iolinplot([x1, x2, x3, x4])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Thur', 'Fri', 'Sat', 'Sun']])
ax.set_xlabel('day')
ax.set_ylabel('total_bill')
