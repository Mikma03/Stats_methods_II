import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use("fivethirtyeight")

df = pd.read_csv(r'C:\Users\mikol\Desktop\Python_Met_II/Fish.csv')
data = df.copy()
data.head()
data["Species"].value_counts(normalize=True)
data = data.loc[data["Species"] == "Roach"]["Length1"]
pd.DataFrame({'values': data.describe()}).reset_index()

print(f'Mean: {data.mean():.2f}')
conf_i = np.percentile(data, [2.5, 97.5])
print(f'Confidence intervals: {conf_i}')

sns.set()
plt.hist(data, bins=15)
plt.xlabel("Lenght1")
plt.ylabel("Number of Roach fish")
plt.savefig("hist1.png")
plt.show()

x = np.sort(data)
n = len(data)
y = np.arange(1, n+1)/n
plt.plot(x,y, marker=".", linestyle="none")
plt.margins(0.02)
plt.show()

ax = sns.distplot(data,bins=10,kde=True,color='skyblue')
ax.axvline(conf_i[0])
ax.axvline(conf_i[1])
ax.text(11,0.12, conf_i[0])
ax.text(28,0.12, round(conf_i[1],2))
ax.set(xlabel='Distribution of Length1 Values', ylabel='Frequency')
ax.set_title("Before Bootstrapping", fontsize=20)
plt.savefig("fish_ci_before.png")
plt.show();

mean_lengths, n = [], 1000
for i in range(n):
    sample = np.random.choice(data,
                              replace=True,
                              size=len(data))
    sample_mean = sample.mean()
    mean_lengths.append(sample_mean)

# Calculate bootstrapped mean and 95% confidence interval.
boot_mean = np.mean(mean_lengths)
boot_ci = np.percentile(mean_lengths, [2.5, 97.5])
print("Bootstrapped Mean Length = {}, 95% CI = {}".format(boot_mean, boot_ci))

ax = sns.distplot(mean_lengths,bins=10,kde=True,color='skyblue')
ax.axvline(boot_ci[0])
ax.axvline(boot_ci[1])
ax.text(18.2,0.45, round(boot_ci[0], 2))
ax.text(22.3,0.45, round(boot_ci[1], 2))
ax.set_title("After Bootstrapping", fontsize=20)
ax.set(xlabel='Distribution of Length1 Values', ylabel='Frequency')
plt.savefig("fish_ci_after.png")
plt.show();

data = df.copy()
data = data.loc[data["Species"] == "Roach"][["Length1", "Weight"]]
data.sample(2)

data.corr()

data_size, lw_corr = data.shape[0], []
for i in range(1000):
    tmp_df = data.sample(n=data_size, replace=True)
    lw_corr.append(tmp_df["Weight"].corr(tmp_df["Length1"]))

corr_ci = np.percentile(lw_corr, [2.5, 97.5])
print("Correlation 95% Confidence Interval between Weight and Length1 = {}".format(corr_ci))

data = df.copy()
data = data.loc[data["Species"] == "Roach"]["Length1"]

mean_lengths, n = [], len(data)
index = np.arange(n)

for i in range(n):
    jk_sample = data[index != i]
    mean_lengths.append(jk_sample.mean())

mean_lengths_jk = np.mean(np.array(mean_lengths))
jk_variance = (n-1)*np.var(mean_lengths)
print("Jackknife estimate of the mean = {}".format(mean_lengths_jk))
print("Jackknife estimate of the variance = {}".format(jk_variance))

data = df.copy()
data = data.loc[data["Species"] == "Roach"][["Length1", "Length2"]]
data.head()

print(f'Mean of Length1: {data["Length1"].mean():.2f}')
print(f'Mean of Length2: {data["Length2"].mean():.2f}')

sample1 = data["Length1"]
sample2 = data["Length2"]

data = np.concatenate([sample1, sample2])

perm = np.array([np.random.permutation(len(sample1) + len(sample2)) for i in range(10000)])
permuted_1_db = data[perm[:, :len(sample1)]]
permuted_2_db = data[perm[:, len(sample1):]]

samples = np.mean(permuted_1_db, axis=1) - np.mean(permuted_2_db, axis=1)

test_stat = np.mean(sample1) - np.mean(sample2)
p_val = 2*np.sum(samples >= np.abs(test_stat))/10000
print("test_statistic = {}".format(test_stat))
print("p-value = {}".format(p_val))