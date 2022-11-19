import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/dense_ncf_result.csv')
df['diff'] = abs(df['label'] - df['preds'])

plt.hist(df['diff'])

# print(df[df['label'] == 1])

# plt.scatter(df['iid'], df['diff'], s=0.1)

plt.savefig('tmp_fig.png')

# from IPython import embed
# embed()
