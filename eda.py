import pandas as pd
import numpy as np
# df = pd.read_csv('stimuli2018_weUsed.csv', header=0, names=['Source','Number','Title','Artist','Genre','Subgenre/Style','Year','Clip name','Source all','Genre all'])
# print(df['Genre'].value_counts())

# y = np.load('extracted_features/onehot_labels.npy')

# labels = y.argmax(axis=1)

# unique, counts = np.unique(labels, return_counts=True)
# print(unique, counts)

ratings = pd.read_csv("musicRatings.csv", header=None)

dense_ratings = ratings.iloc[:, :300]
arr = dense_ratings.to_numpy()

u, s, v = np.linalg.svd(arr)
sigma = np.pad(np.diag(s),((0, 644), (0, 0)))

recover = u @ sigma @ v


from IPython import embed
embed()