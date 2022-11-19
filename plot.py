import pandas as pd
import numpy as np

from scipy import stats
from models.ncf import NCF
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

meta_info = pd.read_csv("stimuli2018_weUsed.csv")

genres = meta_info['Genre'].values
mode = int(stats.mode(genres).mode)
genres = [int(i)/10 if not np.isnan(i) else 9 for i in genres]

genres_dict = {i:(int(g) if not np.isnan(g) else mode) for i,g in enumerate(genres) }

user_num, item_num = 944, 1000

ckpt = 'save/baseline/model_epoch_19_loss_1.9589422941207886.bin'
def load_model(ckpt):
    model = NCF(user_num, item_num, 30, 5, 0.1, 'MLP')
    model_state_dict = torch.load(ckpt)
    model.load_state_dict(model_state_dict['model_state_dict'])
    return model

model = load_model(ckpt)

# item_embeddings = []
# user_embed = model.embed_user_GMF
item_embeds = model.embed_item_MLP.weight.detach().cpu().numpy()

pca = PCA(n_components=2)
reduced_item_embeds = pca.fit_transform(item_embeds)
print("Variance explained ratio:", np.sum(pca.explained_variance_ratio_))

plt.scatter(reduced_item_embeds[:, 0], reduced_item_embeds[:, 1], c=genres, linewidths=0.1)
plt.savefig("Tmp.jpg")

# torch.tensor(199)
from IPython import embed 
embed() or exit(0)
# for i in range(item_num):
    