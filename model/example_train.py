import numpy as np
import itertools
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng()
import pandas as pd
from dadapy import Data
from dadapy._utils import utils as ut
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
from scipy.spatial import distance

from sentence_transformers import SentenceTransformer, util
import random

import anthropic
import openai
import gc
from scipy import stats
import torch
import os
import json

set_seed(42)

model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

def get_embedding(text, model):
    return model.encode(text)

items_embs = get_embedding(sentences_bdi, model)

predictions_MiniLM1 = train(cosine = True, type_embs = 1)
