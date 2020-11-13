from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

def generate_user_data(dataset='Mnist', num_users=20, num_labels=2, type='niid'):
    
    