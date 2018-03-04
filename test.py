import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
from rllab.envs.baseinmiddle import ChaserInvaderEnv
from rllab.envs.normalized_env import normalize
import csv

# If the snapshot file use tensorflow, do:
# import tensorflow as tf
# with tf.Session():
#     [rest of the code]
with open("simulation.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = dict(row)
            with tf.Session() as sess:
                data = joblib.load('/home/aroraiiit/rllab/data/local/experiment/experiment_2018_02_26_17_56_54_0001/params.pkl')
                policy = data['policy']
                #env = normalize(ChaserInvaderEnv())
                env = data['env']
                path = rollout(env, policy, max_path_length=1000,animated=False, speedup=1)
                print(path)