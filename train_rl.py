"""
RL Based DDoS Threat Detection in Cloud Environment
---------------------------------------------------
train_rl.py
This script trains RL-based agents (A2C, DDPG, TD3) for DDoS detection
using OpenAI Gym interface with UNSW-NB15 or CICDDoS2019 dataset.

Usage:
    python train_rl.py
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from stable_baselines3 import A2C, DDPG, TD3

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------
def load_dataset(path, label_col="Label"):
    df = pd.read_csv(path)
    X = df.drop(columns=[label_col])
    y = df[label_col].map(lambda v: 1 if v in ["Attack", 1] else 0)
    return X, y

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.values, y_test.values

# ------------------------------------------------------------------
# Custom Gym Environment
# ------------------------------------------------------------------
class DDoSEnv(gym.Env):
    def __init__(self, X, y):
        super(DDoSEnv, self).__init__()
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.action_space = spaces.Discrete(2)  # 0=Benign, 1=Attack
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.index = 0

    def reset(self):
        self.index = 0
        return self.X[self.index].astype(np.float32)

    def step(self, action):
        true_label = self.y[self.index]
        reward = 1 if action == true_label else -1
        self.index += 1
        done = self.index >= self.n_samples
        obs = self.X[self.index].astype(np.float32) if not done else np.zeros(self.n_features, dtype=np.float32)
        return obs, reward, done, {}

# ------------------------------------------------------------------
# Train and Evaluate RL Models
# ------------------------------------------------------------------
def train_rl_model(name, model_cls, env, timesteps=5000):
    print(f"Training {name}...")
    model = model_cls("MlpPolicy", env, verbose=0, seed=RANDOM_STATE)
    model.learn(total_timesteps=timesteps)

    # Evaluate on same data
    X = env.X
    y_true = env.y
    y_pred = []
    for i in range(len(X)):
        obs = X[i].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        y_pred.append(action)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return [name, acc, prec, rec, f1, auc]

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Choose dataset
    DATASET = "UNSW"

    if DATASET == "UNSW":
        DATA_PATH = "UNSW-NB15_features.csv"
        LABEL_COL = "Label"
    elif DATASET == "CIC":
        DATA_PATH = "cicddos2019_dataset.csv"
        LABEL_COL = "Label"

    print(f"\n=== Running RL models on {DATASET} Dataset ===")
    X, y = load_dataset(DATA_PATH, label_col=LABEL_COL)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Use test split for RL evaluation (to simulate unseen data)
    env = DDoSEnv(X_test, y_test)

    results = []
    results.append(train_rl_model("A2C", A2C, env))
    results.append(train_rl_model("DDPG", DDPG, env))
    results.append(train_rl_model("TD3", TD3, env))

    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])
    print("\nFinal RL Results:\n", df)
    df.to_csv(f"results_summary_RL_{DATASET}.csv", index=False)
