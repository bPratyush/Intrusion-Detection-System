import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from imblearn.datasets import fetch_datasets
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")
dataset_train=pd.read_csv('/Users/bpratyush/Downloads/NSL_KDD_Train.csv')
dataset_test=pd.read_csv('/Users/bpratyush/Downloads/NSL_KDD_Test.csv')
dataset_train.head()
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)
dataset_train = pd.read_csv("/Users/bpratyush/Downloads/NSL_KDD_Train.csv", header=None, names = col_names)
dataset_test = pd.read_csv("/Users/bpratyush/Downloads/NSL_KDD_Test.csv", header=None, names = col_names)
print('Label distribution Training set:')
print(dataset_train['label'].value_counts())
print()
print('Label distribution Test set:')
print(dataset_test['label'].value_counts())
# Parameters
epochs = 100
batch_size = 180
sample_size = 180
learning_rate = 0.1
tau = 0.001
gamma = 0.2
memory_size = 1000
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Initialize the networks
actor = Actor(122, 23)
q_critic = Critic(122, 23)
v_critic = Critic(122, 1)

# Initialize the optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
q_critic_optimizer = optim.Adam(q_critic.parameters(), lr=learning_rate)
v_critic_optimizer = optim.Adam(v_critic.parameters(), lr=learning_rate)
class EnvironmentalAgent:
    def __init__(self, input_size, output_size, learning_rate):
        self.actor = Actor(input_size, output_size)
        self.q_critic = Critic(input_size, output_size)
        self.v_critic = Critic(input_size, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=learning_rate)
        self.v_critic_optimizer = optim.Adam(self.v_critic.parameters(), lr=learning_rate)

    def get_next_state(self, data, current_state, action):
        # Find the index of the current state in the data
        index = np.where(np.all(data == current_state, axis=1))[0][0]
        
        # Use the action to determine the index of the next state
        next_index = (index + action) % len(data)
        
        # Get the next state
        next_state = data[next_index]
        
        return next_state

    def resample_data(self, data, labels, counts):
        # Convert the data and labels to a pandas DataFrame
        df = pd.DataFrame(data)
        df['label'] = labels

        # Resample each class to the specified count
        resampled_df = pd.DataFrame()
        for label, count in counts.items():
            class_df = df[df['label'] == label]
            resampled_class_df = class_df.sample(count, replace=True)
            resampled_df = pd.concat([resampled_df, resampled_class_df])

        # Shuffle the resampled DataFrame
        resampled_df = resampled_df.sample(frac=1).reset_index(drop=True)

        # Convert the resampled DataFrame back into data and labels
        resampled_data = resampled_df.drop('label', axis=1).values
        resampled_labels = resampled_df['label'].values

        return resampled_data, resampled_labels
environmental_agent = EnvironmentalAgent(122, 23, learning_rate)
print("Actor Model:")
print(environmental_agent.actor)
print("\nQ-Critic Model:")
print(environmental_agent.q_critic)
print("\nV-Critic Model:")
print(environmental_agent.v_critic)
class ClassifierAgent:
    def __init__(self, input_size, output_size, learning_rate):
        self.actor = Actor(input_size, output_size)
        self.q_critic = Critic(input_size, output_size)
        self.v_critic = Critic(input_size, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=learning_rate)
        self.v_critic_optimizer = optim.Adam(self.v_critic.parameters(), lr=learning_rate)
classifier_agent = ClassifierAgent(122, 5, learning_rate)
print("Actor Model:")
print(classifier_agent.actor)
print("\nQ-Critic Model:")
print(classifier_agent.q_critic)
print("\nV-Critic Model:")
print(classifier_agent.v_critic)
def calculate_classifier_reward(predicted_label, actual_label, high_percentage_categories, low_percentage_categories):
    if predicted_label == actual_label:
        if actual_label in high_percentage_categories:
            return 1
        elif actual_label in low_percentage_categories:
            return 2
    return 0

def calculate_environment_reward(predicted_label, actual_label, high_percentage_categories, low_percentage_categories):
    if predicted_label != actual_label:
        if actual_label in high_percentage_categories:
            return 1
        elif actual_label in low_percentage_categories:
            return 2
    return 0
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
# Scale continuous features
continuous_features = dataset_train.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
dataset_train[continuous_features] = scaler.fit_transform(dataset_train[continuous_features])
num_continuous_features = len(continuous_features)
print(f"Number of continuous features: {num_continuous_features}")
# One-hot encode categorical features
categorical_features=['protocol_type', 'service', 'flag'] 
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("OneHot", one_hot, categorical_features)], remainder='passthrough')
num_categorical_features = len(categorical_features)
print(f"Number of categorical features: {num_categorical_features}")
# Define the mapping from specific attack types to general categories
attack_mapping = {
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r',
    'normal': 'normal'
}

# Apply the mapping to the labels
dataset_train['label'] = dataset_train['label'].map(attack_mapping)

# Check the distribution of the general categories
print(dataset_train['label'].value_counts())
num_features = dataset_train.shape[1] - 1 #-1 for label
print(f'Total number of features: {num_features}')
data = dataset_train.drop('label', axis=1)
labels = dataset_train['label']
counts = {'normal': 4689, 'dos': 2789, 'probe': 2187, 'r2l': 5795, 'u2r': 3971}
environmental_agent = EnvironmentalAgent(122, 23, learning_rate)
resampled_data, resampled_labels = environmental_agent.resample_data(data, labels, counts)
resampled_dataset = pd.DataFrame(resampled_data, columns=dataset_train.columns[:-1])
resampled_dataset['label'] = resampled_labels
import pandas as pd
original_counts = dataset_train['label'].value_counts()
resampled_counts = resampled_dataset['label'].value_counts()
counts_df = pd.DataFrame({
    'normal': [original_counts['normal'], resampled_counts['normal']],
    'dos': [original_counts['dos'], resampled_counts['dos']],
    'probe': [original_counts['probe'], resampled_counts['probe']],
    'r2l': [original_counts['r2l'], resampled_counts['r2l']],
    'u2r': [original_counts['u2r'], resampled_counts['u2r']],
}, index=['Original', 'Resampled'])
print(counts_df)
import random

class SAC:
    def __init__(self, actor, q_critic, v_critic, actor_optimizer, q_critic_optimizer, v_critic_optimizer, gamma=0.99, tau=0.005):
        self.actor = actor
        self.q_critic = q_critic
        self.v_critic = v_critic
        self.actor_optimizer = actor_optimizer
        self.q_critic_optimizer = q_critic_optimizer
        self.v_critic_optimizer = v_critic_optimizer
        self.gamma = gamma
        self.tau = tau
        self.memory_env = []
        self.memory_class = []

    def update(self, state, action, reward, next_state, done):
        # Add to memory
        self.memory_env.append((state, action, reward, next_state, done))
        self.memory_class.append((state, action, reward, next_state, done))

        # Sample a batch of experiences from memory
        batch_env = random.sample(self.memory_env, min(len(self.memory_env), batch_size))
        batch_class = random.sample(self.memory_class, min(len(self.memory_class), batch_size))

        for state, action, reward, next_state, done in batch_env:
            self.update_env(state, action, reward, next_state, done)

        for state, action, reward, next_state, done in batch_class:
            self.update_class(state, action, reward, next_state, done)

    def update_env(self, state, action, reward, next_state, done):
        # Update environment agent (Q critic)
        with torch.no_grad():
            next_action = self.actor(next_state)
            q_next_state = self.q_critic(next_state, next_action)
            q_target = reward + (1 - done) * self.gamma * q_next_state

        q_current = self.q_critic(state, action)
        q_loss = nn.MSELoss()(q_current, q_target)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

    def update_class(self, state, action, reward, next_state, done):
        # Update classifier agent (V critic and actor)
        with torch.no_grad():
            next_action = self.actor(next_state)
            q_next_state = self.q_critic(next_state, next_action)
            v_target = q_next_state

        v_current = self.v_critic(state)
        v_loss = nn.MSELoss()(v_current, v_target)
        self.v_critic_optimizer.zero_grad()
        v_loss.backward()
        self.v_critic_optimizer.step()

        action_pred = self.actor(state)
        q_current = self.q_critic(state, action_pred)
        actor_loss = -torch.mean(q_current)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of V critic
        for target_param, param in zip(self.v_critic.parameters(), self.q_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            