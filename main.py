import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from scipy.stats import shapiro

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

print(train_df)