#model training part using xgboost, i used logistic regression and random forrest but xgboost outperforms them.
#the model is trained on the ansur data for now, it will be trained on on bigger data sets in future updates.
#if you see any part of the code that is not used, then it was probably used when i was testing and tuning the model.
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib 

#file paths
FEMALE_DATA_PATH = "data/ANSUR_II_FEMALE_Public.csv"
MALE_DATA_PATH = "data/ANSUR_II_MALE_Public.csv"
MODEL_SAVE_PATH = "model/model.joblib"

def load_and_preprocess_data(female_data_path, male_data_path):
    female_df = pd.read_csv(female_data_path)
    male_df = pd.read_csv(male_data_path)

    #columns in the ansur csv files have different names, we match the most relevant ones but more can be added.
    column_mapping = {
        "STATURE": "stature",
        "BIACROMIAL_BRTH": "biacromialbreadth",
        "CHEST_BRTH": "chestbreadth",
        "CHEST_CIRC": "chestcircumference",
        "HIP_BRTH": "hipbreadth",
        "BUTTOCK_CIRC": "buttockcircumference",
        "WAIST_CIRC-OMPHALION": "waistcircumference",
        "WAIST_BRTH_OMPHALION": "waistbreadth",
        "THIGH_CIRC-PROXIMAL": "thighcircumference",
        "FOREARM_CIRC-FLEXED": "forearmcircumferenceflexed",
        "WRIST_CIRC-STYLION": "wristcircumference",
        "ANKLE_CIRC": "anklecircumference",
        "BIDELTOID_BRTH": "bideltoidbreadth",
        "CALF_CIRC": "calfcircumference",
        "FOOT_LNTH": "footlength",
        "FOREARM-HAND_LENTH": "forearmhandlength"
    }
    male_df = male_df.rename(columns=column_mapping)
    #the measurements chosen to calculate the ratios on.
    key_measurements = [
        "stature", "biacromialbreadth", "chestbreadth", "chestcircumference",
        "hipbreadth", "buttockcircumference", "waistcircumference",
        "waistbreadth", "thighcircumference", "forearmcircumferenceflexed",
        "wristcircumference", "anklecircumference", "bideltoidbreadth",
        "calfcircumference", "footlength", "forearmhandlength"
    ]

    female_df = female_df[key_measurements]
    male_df = male_df[key_measurements]
    male_df['gender'] = 'male'
    female_df['gender'] = 'female'

    combined_df = pd.concat([male_df, female_df], ignore_index=True)

    return combined_df

#Calculating the ratios.
def calculate_ratio_features(df):
    ratio_features = {
        'WHR': ('waistcircumference', 'buttockcircumference'),          
        'HBS': ('hipbreadth', 'stature'),   
        'CS': ('chestcircumference', 'stature'),  
        'FSR': ('forearmcircumferenceflexed', 'stature'),  
        'CBR': ('calfcircumference', 'buttockcircumference'),   
        'BBSR': ('biacromialbreadth', 'stature'),              
        'BBHB': ('biacromialbreadth', 'hipbreadth'),
        'ANKLS': ('anklecircumference', 'stature'),
        'FLS': ('forearmhandlength', 'stature'),
        'WCS': ('wristcircumference', 'stature')
    }

    #Replace zeros in denominator columns to avoid division by zero ( there aren't any afaik but just in case other data is added)
    denominator_columns = list(set([denominator for _, denominator in ratio_features.values()]))
    df[denominator_columns] = df[denominator_columns].replace(0, 0.1)

    #Create the ratio columns.
    for ratio_name, (numerator, denominator) in ratio_features.items():
        df[ratio_name] = df[numerator] / df[denominator]

    return df, list(ratio_features.keys())

#Preparing training data.
def prepare_training_data(df, ratio_feature_names):
    train_df = df[ratio_feature_names + ['stature', 'gender']]
    train_df['gender'] = train_df['gender'].map({'male': 1, 'female': 0})
    X = train_df.drop('gender', axis=1)
    y = train_df['gender'].astype(int)
    return X, y

#using xgboost.
#i used bayesian optimization with hyperopt to get these hyperparameters (methode below).
"""
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'learning_rate': params['learning_rate'],
        'n_estimators': int(params['n_estimators']),
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda'],
        'eval_metric': 'logloss',
        'random_state': 42
    }

    # Initialize the model with given parameters
    model = XGBClassifier(**params)

    # Use cross-validation to evaluate the model
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    # Return the negative of accuracy because Hyperopt minimizes the objective
    return {'loss': -cv_score, 'status': STATUS_OK}

space = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
best['max_depth'] = int(best['max_depth'])
best['n_estimators'] = int(best['n_estimators'])

print("Best hyperparameters:", best)
"""
#changing them will change the models output. 
def train_model(X, y):
    tuned_hyperparameters = {
        'learning_rate': 0.13338404745122975,
        'max_depth': 8,
        'min_child_weight': 3,
        'n_estimators': 250,
        'reg_alpha': 0.12652039232323586,
        'reg_lambda': 0.3114131531865676,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**tuned_hyperparameters)
    model.fit(X, y)
    return model

#save using joblib.
def save_model(model, save_path):
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

#main.
if __name__ == "__main__":
    df = load_and_preprocess_data(FEMALE_DATA_PATH, MALE_DATA_PATH)
    print(f"Combined DataFrame shape: {df.shape}")

    df, ratio_feature_names = calculate_ratio_features(df)

    X, y = prepare_training_data(df, ratio_feature_names)

    model = train_model(X, y)
    print("Model training complete.")

    save_model(model, MODEL_SAVE_PATH)