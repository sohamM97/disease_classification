import argparse
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib


def drop_unnecessary_cols(df):

    df.drop(columns=['HEART_RATE','HEAD_CIRCUMFERENCE','UPPER_ARM_CIRCUMFERENCE'],inplace = True)
    df = df[pd.notnull(df['DISEASE_ID'])]
    df.drop(df[df.REFERRED == 'Y'].index, inplace=True)
    df.drop(df[df.DISEASE_ID == '0'].index, inplace=True)
    df = df[df.SYMPTOM_ID.notnull()]

    return df

def drop_wrong_vitals(df):

    df.drop(df[(df.SYSTOLIC_BP>230) | (df.DIASTOLIC_BP>150)].index, inplace = True)
    df.drop(df[(df.SYSTOLIC_BP<80) | (df.DIASTOLIC_BP<50)].index, inplace=True)
    df.drop(df[(df.BODY_TEMPERATURE<90) | (df.BODY_TEMPERATURE>110)].index, inplace=True)
    df.drop(df[df.AGE>100].index, inplace=True)
    df.drop(df[(df.SPO2>100) | (df.SPO2<80)].index,inplace=True)
    df.drop(df[(df.PULSE<50) | (df.PULSE>120)].index,inplace=True)
    df.drop(df[(df.RESPIRATION_RATE>30) | (df.RESPIRATION_RATE<13)].index,inplace=True)
    df.drop(df[df.BODY_WEIGHT>200].index,inplace=True)
    df.drop(df[(df.HEIGHT>200) | (df.HEIGHT<45)].index,inplace=True)

    return df

def bin_ages(df):

    bins = [x for x in range(0,101,5)]
    labels = [x for x in range(1,21)]
    df['age_binned'] = pd.cut(df.AGE,bins,labels=labels,include_lowest=True)
    
    return df

def fill_nas(df):

    df.SYSTOLIC_BP.fillna(120,inplace=True)
    df.DIASTOLIC_BP.fillna(80,inplace=True)
    df.PULSE.fillna(df.PULSE.median(),inplace=True)
    df.RESPIRATION_RATE.fillna(df.RESPIRATION_RATE.median(),inplace=True)
    df.BODY_TEMPERATURE.fillna(df.BODY_TEMPERATURE.median(),inplace=True)
    df.BODY_WEIGHT.fillna(df.BODY_WEIGHT.median(),inplace=True)
    df.HEIGHT.fillna(df.HEIGHT.median(),inplace=True)
    df.SPO2.fillna(df.SPO2.median(),inplace=True)

    return df

def categorize_bp(df):

    df['blood_pressure']=0
    df['blood_pressure'][(df.SYSTOLIC_BP<90) | (df.DIASTOLIC_BP<60)] = 0
    df['blood_pressure'][((df.SYSTOLIC_BP>=90) & (df.SYSTOLIC_BP<=120)) & ((df.DIASTOLIC_BP>=60) & (df.DIASTOLIC_BP<=80))] = 1
    df['blood_pressure'][((df.SYSTOLIC_BP>120) & (df.SYSTOLIC_BP<130)) & ((df.DIASTOLIC_BP>=60) & (df.DIASTOLIC_BP<=80))] = 2
    df['blood_pressure'][((df.SYSTOLIC_BP>=130) & (df.SYSTOLIC_BP<140)) | ((df.DIASTOLIC_BP>80) & (df.DIASTOLIC_BP<90))] = 3
    df['blood_pressure'][((df.SYSTOLIC_BP>=140) & (df.SYSTOLIC_BP<180)) | ((df.DIASTOLIC_BP>=90) & (df.DIASTOLIC_BP<120))] = 4
    df['blood_pressure'][(df.SYSTOLIC_BP>=180) | (df.DIASTOLIC_BP>=120)] = 5

    return df

def categorize_pulse_rate(df):
    
    df['pulse_rate_categorized']=0
    df['pulse_rate_categorized'][df.PULSE<60]=0
    df['pulse_rate_categorized'][(df.PULSE>=60) & (df.PULSE<=100)]=1
    df['pulse_rate_categorized'][df.PULSE>100]=2

    return df

def categorize_respiration_rate(df):

    df['respiration_rate_categorized']=0
    df['respiration_rate_categorized'][df.RESPIRATION_RATE<16]=0
    df['respiration_rate_categorized'][(df.RESPIRATION_RATE>=16) & (df.RESPIRATION_RATE<=20)]=1
    df['respiration_rate_categorized'][df.RESPIRATION_RATE>20]=2

    return df

def categorize_spo2(df):

    df['SPO2_categorized']=0
    df['SPO2_categorized'][df.SPO2>=95]=1
    df['SPO2_categorized'][df.SPO2<95]=0

    return df

def categorize_gender(df):

    df['gender_categorized'] = 0
    df['gender_categorized'][df.GENDER == 'Male'] = 0
    df['gender_categorized'][df.GENDER == 'Female'] = 1

    return df

def categorize_district(df):

    label_enc_district = LabelEncoder()
    df['district_categorized'] = label_enc_district.fit_transform(df.DISTRICT_NAME)

    return df

def encode_symptoms(df):

    df.SYMPTOM_ID = (df.SYMPTOM_ID.str.split('~'))
    mlb_symtoms = MultiLabelBinarizer()
    print(df.SYMPTOM_ID.head())
    symptoms_encoded = mlb_symtoms.fit_transform(df.SYMPTOM_ID)
    df['symptoms_encoded'] = symptoms_encoded.tolist()

    return df, symptoms_encoded

def encode_diseases(df):

    df.DISEASE_ID = (df.DISEASE_ID.str.split('~'))
    mlb_diseases = MultiLabelBinarizer()
    diseases_encoded = mlb_diseases.fit_transform(df.DISEASE_ID)
    df['diseases_encoded'] = diseases_encoded.tolist()

    return df, diseases_encoded

def format_data(df):

    df = drop_unnecessary_cols(df)
    df = drop_wrong_vitals(df)
    df = bin_ages(df)
    df = fill_nas(df)
    df = categorize_bp(df)
    df = categorize_pulse_rate(df)
    df = categorize_respiration_rate(df)
    df = categorize_spo2(df)
    df = categorize_gender(df)
    df = categorize_district(df)
    df, symptoms_encoded = encode_symptoms(df)
    df, diseases_encoded = encode_diseases(df)

    return df, symptoms_encoded, diseases_encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_depth', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file,encoding = "ISO-8859-1", engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)
    
    print(train_data.columns)
      
    # formatting

    cols_list = ['gender_categorized','age_binned','district_categorized','blood_pressure',
                'pulse_rate_categorized','respiration_rate_categorized','BODY_TEMPERATURE',
                'BODY_WEIGHT','HEIGHT','SPO2_categorized']

    train_data, train_symptoms, train_diseases = format_data(train_data)
    X_train = train_data[cols_list].values
    X_train = np.hstack((X_train, train_symptoms))
    Y_train = train_diseases

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    max_depth = args.max_depth

    # Now use scikit-learn's classifier to train the model.
    clf = OneVsRestClassifier(RandomForestClassifier(max_depth = max_depth))
    clf.fit(X_train, Y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf