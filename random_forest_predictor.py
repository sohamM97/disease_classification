import argparse
import numpy as np
import pandas as pd
import os
import json
import subprocess
import sys
from io import StringIO

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

from sagemaker_containers.beta.framework import worker

model_dir = '/opt/ml/model'
cols_list = ['gender_categorized','age_binned','blood_pressure','pulse_rate_categorized','respiration_rate_categorized','BODY_TEMPERATURE',
                'BODY_WEIGHT','HEIGHT','SPO2_categorized'] #'district_categorized',
vitals =['SYSTOLIC_BP','DIASTOLIC_BP','PULSE','RESPIRATION_RATE','BODY_TEMPERATURE','BODY_WEIGHT','HEIGHT','SPO2']

def drop_unnecessary_cols(df):

    df.drop(columns=['HEART_RATE','HEAD_CIRCUMFERENCE','UPPER_ARM_CIRCUMFERENCE'],inplace = True)
    df = df[pd.notnull(df['DISEASE_ID'])]
    #df.drop(df[df.REFERRED == 'Y'].index, inplace=True)
    df.drop(df[df.DISEASE_ID == '0'].index, inplace=True)
    df = df[df.SYMPTOM_ID.notnull()]
    df[vitals] = df[vitals].replace(0, np.nan)
    df.drop(df[df.DISEASE_ID == 'Confusion'].index, inplace=True)

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
    
    '''WARNING: These values need to be entered as per the median of data'''

    df.SYSTOLIC_BP.fillna(120,inplace=True)
    df.DIASTOLIC_BP.fillna(80,inplace=True)
    df.PULSE.fillna(84,inplace=True)
    df.RESPIRATION_RATE.fillna(18,inplace=True)
    df.BODY_TEMPERATURE.fillna(98.2,inplace=True)
    df.BODY_WEIGHT.fillna(45,inplace=True)
    df.HEIGHT.fillna(151,inplace=True)
    df.SPO2.fillna(99,inplace=True)

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

def categorize_district(df,model_dir,type='train'):

    if type == 'train':
        
        label_enc_district = LabelEncoder()
        df['district_categorized'] = label_enc_district.fit_transform(df.DISTRICT_NAME)
        joblib.dump(label_enc_district, os.path.join(model_dir, "district_encoder.joblib"))
        
    elif type == 'test':
        
        label_enc_district = joblib.load(os.path.join(model_dir, "district_encoder.joblib"))
        df['district_categorized'] = label_enc_district.transform(df.DISTRICT_NAME)
        
    return df

def encode_symptoms(df,model_dir,type='train'):

    df.SYMPTOM_ID = [[symptom for symptom in symptoms_row if symptom!=''] for symptoms_row in df.SYMPTOM_ID.str.split('~').tolist()]
    
    symptoms_encoded = []
    
    if type == 'train':
                
        mlb_symtoms = MultiLabelBinarizer()
        symptoms_encoded = mlb_symtoms.fit_transform(df.SYMPTOM_ID)
        joblib.dump(mlb_symtoms, os.path.join(model_dir, "symptoms_binarizer.joblib"))
        
    elif type == 'test':
                
        mlb_symtoms = joblib.load(os.path.join(model_dir, "symptoms_binarizer.joblib"))
        symptoms_encoded = mlb_symtoms.transform(df.SYMPTOM_ID)
        
    df['symptoms_encoded'] = symptoms_encoded.tolist()
    
#     print(mlb_symtoms.classes_)

    return df, symptoms_encoded

def encode_diseases(df,model_dir):

    df.DISEASE_ID = (df.DISEASE_ID.str.split('~'))
    mlb_diseases = MultiLabelBinarizer()
    diseases_encoded = mlb_diseases.fit_transform(df.DISEASE_ID)
    
    joblib.dump(mlb_diseases, os.path.join(model_dir, "diseases_binarizer.joblib"))
    
    df['diseases_encoded'] = diseases_encoded.tolist()

    return df, diseases_encoded

def format_data(df,model_dir,type='train'):

    if type == 'train':
        df = drop_unnecessary_cols(df)
        df = drop_wrong_vitals(df)
        
    df = bin_ages(df)
    df = fill_nas(df)
    df = categorize_bp(df)
    df = categorize_pulse_rate(df)
    df = categorize_respiration_rate(df)
    df = categorize_spo2(df)
    df = categorize_gender(df)
    #df = categorize_district(df,model_dir,type)
    df, symptoms_encoded = encode_symptoms(df,model_dir,type)
    
    if type == 'train':
        df, diseases_encoded = encode_diseases(df,model_dir)
        return df, symptoms_encoded, diseases_encoded  
    elif type == 'test':
        return df, symptoms_encoded



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
    try:
        raw_data = [ pd.read_excel(file) for file in input_files ]
    except ImportError as e:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'xlrd'])
        raw_data = [ pd.read_excel(file) for file in input_files ]
        
    train_data = pd.concat(raw_data)
    
    #print(train_data.columns)
      
    # formatting
    
    train_data, train_symptoms, train_diseases = format_data(train_data,args.model_dir)
    X_train = train_data[cols_list].values
    X_train = np.hstack((X_train, train_symptoms))
    Y_train = train_diseases

    # Here we support a single hyperparameter, 'max_depth'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    max_depth = args.max_depth

    # Now use scikit-learn's classifier to train the model.
    clf = OneVsRestClassifier(RandomForestClassifier(max_depth = max_depth))
    #clf = RandomForestClassifier(max_depth = max_depth)
    #clf = MultiOutputClassifier(RandomForestClassifier(max_depth = max_depth))
    clf.fit(X_train, Y_train)
    
    #print("FIT!!!")
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    
def input_fn(input_data, content_type):
    
    if content_type == 'text/csv':
        
        df = pd.read_csv(StringIO(input_data))
        df, symptoms_enc = format_data(df,model_dir,'test')
        X_test = df[cols_list].values
        X_test = np.hstack((X_test, symptoms_enc)) 
       
        return X_test
        
    else:
        raise ValueError("{} not supported by script!".format(content_type))
    

def predict_fn(input_data, model):
    
    #prediction = model.predict(input_data)
    return model.predict_proba(input_data)
    #return np.array([prediction, pred_prob]) 
    
    
def output_fn(prediction, accept):
    
    mlb_diseases = joblib.load(os.path.join(model_dir, "diseases_binarizer.joblib"))
    diseases_list = list(mlb_diseases.classes_)
    output_list = []
    
    for encoded_disease in prediction:
        
        diseases = []
        for i, disease in enumerate(diseases_list):
            if encoded_disease[i] >= 0.3:
                diseases.append(disease)
        
        output_list.append(diseases)
            
    if accept == "application/json":
        
        if len(output_list) == 1:         
            json_output = {"diseases": output_list[0]}
            
        else:    
            
            instances = []
            for row in output_list:
                instances.append({"diseases": row})

            json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)

    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
    
    
    return output_list

def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf