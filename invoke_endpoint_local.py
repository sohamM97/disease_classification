import boto3
import sagemaker
import math
import dateutil
from time import time

start_time = time()

boto_session = boto3.Session(profile_name='cli_ml_access',region_name='us-east-1')
sess = sagemaker.Session(boto_session=boto_session)

endpoint_final = 'sagemaker-scikit-learn-2019-12-18-10-11-18-055'

predictor = sagemaker.predictor.RealTimePredictor(
    endpoint=endpoint_final,
    sagemaker_session=sess,
    content_type='text/csv')

f = open('Test DF single.csv', 'r')
print(predictor.predict(f.read()))

end_time = time()

print(end_time - start_time)