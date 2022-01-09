import numpy as np
from keras.models import load_model



print("Model Comparison (Case 1 and 2")

y_test = np.load('/four_predict_model/numpy/y_test.npy')
x_test = np.load('/four_predict_model/numpy/x_test.npy')

model_predict_2 = load_model('/two_predict_model/models/weights20.h5')
model_predict_3 = load_model('/three_predict_model/models/weights30.h5')
model_predict_4 = load_model('/four_predict_model/models/weights25.h5')

print("Size of the X test dataset: ",x_test.shape)
print("Size of the Y test dataset: ",y_test.shape)

score_case_1 = model_predict_4.evaluate(x_test, y_test , verbose=2) 

y_predict_case_2_2predict = model_predict_2.predict(x_test).argmax(axis=1)

true_count=0
false_count=0
r=-1
for i in range(len(y_predict_case_2_2predict)):
    if y_predict_case_2_2predict[i]==0:
        r=0
    if y_predict_case_2_2predict[i]==1:
        temp = x_test[i].reshape(1, 128, 128, 3)    
        r=model_predict_3.predict(temp).argmax(axis=-1)[0]+1
    if  r==y_test.argmax(axis=1)[i]:
        true_count=true_count+1
    if r!=y_test.argmax(axis=1)[i]:
        false_count=false_count+1

print('True count: ',true_count)
print('False count: ',false_count)
print("\n") 

score_case_2= (true_count/len(y_predict_case_2_2predict))*100

## case 1 accuracy
print('Testing accuracy score for case 1: ', score_case_1[1]*100,'%')     

## case 2 accuracy   
print("Testing accuracy score for case 2: ", score_case_2,'%')