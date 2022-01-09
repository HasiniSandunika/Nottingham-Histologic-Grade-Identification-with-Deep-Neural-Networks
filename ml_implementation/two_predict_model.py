import glob
import cv2
import numpy as np
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import DenseNet201
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt 
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report



print("Two Predict Model Implementation")

def create_numpy(read_path, save_path, set_type, value):
  images = glob.glob(read_path + '/*.JPG')
  random.shuffle(images)
  x = []
  y= []
  r=0
  for i in images:
    temp = cv2.imread(i, cv2.IMREAD_COLOR)
    temp = np.array(temp)
    temp = temp.astype('float32')
    temp = temp/ 255
    temp = cv2.resize(temp, (128, 128), interpolation = cv2.INTER_AREA)
    x.append(temp)
    if int(i[value:value + 1]) == 0:
      r = 0
    else:
      r = 1    
    y.append(r)
  x = np.array(x)
  y = np.array(y)
  x = x.reshape(len(x), 128, 128, 3)
  y = np_utils.to_categorical(y, 2)
  if set_type == 'train':
      np.save(save_path + '/x_train.npy',x)
      np.save(save_path + '/y_train.npy',y)
  else:
      np.save(save_path + '/x_test.npy',x)
      np.save(save_path + '/y_test.npy',y)
  
  
## create numpy arrays for training dataset from processed images
create_numpy('/two_predict_model/dataset/train_data', '/two_predict_model/numpy', 'train', 71) 

## create numpy arrays for testing dataset from processed images
create_numpy('/two_predict_model/dataset/test_data', '/two_predict_model/numpy', 'test', 70) 

## split training dataset for training and validation
x_train_rgb = np.load('/two_predict_model/numpy/x_train.npy')
y_train_rgb = np.load('/two_predict_model/numpy/y_train.npy')
x_train, x_val, y_train, y_val = train_test_split(
    x_train_rgb, y_train_rgb, 
    test_size=0.2, 
    random_state=11
)
print("Size of the X train dataset: ",x_train.shape)
print("Size of the Y train dataset: ",y_train.shape)
print("Size of the X validate dataset: ",x_val.shape)
print("Size of the Y validate dataset: ",y_val.shape)

def create_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))   
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model

## train the model
print('Model')
denseNet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3))
model = create_model(denseNet ,lr = 1e-4)
model.summary()
learn_control = ReduceLROnPlateau(monitor='accuracy', patience=5,
                                  verbose=1,factor=0.2, min_lr=1e-7)
filepath='/two_predict_model/models/weights20.h5'
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
model_fit = model.fit(
    x_train_rgb, y_train_rgb, batch_size=16,
    steps_per_epoch=x_train_rgb.shape[0] / 16,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint]
)
model.save('/two_predict_model/models/model.h5')

## graphs of epochs vs loss and epochs vs accuracy

plt.title('Graph of Epochs vs Loss')
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.plot(model_fit.history['loss']) 
plt.plot(model_fit.history['val_loss']) 
plt.legend(['Training', 'Validation'])
plt.figure() 
plt.show()
 
plt.title('Graph of Epochs vs Accuracy')
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.plot(model_fit.history['accuracy']) 
plt.plot(model_fit.history['val_accuracy']) 
plt.legend(['Training', 'Validation'])
plt.figure() 
plt.show()

## accuracy of the training dataset
score = model.evaluate(x_train_rgb, y_train_rgb, verbose=2) 
print('Training accuracy : ', score[1]*100,'%')

############################## test the model ################################

## accuracy of the testing dataset
model = load_model('/two_predict_model/models/weights20.h5')
model.summary()
y_test = np.load('/two_predict_model/numpy/y_test.npy')
x_test = np.load('/two_predict_model/numpy/x_test.npy')
score = model.evaluate(x_test, y_test , verbose=2) 
print('Testing accuracy : ', score[1]*100,'%')

cm_plot_label =['0', '1']
y_predict = model.predict(x_test).argmax(axis=1)

def plot_normalized_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, with normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

## normalized confusion matrix
cm=confusion_matrix(y_test.argmax(axis=-1), y_predict, normalize='true')
plot_normalized_confusion_matrix(cm, cm_plot_label, title ='Normalized Confusion Matrix for breast cancer grading')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

## non-normalized confusion matrix
cm=confusion_matrix(y_test.argmax(axis=-1), y_predict)
plot_confusion_matrix(cm, cm_plot_label, title ='Non-normalized Confusion Matrix for breast cancer grading')

## classification report
print(classification_report( y_test.argmax(axis=-1), y_predict))