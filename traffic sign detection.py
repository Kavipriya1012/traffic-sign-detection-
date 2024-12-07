importmatplotlib.pyplot as plt
importnumpy as np
importtensorflow as tf
import pandas as pd
importseaborn as sns
import pickle
import random
importos
fordirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename))
withopen("/content/train.p", mode='rb') astraining_data:
train = pickle.load(training_data)
fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))
axes = axes.ravel()
n_training = len(X_train)
for i innp.arange(0, W_grid*L_grid):
index=np.random.randint(0, n_training)
axes[i].imshow(X_train[index])
axes[i].set_title(y_train[index], fontsize=15)
axes[i].axis("off")
plt.subplots_adjust(hspace=0.4)
fromsklearn.utilsimport shuffle
X_train, y_train = shuffle(X_train, y_train)
X_train_gray=np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray=np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray=np.sum(X_validation/3, axis=3, keepdims=True)
X_train_gray.shape
X_train_gray_norm=(X_train_gray-128)/128
X_test_gray_norm=(X_test_gray-128)/128
X_validation_gray_norm=(X_validation_gray-128)/128
X_train_gray_norm
i = random.randint(1, len(X_train_gray))
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')
fromtw.kerasimport datasets, layers, models, layers
CNN=models.Sequential()
CNN.add(layers.Conv2D(6, (5,5), activation='relu',
input_shape=(32,32,1)))
CNN.add(layers.AveragePooling2D())
CNN.add(layers.Dropout(0.2))
CNN.add(layers.Conv2D(16, (5,5), activation='relu'))
CNN.add(layers.AveragePooling2D())
CNN.add(layers.Flatten())
CNN.add(layers.Dense(120, activation='relu'))
CNN.add(layers.Dense(84, activation='relu'))
CNN.add(layers.Dense(43, activation='softmax'))
CNN.summary()
CNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
history=CNN.fit(X_train_gray_norm, y_train, batch_size=500, epochs=30,
verbose=1,validation_data=(X_validation_gray_norm, y_validation))
score = CNN.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))
history.history.keys()
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(accuracy))
plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation loss')
epochs=range(len(accuracy))
plt.plot(epochs, accuracy, 'ro', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
predicted_classes=CNN.predict(X_test_gray_norm).argmax(axis=1)
y_true = y_test
fromsklearn.metricsimportconfusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)
plt.show()
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()
for i innp.arange(0, L*W):
axes[i].imshow(X_test[i])
axes[i].axis('off')
plt.subplots_adjust(wspace = 1)
