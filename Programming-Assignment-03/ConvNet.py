# Building Convolutional Neural Networks to Classify the Dog and Cat Images. This is a Binary Classification Model i.e. 0 or 1
# Used Dataset -- a Subset (10,000) Images ==> (8,000 for training_set: 4,000 Dogs and 4,000 Cats) and (2,000 for test_set: 1,000 Dogs and 1,000 Cats of Original Dataset (25,000 images) of Dogs vs. Cats | Kaggle
# Original Dataset link ==> https://www.kaggle.com/c/dogs-vs-cats/data
# You might use 25 or more epochs and 8000 Samples per epoch

# Installing Theano
# Installing Tensorflow
# Installing Keras

# Part 1 - Building the ConvNet

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the ConvNet
classifier = Sequential()

# Step 1 - Building the Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Building the Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding The Second Convolutional Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Building the Flattening Layer
classifier.add(Flatten())

# Step 4 - Building the Fully Connected Layer
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the ConvNet
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the ConvNet to the Images

from keras.preprocessing.image import ImageDataGenerator
# ..... Fill the Rest (a Few Lines of Code!)
IDGen_training = ImageDataGenerator()
IDGen_testing = ImageDataGenerator()
ID_training = IDGen_training.flow_from_directory(directory='ConvNet_dataset\\training_set', target_size=(64, 64), subset='training', batch_size=8000, class_mode='binary')
ID_testing = IDGen_testing.flow_from_directory(directory='ConvNet_dataset\\test_set', target_size=(64, 64), batch_size=2000, class_mode='binary')
classifier.summary()
print(ID_training.class_indices)
print(ID_training.image_shape)
print(ID_testing.class_indices)
print(ID_testing.image_shape)
classifier.fit(ID_training, batch_size=8000, epochs = 3)

pred = classifier.predict(ID_testing)
pred = (pred > 0.5)
# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ID_testing.classes, pred)

print(cm)