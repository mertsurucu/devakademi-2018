from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

# loads the data from npy
data_features = np.load('Dataset//features.npy')
data_label = np.load('Dataset//labels.npy')

# encode the string features into int type
# and reshape the data 1D to 2D with 6 features in each row
le = preprocessing.LabelEncoder()
data_features = le.fit_transform(data_features)
data_features = np.reshape(data_features, (-1, 6))

count = 0
classes = []
i = 0
while i < len(data_label):
    if data_label[i] not in classes:
        classes.append(data_label[i])
        count += 1
    i += 1
print(count, "Ad Categories")
print("Ad Category Names: ", classes)
generalOverallAccuracy = 0

th_fold = 0
# K-fold splits into 10 and shuffles the indexes
split_size = 10
kf = KFold(n_splits=split_size, shuffle=True)
kf.get_n_splits(data_features)
for train_index, val_index in kf.split(data_features):
    th_fold += 1
    # print("TRAIN:", train_index, "VAL:", val_index)
    train_data, val_data = data_features[train_index], data_features[val_index]
    train_label, val_label = data_label[train_index], data_label[val_index]

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_estimators=10)
    # Train the Classifier to take the training features and learn how they relate to the training(the species)
    clf.fit(train_data, train_label)
    # makes a list to get each accuracy
    prediction = clf.predict(val_data)
    accuracy_counter = 0
    for each in range(len(prediction)):
        if prediction[each] == val_label[each]:
            accuracy_counter += 1
    generalOverallAccuracy += accuracy_counter
    print("Accuracy for", th_fold, ". fold", accuracy_counter / len(prediction))
print("Overall Accuracy", generalOverallAccuracy / len(data_label))