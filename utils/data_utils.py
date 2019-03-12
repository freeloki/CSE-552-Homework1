import pandas as pd
import numpy as np
import math
import operator
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn import metrics, svm

# data_path
DATA_PATH = "../dataset/leaf/leaf.csv"

columns = ["Class",
           "Specimen",
           "Eccentricity",
           "Aspect Ratio",
           "Elongation",
           "Solidity",
           "Stochastic Convexity",
           "Isometric Factor",
           "Maximal Indentation Depth",
           "Lobedness",
           "Average Contrast",
           "Smoothness",
           "Third Moment",
           "Uniformity",
           "Entropy"]


def load_data(path=DATA_PATH):
    data = pd.read_csv(path, index_col=False, header=None, names=columns)
    return data


# test_data = load_data()

# group = test_data.groupby("Class")
# list_temp = []

# print(test_data["Class"].max())


def split_training_and_testing_data(data, split_percentage=0.8):
    training_data = []
    testing_data = []
    for classes, classes_region in data.groupby("Class"):
        # print(int(len(classes_region)))
        split_index = int(len(classes_region) * split_percentage)
        # print(classes_region[:split_index])
        training_data.append(classes_region[:split_index])
        # print("----------------")
        testing_data.append(classes_region[split_index:])
        # print(classes_region[split_index+1:])
        # print(split_index)
        # print("----------------")

    return training_data, testing_data


"""training_data, testing_data = split_training_and_testing_data(load_data())

total_train = 0
for x in range(0,len(training_data)):
    total_train += len(training_data[x])

print("Train: ", total_train )
total_train = 0
for x in range(0,len(testing_data)):
    total_train += len(testing_data[x])

print("Test: ", total_train )

print(len(training_data[29]))
print("----------------")
print(len(testing_data[29]))

print(len(training_data[28]))
print("----------------")
print(len(testing_data[28]))

print(testing_data[0].head())"""


def euclidean_distance(data1, data2, length):
    distance = 0
    # ignore first data because it is class label.
    # print("Euclidean")
    # print(data1)
    # print(data2)
    # print(length)
    # print("Euclidean")
    for x in range(2, length):
        distance += np.square(data1[x] - data2[x])

    # print("Euclidean Result: " + str(distance))
    return np.sqrt(distance)


def k_nearest_neighbourhood(training_set, test_instance, k):
    distances = {}
    sort = {}

    length = len(test_instance)

    # print("Test Instance Shape", test_instance.shape, len(test_instance))

    # print("Training Set Length", len(training_set))

    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(training_set)):
        #### Start of STEP 3.1
        dist = euclidean_distance(test_instance, training_set[x], length)

        distances[x] = dist
        #### End of STEP 3.1

    # print(len(distances))

    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    # print("Distance Items:\n", distances.items())
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2

    # print("Sorted Distances ")

    # print(sorted_d)
    # print(sorted_d[0][0])
    neighbors = []

    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}

    # print(neighbors)
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = training_set[neighbors[x]][0]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5

    # print(classVotes)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)
    #### End of STEP 3.5


training_data, testing_data = split_training_and_testing_data(load_data(DATA_PATH))

total_train = 0
for x in range(0, len(training_data)):
    total_train += len(training_data[x])

print("Train: ", total_train)
total_test = 0
for x in range(0, len(testing_data)):
    total_test += len(testing_data[x])
print("Test: ", total_test)

train_numpy = np.asarray(training_data[0])
test_numpy = np.asarray(testing_data[0])
for x in range(1, len(training_data)):
    train_numpy = np.concatenate((train_numpy, np.asarray(training_data[x])), axis=0)

for x in range(1, len(testing_data)):
    test_numpy = np.concatenate((test_numpy, np.asarray(testing_data[x])), axis=0)

# print(test_numpy)
print(train_numpy.shape)
print(test_numpy.shape)

print(test_numpy[0])
print(train_numpy[0])

"""test_index = 67;
result, neigbours = k_nearest_neighbourhood(train_numpy, test_numpy[test_index], 5)
print("Expected Class: " + str(int(test_numpy[test_index][0])))
print("Predicted Class: ", int(result))
print("Nearest Neighbours: ", neigbours)"""

# print(train_numpy[neigbours])
# print(test_numpy[0])


# normalize data:


positive = 0
negative = 0
print(len(test_numpy[0]))

"""for test_k in range(1, 13, 2):
    positive = 0
    negative = 0
    for x in range(0, len(test_numpy)):
        sorted2, neigbours2 = k_nearest_neighbourhood(train_numpy, test_numpy[x], test_k)
        if int(test_numpy[x][0]) == int(sorted2):
            positive += 1
        else:
            negative += 1

    print("Without Norm Test K : " + str(test_k) + "  Positive:" + str(positive))
    print("Without Norm Test K : " + str(test_k) + "  Negative:" + str(negative))
"""
"""for x in range(0, len(test_numpy), 1):
    temp_point = test_numpy[x]
    # print("BEFORE_NORM")
    # print(temp_point)
    total_sum = sum(temp_point) - test_numpy[x][0] - test_numpy[x][1]
    for y in range(2, len(temp_point)):
        temp_point[y] = temp_point[y] / total_sum

    # print("AFTER_NORM")
    # print(temp_point)
    test_numpy[x] = temp_point

for x in range(0, len(train_numpy), 1):
    temp_point = train_numpy[x]
    # print("BEFORE_NORM")
    # print(temp_point)
    total_sum = sum(temp_point) - train_numpy[x][0] - train_numpy[x][1]
    for y in range(2, len(temp_point)):
        temp_point[y] = temp_point[y] / total_sum

    # print("AFTER_NORM")
    # print(temp_point)
    train_numpy[x] = temp_point
"""
# print("K : " + str(k_size) + " True:" + str(positive) + " False: " + str(negative) + "  Negative Accuracy Rate % " + str(
# (100 * negative) / (positive + negative)))

split_size = 5
kf = KFold(n_splits=split_size, random_state=False, shuffle=True)

splits = kf.get_n_splits(train_numpy)

print("Splits: " + str(splits))

"""for k_size in range(1, 15, 2):
    mean_pos = 0
    for train_index, test_index in kf.split(train_numpy):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_numpy[train_index], train_numpy[test_index]

        positive = 0
        negative = 0
        for x in range(0, len(X_test)):
            sorted2, neigbours2 = k_nearest_neighbourhood(X_train, X_test[x], k_size)
            if int(X_test[x][0]) == int(sorted2):
                positive += 1
            else:
                negative += 1

        temp_acc = (100 * positive) / (positive + negative)
        mean_pos += temp_acc
        print("TrainSize: " + str(len(X_train)) + " TestSize:" + str(len(X_test)))
    # print("K : " + str(k_size) + " True:" + str(positive) + " False: " + str(negative) + "  Positive Accuracy Rate % " + str(
    # (100 * positive) / (positive + negative)))
    # print("K : " + str(k_size) + " True:" + str(positive) + " False: " + str(negative) + "  Negative Accuracy Rate % " + str(
    # (100 * negative) / (positive + negative)))

    print("K SIZE : " + str(k_size) + " MEAN ACC  % " + str(mean_pos / split_size))


for test_k in range(1, 15, 2):
    positive = 0
    negative = 0
    for x in range(0, len(test_numpy)):
        sorted2, neigbours2 = k_nearest_neighbourhood(train_numpy, test_numpy[x], test_k)
        if int(test_numpy[x][0]) == int(sorted2):
            positive += 1
        else:
            negative += 1
    print("TEST DATA K : " + str(test_k) + " True:" + str(positive) + " False: " + str(negative) + "  Positive Accuracy Rate % " + str(
     (100 * positive) / (positive + negative)))


scores = cross_val_score(k_nearest_neighbourhood, train_numpy, train_numpy[0], cv=5, scoring='f1_macro')"""

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
"""

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm"""
knn = KNeighborsClassifier(n_neighbors=1)
# Fit the classifier to the data

label_x = np.empty(257)
test_label_x = np.empty(83)
for x in range(0, len(train_numpy), 1):
    label_x[x] = train_numpy[x][0]

for x in range(0, len(test_numpy), 1):
    test_label_x[x] = test_numpy[x][0]

awesome = knn.fit(train_numpy, label_x)

param_grid = {"n_neighbors": np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
# fit model to data
knn_gscv.fit(train_numpy, label_x)
print(knn.score(test_numpy, test_label_x))


# print(knn_gscv.best_estimator_)
# print(knn_gscv.best_params_)


def manhattan_distance(data1, data2):
    if (len(data1) != len(data2)):
        print("Be sure that both vectors are the same dimension!")
        return

    return sum([abs(data1[i] - data2[i]) for i in range(len(data1))])


def k_nearest_neighbourhood_manhattan(training_set, test_instance, k):
    distances = {}
    sort = {}

    length = len(test_instance)

    # print("Test Instance Shape", test_instance.shape, len(test_instance))

    # print("Training Set Length", len(training_set))

    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(training_set)):
        #### Start of STEP 3.1
        dist = manhattan_distance(test_instance, training_set[x])

        distances[x] = dist
        #### End of STEP 3.1

    # print(len(distances))

    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    # print("Distance Items:\n", distances.items())
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2

    # print("Sorted Distances ")

    # print(sorted_d)
    # print(sorted_d[0][0])
    neighbors = []

    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}

    # print(neighbors)
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = training_set[neighbors[x]][0]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5

    # print(classVotes)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)
    #### End of STEP 3.5


"""
for test_k in range(1, 13, 2):
    positive = 0
    negative = 0
    for x in range(0, len(test_numpy)):
        sorted2, neigbours2 = k_nearest_neighbourhood_manhattan(train_numpy, test_numpy[x], test_k)
        if int(test_numpy[x][0]) == int(sorted2):
            positive += 1
        else:
            negative += 1

    print("Without Norm Test K : " + str(test_k) + "  Positive:" + str(positive))
    print("Without Norm Test K : " + str(test_k) + "  Negative:" + str(negative))
for k_size in range(1, 15, 2):
    mean_pos = 0
    for train_index, test_index in kf.split(train_numpy):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_numpy[train_index], train_numpy[test_index]

        positive = 0
        negative = 0
        for x in range(0, len(X_test)):
            sorted2, neigbours2 = k_nearest_neighbourhood_manhattan(X_train, X_test[x], k_size)
            if int(X_test[x][0]) == int(sorted2):
                positive += 1
            else:
                negative += 1

        temp_acc = (100 * positive) / (positive + negative)
        mean_pos += temp_acc
        print("TrainSize: " + str(len(X_train)) + " TestSize:" + str(len(X_test)))
    # print("K : " + str(k_size) + " True:" + str(positive) + " False: " + str(negative) + "  Positive Accuracy Rate % " + str(
    # (100 * positive) / (positive + negative)))
    # print("K : " + str(k_size) + " True:" + str(positive) + " False: " + str(negative) + "  Negative Accuracy Rate % " + str(
    # (100 * negative) / (positive + negative)))

    print("K SIZE : " + str(k_size) + " MEAN ACC  % " + str(mean_pos / split_size))


for test_k in range(1, 15, 2):
    positive = 0
    negative = 0
    for x in range(0, len(test_numpy)):
        sorted2, neigbours2 = k_nearest_neighbourhood_manhattan(train_numpy, test_numpy[x], test_k)
        if int(test_numpy[x][0]) == int(sorted2):
            positive += 1
        else:
            negative += 1
    print("TEST DATA K : " + str(test_k) + " True:" + str(positive) + " False: " + str(negative) + "  Positive Accuracy Rate % " + str(
     (100 * positive) / (positive + negative)))

"""

############# SVM STARTS HERE #######################


train_labels = []
test_labels = []


def delete_first_two_column(data_array):
    label_array = np.empty(len(data_array))
    for index_label in range(0, len(data_array), 1):
        print("Doing:" + str(index_label))
        print("DATA: " + str(data_array[index_label][0]))
        label_array[index_label] = data_array[index_label][0]

    data_array = np.delete(data_array, np.s_[0:2], axis=1)

    return data_array, label_array


print(train_numpy[0][0])

print(len(train_numpy))
train_numpy, train_labels = delete_first_two_column(train_numpy)
test_numpy, test_labels = delete_first_two_column(test_numpy)

print(train_numpy[0])
print(train_labels[0])

print(test_numpy[80])
print(test_labels[80])

clf = svm.LinearSVC(penalty='l2', verbose=0, max_iter=10000, class_weight=None)

clf.fit(train_numpy, train_labels, sample_weight=None)

# print(str(train_numpy))
print(str(test_labels))

print("###---- WHOLE TRAIN DATA START ----####")
print(clf.score(train_numpy, train_labels))
print("###---- WHOLE TRAIN DATA END  ----####")

print("###---- WHOLE TEST DATA START ----####")
print(clf.score(test_numpy, test_labels))
print("###---- WHOLE TEST DATA END  ----####")

average_train = 0
average_test = 0
for train_index, test_index in kf.split(train_numpy):
    print("####################333")

    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_numpy[train_index], train_numpy[test_index]
    X_train_labels, X_test_labels = train_labels[train_index], train_labels[test_index]
    clf.fit(X_train, X_train_labels)
    average_train += clf.score(X_train,X_train_labels)
    average_test += clf.score(X_test,X_test_labels)
    print(clf.score(X_train, X_train_labels))
    print(clf.score(X_test, X_test_labels))


print("AVERAGE RESULTS")
print(str(average_train/5))
print(str(average_test/5))