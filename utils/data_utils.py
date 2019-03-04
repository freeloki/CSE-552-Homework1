import pandas as pd

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


#test_data = load_data()

#group = test_data.groupby("Class")
#list_temp = []

# print(test_data["Class"].max())


def split_training_and_testing_data(data, split_percentage=0.8):
    training_data = []
    testing_data = []
    for classes, classes_region in data.groupby("Class"):
        #print(int(len(classes_region)))
        split_index = int(len(classes_region) * split_percentage)
        #print(classes_region[:split_index])
        training_data.append(classes_region[:split_index])
        #print("----------------")
        testing_data.append(classes_region[split_index:])
        #print(classes_region[split_index+1:])
        #print(split_index)
        #print("----------------")
        

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


