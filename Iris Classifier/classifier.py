from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris dataset and split into training and testing sets
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = SVC(gamma='auto')
clf.fit(x_train, y_train)


def predict_labels(y_test):
    label_pred = clf.predict([y_test])  # Modify to predict labels for a single sample
    return iris.target_names[label_pred[0]]
