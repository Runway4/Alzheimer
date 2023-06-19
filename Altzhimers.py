from PIL import Image
from numpy import asarray

import numpy as np
#load the image and convert into
#numpy array
import glob
import numpy as np
from sklearn.model_selection import train_test_split
for multiplier in range(1,31):
    y = []
    cv_img = []
    Cn = 0
    for img in glob.glob(r'C:\Users\brani\OneDrive\Desktop\Alzheimer_s\train\MildDemented\*.jpg'):
        y.append(2)
        img1 = Image.open(img)
        # Added code
        img1 = img1.resize((10*multiplier, 10*multiplier))
        reshaped_image=np.transpose(img1)

        # reshape image being weights are diffrent
        x1 = np.array(img1)
        x = x1.flatten()
        cv_img.append(x)
        Cn = Cn+1
    for img in glob.glob(r'C:\Users\brani\OneDrive\Desktop\Alzheimer_s\train\ModerateDemented\*.jpg'):
        y.append(3)
        img1 = Image.open(img)
        # Added code
        img1 = img1.resize((10*multiplier, 10*multiplier))
        reshaped_image=np.transpose(img1)

        # reshape image being weights are diffrent
        x1 = np.array(img1)
        x = x1.flatten()
        cv_img.append(x)
        Cn = Cn + 1
    for img in glob.glob(r'C:\Users\brani\OneDrive\Desktop\Alzheimer_s\train\NonDemented\*.jpg'):
        y.append(0)
        img1 = Image.open(img)
        # Added code
        img1 = img1.resize((10*multiplier, 10*multiplier))
        reshaped_image = np.transpose(img1)

        # reshape image being weights are diffrent
        x1 = np.array(img1)
        x = x1.flatten()
        cv_img.append(x)
        Cn = Cn + 1
    for img in glob.glob(r'C:\Users\brani\OneDrive\Desktop\Alzheimer_s\train\VeryMildDemented\*.jpg'):
        y.append(1)
        img1 = Image.open(img)
        # Added code
        img1 = img1.resize((10*multiplier, 10*multiplier))
        reshaped_image = np.transpose(img1)

        # reshape image being weights are diffrent
        x1 = np.array(img1)
        x = x1.flatten()
        cv_img.append(x)
        Cn = Cn + 1
    X_train, X_test, y_train, y_test = train_test_split(cv_img, y)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=5000).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    Predictions = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test) * 100
    print(f"Test Accuracy Neural Network #1: {accuracy:.2f}%")
    accuracy = clf.score(X_train, y_train) * 100
    print(f"Train Accuracy Neural Network #1: {accuracy:.2f}%")

    from sklearn.neural_network import MLPClassifier

    clfdt = MLPClassifier(random_state=1, max_iter=5000).fit(X_train, y_train)
    clfdt.predict_proba(X_test[:1])
    Predictions_dt = clfdt.predict(X_test)
    accuracy_dt = clfdt.score(X_test, y_test) * 100
    print(f"Test Accuracy Neural Network #2: {accuracy_dt:.2f}%")
    accuracy_dt = clfdt.score(X_train, y_train) * 100
    print(f"Train Accuracy Neural Network #2: {accuracy_dt:.2f}%")


    from sklearn.linear_model import LogisticRegression
    clfd = LogisticRegression(random_state=1).fit(X_train, y_train)
    predictions = clfd.predict(X_test)
    predictions_t = clfd.predict(X_train)

    # compute the percentage of correct predictions
    correct_predictions = sum(predictions == y_test)
    total_predictions = len(y_test)
    percentage_correct = correct_predictions / total_predictions * 100
    print(f"Test Accuracy Liniar Model: {percentage_correct:.2f}%")

    correct_predictions_t = sum(predictions_t == y_train)
    total_predictions_t = len(y_train)
    percentage_correct = correct_predictions_t / total_predictions_t * 100
    print(f"Train Accuracy Liniar Model: {percentage_correct:.2f}%")

    corect_predictions_both = 0
    for i in range(len(X_test)):
        clf_pred = predictions[i]
        clfd_pred = Predictions[i]
        if clf_pred == y_test[i]:
            if clfd_pred == y_test[i]:
                corect_predictions_both +=1
    corect_predictions_one = 0
    for i in range(len(X_test)):
        clf_predx = predictions[i]
        clfd_predx = Predictions[i]
        if clf_predx == y_test[i]:
            corect_predictions_one +=1
        elif clfd_predx == y_test[i]:
            corect_predictions_one +=1
    print("both corect" ,corect_predictions_both / len(X_test))
    print("atlest one corect" ,corect_predictions_one/ len(X_test))

    clfd3 = LogisticRegression(random_state=1).fit(X_train, y_train)
    predictions3 = clfd3.predict(X_test)
    correct_predictions_both = 0
    for i in range(len(X_test)):
        clf_pred = predictions[i]
        clfd_pred = Predictions[i]
        if clf_pred == y_test[i] and clfd_pred == y_test[i]:
            correct_predictions_both += 1

    print("Both models correct: ", correct_predictions_both / len(X_test))
    three_models_agree = 0
    for i in range(len(X_test)):
        if Predictions[i] == Predictions_dt[i] and Predictions[i] == predictions[i]:
            three_models_agree +=1
    print("all corect", three_models_agree / len(X_test) * 100)

    w1= 1/3
    w2= 1/3
    w3= 1/3
    for i in range(len(y_test)):
        if y_test[i] != Predictions[i]:
            w1 = w1/2
        if y_test[i] != predictions[i]:
            w2 = w2/2
        if y_test[i] != Predictions_dt[i]:
            w3 = w3/2
        sum = w1 + w2 +w3
        w1= sum/w1
        w2= sum/w2
        w3= sum/w3
        print("w1=",w1)
        print("w2=",w2)
        print("w3=",w3)