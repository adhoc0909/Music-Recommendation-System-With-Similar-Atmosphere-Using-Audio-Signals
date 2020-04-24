from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics 
from sklearn.model_selection import train_test_split

def RandomForest(X_train, X_test, y_train, y_test):
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print('총 {}곡 중 {:.2f}% 정확도로 장르를 맞춤'.format(y_test.shape[0], 100 * metrics.accuracy_score(prediction, y_test)))
    
    return model