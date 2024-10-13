from .Preprocessing import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score

class DecisionTreeClassification(Preprocessing):

    def __init__(self):
        Preprocessing.__init__(self)
        self._hypertune = False
    
    def Classify(self, hypertune = False):

        if(hypertune is False):

            self._classifier = DecisionTreeClassifier(random_state=0)
            self._classifier.fit(self._x_train, self._y_train)
            return

        parameters = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2,4,6,8]
        }

        if(self._hypertune is False):

            dt_class = DecisionTreeClassifier(random_state=0)
            dt_class = GridSearchCV(dt_class, param_grid = parameters, scoring=('accuracy'), cv=5)
            dt_class.fit(self._x_train,self._y_train)
            self._best_param = list(dt_class.best_params_.values())

        self._classifier = DecisionTreeClassifier(criterion=self._best_param[0], max_depth=self._best_param[1])
        self._classifier.fit(self._x_train, self._y_train)
        self._hypertune = True
    
    def GetBestParam(self):
        if (self._hypertune == True):
            return self._best_param
    
    def Evaluate(self, target_names:list, show_curve = False):
        y_pred = self._classifier.predict(self._x_test)
        print('\nClassification Report\n')
        print(classification_report(self._y_test, y_pred, target_names = target_names))

        if(show_curve is True):

            fpr, tpr, thresholds = roc_curve(self._y_test, y_pred)
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            print(f'AUC - ROC Score for the model: {roc_auc_score(self._y_test, y_pred)}')
            plt.show()

    def Predict(self, features:list):
        return self._classifier.predict(features)

class SVClassification(Preprocessing):

    def __init__(self):
        Preprocessing.__init__(self)
    
    def Classify(self, hypertune=False):

        if(hypertune is False):
            self._classifier = make_pipeline(SVC(random_state=0))
            self._classifier.fit(self._x_train, self._y_train)
            return

        parameters = {
            'C':[1,10,100],
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma':['scale', 'auto']
        }

        svm_class = make_pipeline(SVC(random_state=0))
        svm_class = GridSearchCV(svm_class, param_grid = parameters, scoring=('accuracy'), cv=5)
        svm_class.fit(self._x_train,self._y_train)
        self._best_param = list(svm_class.best_params_.values())

        self._classifier = make_pipeline(SVC(random_state=0, C=self._best_param[0], kernel=self._best_param[1], gamma=self._best_param[2]))
        self._classifier.fit(self._x_train, self._y_train)
        self._hypertune = True
    
    def Evaluate(self, target_names:list, show_curve = False):
        y_pred = self._classifier.predict(self._x_test)
        print('\nClassification Report\n')
        print(classification_report(self._y_test, y_pred, target_names = target_names))

        if(show_curve is True):

            fpr, tpr, thresholds = roc_curve(self._y_test, y_pred)
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            print(f'AUC - ROC Score for the model: {roc_auc_score(self._y_test, y_pred)}')
            plt.show()
    
    def Predict(self, features:list):
        return self._classifier.predict(features)