import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import decomposition

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from helper_functions_genus import *

class ModelBuilder: 

    def __init__(self,data, df_tree):
        print('data cleaning in progress ...\n')
        self.image = data_cleaning(df_tree,data)
        print('data cleaning completed.')
        print(self.image.head(5))

    def split_data(self):
        X = self.image[self.image.columns[2:-1]].values
        #(ACSA:0,CECA:1,FAPE:2,PIAB:3,PINI:4,PIPO:5,PIPU:6,PISY:7,QUBI:8,QUMA:9,QURU:10)
        y = LabelEncoder().fit_transform(self.image.TARGET)
        #70:30 split for training and validation data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
        return (X_train,X_test, y_train,y_test)

    def build_model(self,X_train, y_train):
        #model training
        print('model building in progress ...')
        model = RandomForestClassifier(n_estimators=100,oob_score = True, verbose =1, n_jobs = -1)
        model.fit(X_train, y_train)
        print('model building completed.')
        return model
        
    def create_confusion_matrix(self,model_name,y_test, preds,nclasses, labels):
        #evaluate the model
        #create confusion matrix
        #print(classification_report(y_test,preds))
        cf1 = confusion_matrix(y_test, preds)
        cf1_ = []
        for i in range(nclasses): 
            cf1_.append(cf1[i][:]/cf1.sum(axis=1)[i])
        cf1_ = np.array(cf1_)  

        #plot confusion matrix
        sns.set_context('talk', font_scale=0.8)
        #labels = ['Ash', "Pines", "Spruce", 'Oaks']
        plt.figure(figsize = (12,7))

        sns.heatmap(cf1_,annot=True, xticklabels = labels,yticklabels= labels,
                cmap = "YlGnBu", cbar = False, fmt = '.2%')
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig(f'confusion_matrices/{model_name}.png')

    def grid_search(self, param_grid, x_train, y_train ):
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3,n_jobs = -1,verbose = 5)
        grid_search.fit(x_train,y_train)
        return (grid_search, grid_search.best_params_)

    def random_search(self,random_grid,x_train,y_train):
        rf = RandomForestClassifier()
        random_search = RandomizedSearchCV(rf, random_grid,n_iter = 100, cv = 3, verbose=5, n_jobs =-1)
        random_search.fit(x_train,y_train)
        return(random_search, random_search.best_params_)
    
    def build_pca_model(self,n_components):
        X = self.image[self.image.columns[2:-1]].values
        pca = PCA(n_components=n_components)
        pca_ = pca.fit(X)
        new_x = pca_.transform(X)

        princ_df = pd.DataFrame(data = new_x
             , columns = ["PC"+str(i) for i in list(range(1,n_components+1))])
        princ_df['target'] = list(self.image.TARGET)
        princ_df.tail()

        #train_test_split
        X = princ_df[princ_df.columns[:-1]].values
        y = LabelEncoder().fit_transform(princ_df.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
        return (X_train, X_test,y_train,y_test,(pca_.explained_variance_ratio_,list(princ_df.columns)[:-1]))
    

    
if __name__ == '__main__':
    aug = pd.read_csv("../data/sample.csv")
    df_tree = pd.read_csv("../data/Export_Output_2_0630.txt")
    
    #init method
    mb = ModelBuilder(aug, df_tree)
    print(mb.image.head(3))

    #get_date method
    x_train,x_test, y_train,y_test = mb.split_data()
    print(x_train)

    #build_model
    model, preds = mb.build_model(x_train,x_test, y_train)

    # #evaluate the model
    # print(mb.evaluate_model(model,"Base Model",x_train, y_train, x_test,y_test))

    #create confusion matrix
    labels = ['Sugar Maple', "Eastern Redbud", "Green Ash", 'Norway Spruce','Austrian Pine','Ponderosa Pine',
          'Colorado Spruce', 'Scotch Pine','White Oak','Bur Oak', 'Red Oak']
    mb.create_confusion_matrix(y_test,preds,11,labels)

    #tuning hyperparameters
    #random_search
    #number of trees
    n_estimators = [100,200,300]
    #number of features to consider at split
    max_features = ['auto','sqrt']
    #maximum numbero f levels in tree
    max_depth = [int(x) for x in np.linspace(10,50, num= 11)]
    #min number to split node
    min_node = [2,6,8,10]
    #min number of sample per leaf node
    min_sample = [1,3,5]
    #boostrap
    # bstrap = [True, False] 
    # param_grid ={'n_estimators':n_estimators,
    #              'max_features':max_features,
    #              'max_depth':max_depth,
    #              'min_samples_split':min_node,
    #              'min_samples_leaf':min_sample,
    #              'bootstrap':bstrap}
    # rf_random,best_params = mb.random_search(param_grid,x_train,y_train)
    # print(mb.evaluate_model(rf_random,"Model After RandomCV Search",x_train, y_train, x_test,y_test))
    # print(best_params)

    #gridsearch from randomsearch resutls
    n_estimators  = [50,75,100]
    max_features = ['auto']
    max_depth = [20,25,30]
    min_split_node = [2,4]
    min_sample_leaf = [1,3]
    bstrap  = [False]

    param_grid_ ={'n_estimators':n_estimators,
                'max_features':max_features,
                'max_depth':max_depth,
                'min_samples_split':min_split_node,
                'min_samples_leaf':min_sample_leaf,
                'bootstrap':bstrap}

    rf_grid,best_params = mb.grid_search(param_grid_,x_train,y_train)
    print(evaluate_model(rf_grid,"Model After GridCV Search",x_train, y_train, x_test,y_test))
    print(best_params)

    preds_ = rf_grid.predict(x_test)
    mb.create_confusion_matrix(y_test,preds_,11,labels)

    #try principal component analysis
    x_train_,x_test_,y_train_,y_test_ = mb.build_pca_model(10)
    rf_grid_pc,best_params = mb.grid_search(param_grid_,x_train_,y_train_)
    print(evaluate_model(rf_grid_pc,"Model GridSearch on PC",x_train_, y_train_, x_test_,y_test_))
    print(best_params)

    preds_ = rf_grid_pc.predict(x_test_)
    mb.create_confusion_matrix(y_test_,preds_,11,labels)
   




