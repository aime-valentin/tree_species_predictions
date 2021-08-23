

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
import joblib

#important functions for data cleaning
def calculate_NDVI(df):
    #takes a dataset of reflectance values
    #calculates NDVI using the following formula:
    #tRef.NDVI = (tRef.x800_67 - tRef.x679_92) ./ (tRef.x800_67 + tRef.x679_92)
    #adds the NDVI column on the dataset
    #filters out all the pixels whose NDVI < 0.4 
    #drops NDVI column
    #returns transformed df

    if ('800.67' in df.columns) and ('679.92' in df.columns):
        df['NDVI'] = df.apply(lambda x: (x['800.67'] - x['679.92'])/(x['800.67'] + x['679.92']), axis = 1)#calculate NDVI
        df = df[df['NDVI']>=0.4] #filter out all those pixels with NDVI <0.4 
        df.drop('NDVI', axis = 1, inplace = True) #drops NDVI column from the dataframe
        return df
    return df
def tree_id_mapping(df1, df2):
    #df1: dataframe of tree species
    #df2: dataframe of reflectance values
    #Creates a dictionary keys=TreeID, values =Tree species
    #Maps dictionary on df2, adding column named target
    #returns transformed df2
    keys = dict(zip(df1.OBJECTID, df1.SPECIES)) #create key,value pair for OBJECTID and SPECIES
    df2['TARGET'] = df2.ROIID.apply(lambda x: keys[x] if x in keys else float('NaN')) #adds species name in reflectance
    df2.dropna(inplace = True)
    return df2
     

def shade_filter_bands(df):
    #takes in dataframe of reflectance values
    #filters out those pixels with zero values
    #filters out those pixels whose reflectance at 800 nm < 0.15 (brighness filter)
    #selects only 400 nm - 900 nm bands
    #returns transformed df
    
    df = df[list(df.columns[:2]) + list(df.columns[8:])] #selects ROIID and wavelengths only
                                                         #skips map and longitude data
    #remove any row with zero reflectance at any wavelength
    for i in df.columns[2:]:
        df = df[df[i] != 0]

    if ('800.67' in df.columns) and ('900.53' in df.columns):
        #remove pixels whose reflectnace at 800 nm <0.15
        df = df[df['800.67'] >= 0.15]
        #selects wavelengths 400nm - 900 nm
        cols = list(df.columns)
        index_900 = cols.index('900.53')
        df = df[df.columns[0:index_900 + 1]]
        #print(df.columns)
        return df
    return df

#wrapper method
def data_cleaning(df_tree, df_reflectance):
    df = shade_filter_bands(df_reflectance)
    df = calculate_NDVI(df)
    df = tree_id_mapping(df_tree,df)
    #df = df.groupby(['ROIID','TARGET'], as_index = False).mean()
    return df

def build_model(X,y,model):
    #df = df.groupby(['ROIID','TARGET'], as_index = False).mean()
    #df['target'] = LabelEncoder().fit_transform(df["TARGET"])
    #X = df[df.columns[1:-2]].values #features
    #y = df['target'].values #target
    #70:30 split for training and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    #model training
    #model = RandomForestClassifier(n_estimators=500, verbose = 10)
    model.fit(X,y)
    predictions = model.predict(X_test)
    return (X_test,y_test, predictions)
    #return (model)
    
def generate_cf(model_name,y_test, preds,nclasses, labels):
    #evaluate the model
    #create confusion matrix
    #print(classification_report(y_test,preds))
    cf1 = confusion_matrix(y_test, preds)
    cf1_ = []
    for i in range(nclasses): 
        cf1_.append(cf1[i][:]/cf1.sum(axis=1)[i])
    cf1_ = np.array(cf1_)  

    #plot confusion matrix
    sns.set_context('talk')
    plt.figure(figsize = (15,10))

    sns.heatmap(cf1_,annot=True, xticklabels = labels,yticklabels= labels,
               cmap = "YlGnBu", cbar = False, fmt = '.2%')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig(f'confusion_matrices/{model_name}.png')

def evaluate_model(model,model_name,x_train,y_train,x_test, y_test, nclasses, labels):
    train_preds = model.predict(x_train)
    preds = model.predict(x_test)
    train_accuracy = 100*(accuracy_score(train_preds,y_train))
    test_accuracy = 100*(accuracy_score(y_test, preds))
    generate_cf(model_name,y_test,preds,nclasses,labels)
    return {"Model":[model_name], "Train Accuracy(%)":[train_accuracy],
            "Test Accuracy(%)":[test_accuracy]}

def calculate_pcs(df,x, n_components):
    pca = PCA(n_components= n_components)
    pca_= pca.fit_transform(x)
    princ_df = pd.DataFrame(data = pca_
                 , columns = ["PC"+str(i) for i in list(range(1,n_components+1))])
    princ_df['target'] = list(df.TARGET) #assumes that df has 'TARGET' predictor
    print(princ_df.tail())
    return(princ_df)

def merge_images(array):
    '''combines images and assumes that array is [df_aug, df_sep, df_oct'''
    #combines images into one image
    if len(array) == 3:
        #column-wise left-join of the three datasets
        #join August and September images by extracting common pixels (intersections)
        df_temp = array[0].join(array[1].set_index(['ROIID','IDwithROI','TARGET']), on = ['ROIID','IDwithROI','TARGET'],
                            how ='inner', lsuffix ='_aug', rsuffix ='_sep')
        #join August, September, and October images by extracting common pixels (intersections)
        df = df_temp.join(array[2].set_index(['ROIID','IDwithROI','TARGET']), on = ['ROIID','IDwithROI','TARGET'],
                            how ='inner', lsuffix ='', rsuffix ='_oct')
        df = df.groupby(['ROIID','IDwithROI','TARGET'], as_index = False).last()
    elif len(array)==2:
        #join only two images
        df= array[0].join(array[1].set_index(['ROIID','IDwithROI','TARGET']), on = ['ROIID','IDwithROI','TARGET'],
                            how ='inner', lsuffix ='_aug', rsuffix ='_sep')
    else:
        return "please provide accurate dataframes"
    return df

#extract only RGB NI wavelengths
def extract_rgbi(df, new_cols):
    #cols = [float(i) for i in list(df.columns)[8:]]
    #red = [i  for i in cols if 640<=i<=660]
    #blue = [i for i in cols if 440<=i<=460]
    #green = [i for i in cols if 540<=i<=560]
    #nir = [i for i in cols if 840<=i<=850]
    #new_cols = red + blue + green +nir
    cols_ = list(df.columns)[0:8] + [str(i) for i in new_cols]
    df = df[cols_]
    return df

def feature_imp(cols, feats_imp):
    df = pd.DataFrame({"Features":cols,"importances":feats_imp}
                      ).sort_values('importances',ascending = False).reset_index(drop = True)
    return df

def save_model(model, model_name):
    joblib.dump(model, f'models/{model_name}.joblib')

def make_pc_plot(pc_plot):
    var = pc_plot[0]
    cum_var  = np.cumsum(var)
    cols = pc_plot[1]
    
    plt.figure(figsize=(8,6))
    sns.set_context("talk")
    sns.barplot(x =cols, y=var, color = 'grey')
    plt.ylabel("Explained Variance")
    plt.xlabel("Principal Components")
    plt.plot(cum_var)