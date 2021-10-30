# MachineLearningPhw1-2
This program is machine Learning assignment programming homework 1, 2

# PHW-1
phw-1 file is find the best model and best parameters and best scalers and best encoder

The dataset is breast-cancer dataset 
Id and feature1~9 are feature attributes, target is target value (categorical column)
Some nan value exists in feature 6 (ex)…?)->So, feature6 is not integer object type.
16 values are ‘?’ So, we replace ‘?’ to np.nan value. (Because its’s trash data)

<img width="230" alt="캡처2" src="https://user-images.githubusercontent.com/74089524/139522849-8df46b3b-9827-4a8f-8a48-e426dadb9caa.PNG">

After preprocessing

<img width="252" alt="캡처1" src="https://user-images.githubusercontent.com/74089524/139522848-c5d9c97e-7e8a-4685-8bc1-ade97d447e04.PNG">

 I use model: decision Tree with gini index, entrophy, support vector machine, logistic regression and normalizer, standard scaler, maxabs scaler, robust scaler, min max scaler, label encoder, ordinal encoder
 
 First, i made the function with each model
 
 like this:
 def decisionTreeWithGini(x,y):
 x is the features and y is the target
 
 The whol structure is 
 def bestOfbest(x,y):
 
 1. we put the feature and target
 2. they divide train and test dataset
 3.  they execute at each model and they measure the score of best model 
 4.  they tried the all scaler and encoder combination and return the best combination

Output:

<img width="465" alt="캡처3" src="https://user-images.githubusercontent.com/74089524/139522958-44762c9c-282e-41d7-b917-73a4682b8550.PNG">

# PHW-2

def auto_ml(dataset, model)

-	Select features randomly.
-	Run algorithms with the selected combination.

●	Parameter
dataset: DataFrame to be used
model: A model to be used

●	Examples
df = pd.read_csv(‘housing.csv’)

df.fillna(df.mean(), inplace=True)

medianHouseValue = df['median_house_value']
df.drop(['median_house_value'], axis=1, inplace=True)

auto_ml(df, ‘kmeans’)

●	Return
	All the results of the selected model.

●	How to operate
1.	Randomly select features from the list of numeric features.
2.	Encoding and Scaling using scale_encode_combination().
3.	Run the selected algorithm; one of [ test_kmeans(), test_gaussian(), test_clarans(), test_dbscan() and test_mean_shift() ].



def scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list)

-	Scaling and Encoding with 15 combinations.

●	Parameters
dataset: DataFrame to be scaled and encoded
numerical_feature_list: Features to scale
categorical_feature_list: Features to encode

●	Examples
    for combination in feature_combination_list:
        data_combination = scale_encode_combination(dataset, combination, ['ocean_proximity'])
        for data_name, data in data_combination.items():
            data = data[combination]
            test_kmeans(data)
            test_gaussian(data)
            test_clarans(data)
            test_dbscan(data)
            test_mean_shift(data)

●	Return
	Dictionary included all the dataframe combinations of Scalers and Encoders.

●	How to operate
1.	for in scalers [StandardScaler(), MinMaxScaler(), RobustScaler(),  MaxAbsScaler(), Normalizer()]
2.	for in encoders [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
3.	Save each dataset in dictionary

Output:
->Kmeans

<img width="383" alt="캡처4" src="https://user-images.githubusercontent.com/74089524/139523330-494fcc85-f701-4748-904b-2420f05620db.PNG">

<img width="391" alt="캡처5" src="https://user-images.githubusercontent.com/74089524/139523359-60209d33-b6b3-4bfa-b674-12d9617ad626.PNG">

<img width="403" alt="캡처6" src="https://user-images.githubusercontent.com/74089524/139523360-be3f77d8-d6a3-415d-a4fd-dd2a6fd60f77.PNG">

<img width="408" alt="캡처7" src="https://user-images.githubusercontent.com/74089524/139523361-59bc20ae-a2e9-4476-974c-8c38d0eb8a1a.PNG">

Comaring with Median value

<img width="384" alt="캡처8" src="https://user-images.githubusercontent.com/74089524/139523414-8b30ae10-bb12-4484-b03b-373425b3738d.PNG">

<img width="339" alt="캡처9" src="https://user-images.githubusercontent.com/74089524/139523420-2944a5fa-28aa-46cb-a1c0-121359267455.PNG">

K-means Elbow Curve

<img width="390" alt="캡처10" src="https://user-images.githubusercontent.com/74089524/139523471-02d17c0c-68b0-467e-ba76-cf9e5264d707.PNG">

3 & 4 are decreasing sharply. So, 3 and 4 is the optimal K in Elbow curve
 
->EM(GMM)

<img width="451" alt="캡처11" src="https://user-images.githubusercontent.com/74089524/139523542-d8e6980f-b024-4d64-8b0f-895df8a81773.PNG">

Comparing median value

<img width="405" alt="캡처12" src="https://user-images.githubusercontent.com/74089524/139523569-3eae70af-2164-4fd3-ba38-b4bb596cf973.PNG">

->Clarans

<img width="345" alt="캡처13" src="https://user-images.githubusercontent.com/74089524/139523612-fdc757ae-2078-4d78-a06b-2a51c3cc422d.PNG">

Comparing with median value

<img width="398" alt="캡처14" src="https://user-images.githubusercontent.com/74089524/139523615-36add5e2-7eca-4e41-81b7-a648970678da.PNG">

->DBSCAN

<img width="511" alt="캡처15" src="https://user-images.githubusercontent.com/74089524/139523658-3cf1f3fa-fcfb-49be-b4ae-47e2de85e039.PNG">

Comaring with median value

<img width="331" alt="캡처16" src="https://user-images.githubusercontent.com/74089524/139523660-0a6430e6-830a-4fc7-8fe1-8e3d846099aa.PNG">

DB scan Elbow curve

<img width="384" alt="캡처17" src="https://user-images.githubusercontent.com/74089524/139523664-ae24d2a1-da9c-4c67-8096-e4303818b5b2.PNG">

Through this graph, eps=0.01, min_samples=19000 can be determined as appropriate hyperparameter values

->MeanShift

<img width="359" alt="캡처18" src="https://user-images.githubusercontent.com/74089524/139523702-35942c61-26be-473a-b5dc-f8f7b08a9c46.PNG">

Comparing with median value

<img width="342" alt="캡처19" src="https://user-images.githubusercontent.com/74089524/139523703-c2047629-2117-4587-8237-778c6d496b3c.PNG">

-->Our measurement is silhouette score with Euclid & Manhattan & L2 & L1 and Purity Score
