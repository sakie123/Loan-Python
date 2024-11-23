from sklearn.neighbors import KNeighborsClassifier

# Function missing values
def fill_missing_values(df, categorical_columns, numerical_columns):
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())
    return df

# KNN
def knn_varying_k(X_train, y_train, X_test, y_test, max_k=20):
    scores = []
    
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)
    
    return scores