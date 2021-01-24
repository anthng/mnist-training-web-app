def preprocess_data(X, y=None):
    from keras.utils import to_categorical
    
    X_pre, y_pre = X.reshape(-1, 8, 8, 1), to_categorical(y)
    X_pre = normalize_data(X_pre)
    
    return X_pre, y_pre


def normalize_data(X):
    return X.astype('float32')/255.0