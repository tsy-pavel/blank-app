import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_access_model():
    all_files = glob.glob("train_data_*.csv")
    if not all_files: return

    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    # 1. Более глубокая очистка данных
    for col in ['surface', 'smoothness', 'highway', 'lit']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x.replace("[", "").replace("'", "").split(",")[0].strip().lower())
            df[col] = df[col].replace('nan', 'unknown')

    # 2. Выделяем признаки (теперь включая длину)
    cat_features = ['highway', 'surface', 'smoothness', 'lit']
    num_features = ['length']

    X = df[cat_features + num_features]
    y = df['target']

    # 3. Pipeline
    # Категории кодируем, числа — масштабируем (StandardScaler)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features),
            ('num', numeric_transformer, num_features)
        ])

    # 4. Тюнинг классификатора
    # class_weight='balanced' помогает, если доступных дорог в данных гораздо больше, чем нет
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    accuracy = model_pipeline.score(X_test, y_test)
    print(f"🎯 Точность модели: {accuracy:.2%}")

    joblib.dump(model_pipeline, 'city_access_model.pkl')

if __name__ == "__main__":
    train_access_model()
