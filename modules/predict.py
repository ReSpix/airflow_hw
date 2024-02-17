from datetime import datetime
import json
import dill
import pandas as pd
import os


proj_path = os.environ.get('PROJECT_PATH', '.')


def get_model_filename() -> str:
    directory = f"{proj_path}/data/models/"
    file_paths = os.listdir(directory)
    return directory + file_paths[-1]


def load_model():
    model_filename = get_model_filename()
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    return model


def load_test_data():
    directory = f"{proj_path}/data/test/"
    file_paths = os.listdir(directory)
    tests = []
    for path in file_paths:
        with open(directory + path, 'rb') as file:
            tests.append(json.load(file))
    return tests


def save_predictions(df: pd.DataFrame):
    path = proj_path + \
        f"/data/predictions/preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
    df.to_csv(path, index=False)


def predict():
    model = load_model()
    tests = load_test_data()
    df = pd.DataFrame(tests)
    preds_df = pd.DataFrame()
    preds = model.predict(df)
    preds_df['car_id'] = df['id']
    preds_df['pred'] = preds
    save_predictions(preds_df)


if __name__ == '__main__':
    predict()
