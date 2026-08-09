import os
import logging
from pathlib import Path


import pandas as pd
from dotenv import load_dotenv
from joblib import dump


from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    recall_score,
)


def train_model():
    try:
        # load .env to my file
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        DATASET_PATH = (
            PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        )

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(
            os.getenv("TEST_SIZE")
        )  # it will be saved as string so we should use `int` to make it a number
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        # here we create the folders that will be needed
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
        )

        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape of {df.shape}")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        row_sig = pd.util.hash_pandas_object(X, index=False)

        gss = GroupShuffleSplit(
            n_splits=2, random_state=RANDOM_STATE, test_size=TEST_SIZE
        )

        train_idx, test_idx = next(gss.split(X, y, groups=row_sig))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(
            f"dataset splited with shapes of \n train: {X_train.shape}, {y_train.shape} \n test {X_test.shape}, {y_test.shape} "
        )

        best_rfc_pip = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        ccp_alpha=0.004744981749936002,
                        max_depth=3,
                        max_features="sqrt",
                        max_samples=0.7978948430708144,
                        min_samples_leaf=19,
                        min_samples_split=17,
                        n_estimators=534,
                        bootstrap=True,
                        n_jobs=-1
                    ),
                ),
            ]
        )
        
        
        best_rfc_pip.fit(X_train, y_train)
        logging.info('Model Training is DONE')
        
        # you can put evaluation metrics in here too
        
        # save trained model
        dump(best_rfc_pip, MODEL_PATH)
        logging.info(f'Model Saved to {MODEL_PATH}')
        logging.info('Model Training Completed')

    except Exception as e:
        print(f"Training Failed: {e}")
        logging.exception(f"Training Failed {e}")
        raise


# the purpose of using the block of code is to prevent the function `train_model` to work on
# another scripy, it will only run if we are in this script specificely
if __name__ == "__main__":
    train_model()
