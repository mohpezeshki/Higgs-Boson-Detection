import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HiggsDataPipline:

    def __init__(self, data_path: str, random_state: int = 42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.scaler = StandardScaler()


    def load_data(self):
        logger.info(f"Loading data from {self.data_path}...")

        df = pd.read_csv(self.data_path, names=["label"] + [f"feature_{i}_low" if i < 22 else f"feature_{i}_high" for i in range(1, 29)])
        X = df.drop("label", axis=1).values
        y = df["label"].values

        y = y.astype(int)
        
        logger.info(f"Data loaded: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        return X, y , df

    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray):
        logger.info("Normalizing data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        logger.info("Data normalization completed.")
        return X_train_scaled, X_test_scaled, X_val_scaled

    def train_eval_test_split(self, x , y, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, random_seed=None):
        assert train_ratio + eval_ratio + test_ratio == 1.0, "Ratios must sum to 1"

        logger.info(f"Splitting data into train ({train_ratio*100}%), eval ({eval_ratio*100}%), test ({test_ratio*100}%)...")

        train_dataset_x, temp_dataset_x, train_dataset_y, temp_dataset_y = train_test_split(x, y, test_size=eval_ratio + test_ratio, random_state=random_seed)
        eval_dataset_x, test_dataset_x, eval_dataset_y, test_dataset_y = train_test_split(temp_dataset_x, temp_dataset_y, test_size=test_ratio / (eval_ratio + test_ratio), random_state=random_seed)

        logger.info(f"Train: {train_dataset_x.shape[0]:,} | "
                f"Val: {eval_dataset_x.shape[0]:,} | "
                f"Test: {test_dataset_x.shape[0]:,}")
        return train_dataset_x, eval_dataset_x, test_dataset_x, train_dataset_y, eval_dataset_y, test_dataset_y

    def save_data(self, X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, y_val: np.ndarray, save_path: str):

        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True, exist_ok=True)
            Path(f"{save_path}/train").mkdir(parents=True, exist_ok=True)
            Path(f"{save_path}/test").mkdir(parents=True, exist_ok=True)
            Path(f"{save_path}/val").mkdir(parents=True, exist_ok=True)

        pd.DataFrame(X_train).to_csv(f"{save_path}/train/train_dataset.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{save_path}/test/test_dataset.csv", index=False)
        pd.DataFrame(X_val).to_csv(f"{save_path}/val/val_dataset.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{save_path}/train/train_labels.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{save_path}/test/test_labels.csv", index=False)
        pd.DataFrame(y_val).to_csv(f"{save_path}/val/val_labels.csv", index=False)


        logging.info(f"Data saved to {save_path}.")
    
    def run(data_path: str, save_path: str, random_seed: int = 42 , save_normalized: bool = True):
        logging.info("Starting data preparation...")

        higgs_data_pipeline = HiggsDataPipline(data_path=data_path, random_state=random_seed)
        X, y, df = higgs_data_pipeline.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = higgs_data_pipeline.train_eval_test_split(X, y, random_seed=random_seed)

        if save_normalized:
            X_train_scaled, X_val_scaled, X_test_scaled = higgs_data_pipeline.normalize_data(X_train, X_val, X_test)

        # 4. SAVE PROCESSED DATA
        higgs_data_pipeline.save_data(X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, save_path)

        logging.info("Data preparation completed successfully.")


if __name__ == "__main__":
    data_path = "data/raw/higgs-uci-dataset/HIGGS.csv"
    save_path = "data/processed/higgs-uci-dataset"
    random_seed = 42

    HiggsDataPipline.run(data_path=data_path, save_path=save_path, random_seed=random_seed)

