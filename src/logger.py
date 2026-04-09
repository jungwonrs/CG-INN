import os
import pandas as pd


class ExperimentLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, "training_log.csv")
        self.rows = []

    def log_overall(self, stats_dict):
        self.rows.append(stats_dict)
        df = pd.DataFrame(self.rows)
        try:
            df.to_csv(self.save_path, index=False, encoding='utf-8-sig')
        except PermissionError:
            backup = self.save_path.replace(".csv", "_backup.csv")
            df.to_csv(backup, index=False, encoding='utf-8-sig')
