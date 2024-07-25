import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=5, mode="min", delta=0.0, save_dir=None):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_dir = save_dir

        if self.mode == "min":
            self.monitor_op = np.less
        elif self.mode == "max":
            self.monitor_op = np.greater

    def __call__(self, score, model=None) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.monitor_op(score - self.delta, self.best_score):
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        if model is not None:
            torch.save(model.state_dict(), self.save_dir)
            print(f"Model saved at {self.save_dir}")
        else:
            print("Model is None. Not saved.")
