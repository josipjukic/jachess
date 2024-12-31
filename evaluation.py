import torch
from util import logits_to_probs
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
)
from scipy.stats import spearmanr, pearsonr
import pandas as pd


def brier_multi(y_true, y_prob):
    if len(y_prob.shape) > 1:
        return torch.sum((y_true - y_prob) ** 2, dim=1).mean()
    else:
        return torch.mean((y_true - y_prob) ** 2)


class EvalSet:
    def __init__(self, meta):
        self.meta = meta
        self.evals = {}

    def add(self, eval, seed):
        self.evals[seed] = eval

    def to_pandas(self):
        rows = []
        for k, v in self.evals.items():
            for epoch, (train, test) in enumerate(zip(v.train, v.test), 1):
                result_dict = train.to_dict() | test.to_dict()
                # geomstats = {"emb_grads": train.emb_grads, "enc_grads": train.enc_grads}
                row = self.meta | {"seed": k} | {"epoch": epoch} | result_dict
                rows.append(row)
        df = pd.DataFrame(rows)
        df.set_index(["dataset", "model", "peft", "seed", "epoch"], inplace=True)
        return df


class Evaluation:
    def __init__(self):
        self.train = []
        self.test = []

    def add_train_result(self, result):
        self.train.append(result)

    def add_test_result(self, result):
        self.test.append(result)


class Result:
    def __init__(self, loss, task_type, result_type):
        self.loss = loss
        self.task_type = task_type
        self.result_type = result_type

    def evaluate(self, y_true, logits):
        self.y_true = y_true
        self.logits = logits
        if self.task_type == "clf":
            probs = logits_to_probs(logits)
            y_pred = torch.argmax(probs, dim=1)
            self.accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            self.f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
            self.f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
            self.f1_weighted = f1_score(
                y_true=y_true, y_pred=y_pred, average="weighted"
            )
            self.matthews_corrcoef = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
            self.brier_score = brier_multi(y_true=y_true, y_prob=probs[:, 1])
        else:
            self.spearmanr = spearmanr(y_true.squeeze(), logits.squeeze())
            self.pearsonr = pearsonr(y_true.squeeze(), logits.squeeze())

    def to_dict(self):
        if self.task_type == "clf":
            return {
                f"{self.result_type}_loss": self.loss,
                f"{self.result_type}_accuracy": self.accuracy,
                f"{self.result_type}_f1_micro": self.f1_micro,
                f"{self.result_type}_f1_macro": self.f1_macro,
                f"{self.result_type}_f1_weighted": self.f1_weighted,
                f"{self.result_type}_matthews_corrcoef": self.matthews_corrcoef,
                f"{self.result_type}_brier_score": self.brier_score,
            }

        else:
            return {
                f"{self.result_type}_loss": self.loss,
                f"{self.result_type}_spearmanr": self.spearmanr,
                f"{self.result_type}_pearsonr": self.pearsonr,
            }

    def __repr__(self):
        if self.task_type == "clf":
            return "\n".join(
                [
                    f"Accuracy: {self.accuracy:.3f}",
                    f"F1 micro: {self.f1_micro:.3f}",
                    f"F1 macro: {self.f1_macro:.3f}",
                    f"F1 weighted: {self.f1_weighted:.3f}",
                    f"Matthews corrcoef: {self.matthews_corrcoef:.3f}",
                    f"Brier score: {self.brier_score}",
                ]
            )
        else:
            return "\n".join(
                [f"Spearmanr: {self.spearmanr}", f"Pearsonr: {self.pearsonr}"]
            )

    def __str__(self):
        return self.__repr__()
