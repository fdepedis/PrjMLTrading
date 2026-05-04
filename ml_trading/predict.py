import joblib
import numpy as np
from config.settings import MODEL_PATH

class AIDecisionEngine:

    def __init__(self, threshold: float = 0.55):
        self.model = joblib.load(MODEL_PATH)
        self.threshold = threshold

    def build_features(self, context: dict):
        entry = context["entry_price"]
        sl = context["exp_sl"]
        tp = context["exp_tp"]
        mae = context.get("mae_1m")

        if mae is None:
            mae = 0.0

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        rr_ratio = reward / risk if risk != 0 else 0.0
        mae_ratio = mae / risk if risk != 0 else 0.0

        return np.array([[rr_ratio, mae_ratio]])

    def predict(self, context: dict):
        X = self.build_features(context)

        proba = self.model.predict_proba(X)[0]

        prob_loss = float(proba[0])
        prob_win = float(proba[1])

        decision = prob_win > self.threshold

        return {
            "prob_win": prob_win,
            "prob_loss": prob_loss,
            "decision": decision
        }