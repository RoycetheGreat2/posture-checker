"""``rules-v0``: a hand-written baseline model.

It needs no training data, so the app works from the first calibration. It is
also the baseline in the model registry: a trained model is only deployed if it
beats these rules on held-out data.

The thresholds are starting points chosen from geometry, not tuned values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from posture_checker.inference import (
    FORWARD_HEAD,
    GOOD,
    SLOUCH,
    TOO_CLOSE,
    Prediction,
)

# All thresholds are deviations from the calibrated baseline at sensitivity 1.
#: Face at least 25% larger than at calibration, i.e. roughly 20% closer.
TOO_CLOSE_EXCESS = 0.25
#: Face grew 10% more than the shoulders did: the head moved forward alone.
FORWARD_VS_SHOULDERS_EXCESS = 0.10
#: Without visible shoulders, a moderately larger face is the only lean-in cue.
FORWARD_FACE_ONLY_EXCESS = 0.12
#: Head sits 0.45 face-widths (about 4 cm) lower in the frame.
SLOUCH_FACE_DROP = 0.45
#: Head has sunk 15% of the way towards the shoulder line.
SLOUCH_NECK_SHRINK = 0.15


def _severity(deviation: float, threshold: float) -> float:
    """How far past the threshold a deviation is: 1.0 means exactly at it."""
    if deviation is None or math.isnan(deviation):
        return 0.0
    return max(deviation, 0.0) / threshold


def _score(severity: float) -> float:
    """Map severity to 0..1 so that the firing point (1.0) lands on 0.5."""
    return min(severity / 2.0, 1.0)


class RulesModel:
    """Threshold rules on relative features."""

    version = "rules-v0"

    def __init__(self, sensitivity: float = 1.0) -> None:
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        self.sensitivity = sensitivity

    def predict(self, relative: Mapping[str, float]) -> Prediction:
        s = self.sensitivity
        face_excess = relative["face_scale_ratio"] - 1.0
        has_shoulders = relative.get("shoulders_ok", 0.0) >= 0.5

        too_close = _severity(face_excess, TOO_CLOSE_EXCESS / s)

        if has_shoulders:
            forward = _severity(
                relative["head_vs_shoulder_ratio"] - 1.0, FORWARD_VS_SHOULDERS_EXCESS / s
            )
            neck = _severity(-relative["neck_ratio_change"], SLOUCH_NECK_SHRINK / s)
        else:
            forward = _severity(face_excess, FORWARD_FACE_ONLY_EXCESS / s)
            neck = 0.0
        slouch = max(_severity(relative["face_drop"], SLOUCH_FACE_DROP / s), neck)

        severities = {TOO_CLOSE: too_close, FORWARD_HEAD: forward, SLOUCH: slouch}
        scores = {name: round(_score(value), 4) for name, value in severities.items()}
        scores[GOOD] = round(1.0 - max(scores.values()), 4)

        # When several rules fire, report the most specific explanation first.
        label = GOOD
        for name in (TOO_CLOSE, FORWARD_HEAD, SLOUCH):
            if severities[name] >= 1.0:
                label = name
                break
        return Prediction(label=label, scores=scores, model_version=self.version)
