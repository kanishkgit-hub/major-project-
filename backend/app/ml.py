"""The one place that talks to prediction models.

Every disease runs a clearly labelled STUB (model_version "stub-0") until real models exist.
If ML_SERVICE_URL is set, Diabetes instead calls Shubham's screening service over HTTP (contract:
docs/API.md section 7). To plug in real Heart / CKD models, add an HTTP call like `_diabetes` and route
to it in `run_model`; nothing else in the backend changes.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from fastapi import HTTPException

from .config import get_settings

STUB_VERSION = "stub-0"


@dataclass
class ModelResult:
    probability: float
    confidence: float
    severity: str
    model_version: str
    explanation: Optional[str] = None
    shap_values: Dict[str, float] = field(default_factory=dict)
    base_value: Optional[float] = None
    warnings: List[str] = field(default_factory=list)   # extra notes from the model adapter
    check_boundary: bool = True   # False if "close to 0.5" is not a decision boundary for this model


def severity_for(p: float) -> str:
    # One set of thresholds, so all three diseases grade risk identically.
    return "Low" if p < 0.2 else "Moderate" if p < 0.5 else "High" if p < 0.8 else "Critical"


DIABETES_VERSION = "nhanes-logreg-prototype"   # used when the service does not report its own version

# The service's own labels -> our severity scale. It never produces "Critical".
_DIABETES_SEVERITY = {
    "Normal": "Low", "Prediabetes": "Moderate", "Diabetes": "High",          # lab cut-off path
    "Low risk": "Low", "Moderate risk": "Moderate", "High risk": "High",     # machine-learning path
}


def _text(value) -> str:
    """The service's optional text fields: anything that is not a non-empty string is ignored."""
    return value.strip() if isinstance(value, str) else ""


def _diabetes_result(data: dict, features: dict) -> ModelResult:
    """Turn the screening service's answer into our ModelResult. Raises 502 if the answer is unusable."""
    try:
        label = data["result"]
        severity = _DIABETES_SEVERITY[label]
        probability = data["probability"]
        if isinstance(probability, bool):
            raise ValueError("probability must be a number")
        probability = float(probability)
        if not 0 <= probability <= 1:       # also rejects NaN
            raise ValueError("probability out of range")
        shap, parts = {}, []
        for f in data.get("top_factors") or []:
            contribution = float(f["contribution"])
            if not math.isfinite(contribution):
                raise ValueError("contribution must be finite")
            name = str(f["factor"])
            shap[name] = contribution
            parts.append(f"{name} ({f.get('value', 'n/a')}) {f.get('effect', 'affected')} the score")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HTTPException(502, "The Diabetes model service returned an unexpected answer "
                                 "(it must return result and probability, see docs/API.md section 7)")

    text = f"{label} (method: {_text(data.get('path')) or 'unknown'})."
    if parts:
        text += " Top factors: " + "; ".join(parts) + "."
    dtype = _text(data.get("diabetes_type"))
    if dtype:   # ponytail: rule-based type likelihoods are not stored, only the label
        text += f" Diabetes type estimate (rule-based, not machine learning): {dtype}."

    warnings = ["Prototype model trained on US NHANES data; not clinically validated."]
    if data.get("path") == "Lab cut-offs":
        warnings.append("This result comes from laboratory cut-offs (HbA1c or fasting glucose), not from the "
                        "machine-learning model. The probability shown is the model's estimate from body "
                        "measurements only.")
    if _text(data.get("note")):
        warnings.append(_text(data["note"]))
    for key, name in (("sedentary_min", "Sedentary time"), ("income_ratio", "Income ratio")):
        if features.get(key) is None:
            warnings.append(f"{name} was not provided: the model used the training-data median.")

    return ModelResult(probability, round(max(probability, 1 - probability), 4), severity,
                       _text(data.get("model_version")) or DIABETES_VERSION, text, shap, None, warnings,
                       check_boundary=False)   # its risk bands are 0.23 / 0.51, so 0.5 means nothing special


async def _diabetes(features: dict) -> ModelResult:
    base = get_settings().ml_service_url.rstrip("/")
    # The patient's name is never sent: the model does not need it.
    payload = {k: v for k, v in features.items() if k != "answers"}
    payload["answers"] = features.get("answers") or {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base}/api/screen", json=payload)
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach the Diabetes model service")
    if resp.status_code == 503:
        raise HTTPException(503, "The Diabetes model is not loaded yet")
    if resp.status_code != 200:
        raise HTTPException(502, "The Diabetes model service returned an error")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(502, "The Diabetes model service returned an unexpected answer")
    return _diabetes_result(data, features)


# (baseline probability, {feature: (reference value, weight per unit above reference)})
_STUBS = {
    "diabetes": (0.10, {"age": (40, 0.006), "waist_cm": (90, 0.004), "systolic_bp": (120, 0.002),
                        "hypertension": (0, 0.15), "smoker": (0, 0.04)}),
    "heart": (0.10, {"age": (40, 0.006), "trestbps": (120, 0.002), "chol": (200, 0.0008),
                     "oldpeak": (0, 0.06), "exang": (0, 0.15), "ca": (0, 0.10)}),
    "ckd": (0.05, {"sc": (1.2, 0.15), "hemo": (13, -0.05), "bp": (80, 0.003),
                   "htn": (0, 0.10), "dm": (0, 0.10)}),
}


def _stub(key: str, features: dict) -> ModelResult:
    base, terms = _STUBS[key]
    p = base + sum(w * (features[n] - ref) for n, (ref, w) in terms.items())
    p = round(min(0.98, max(0.02, p)), 4)
    return ModelResult(p, round(0.5 + abs(p - 0.5), 4), severity_for(p), STUB_VERSION)


async def run_model(key: str, features: dict) -> ModelResult:
    """key: 'diabetes' | 'heart' | 'ckd'."""
    if key == "diabetes" and get_settings().ml_service_url:
        return await _diabetes(features)
    return _stub(key, features)
