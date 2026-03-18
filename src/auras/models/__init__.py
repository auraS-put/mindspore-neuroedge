"""Model factory placeholders for MindSpore implementations."""


def build_model(model_cfg: dict):
    family = model_cfg.get("family", "lstm")
    return {"family": family, "status": "placeholder"}
