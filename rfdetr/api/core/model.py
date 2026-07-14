import importlib
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from api.core.config import ROOT_DIR, RFDETR_CONFIG_PATH, RFDETR_SIZE

_model: Optional[Any] = None
_model_error: Optional[str] = None


def _ensure_project_root_on_path():
    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_load_from_utils_inference():
    """
    Notebook flow:
        from utils.inference import RFDETRInference
        model_wrapper = RFDETRInference(size="M", config_path="../configs/config.json")
    """
    module = importlib.import_module("utils.inference")
    cls = getattr(module, "RFDETRInference")

    config_path = Path(RFDETR_CONFIG_PATH)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    # Try common constructor styles
    try:
        return cls(size=RFDETR_SIZE, config_path=str(config_path))
    except TypeError:
        try:
            return cls(size=RFDETR_SIZE, config_path=RFDETR_CONFIG_PATH)
        except TypeError:
            try:
                return cls()
            except TypeError:
                return cls(RFDETR_SIZE, str(config_path))


def _try_load_from_rfdetr_package():
    """
    Fallback for official RF-DETR style packages.
    This is intentionally flexible because different RF-DETR installs expose different classes.
    """
    candidates = [
        ("rfdetr", "RFDETRBase"),
        ("rfdetr", "RFDETRMedium"),
        ("rfdetr", "RFDETRLarge"),
        ("rfdetr.main", "RFDETRBase"),
        ("rfdetr.main", "RFDETRMedium"),
        ("rfdetr.main", "RFDETRLarge"),
    ]

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls()
        except Exception:
            continue

    raise ImportError("Could not load any RF-DETR class from installed rfdetr package.")


def load_model():
    global _model, _model_error

    _ensure_project_root_on_path()

    if _model is not None:
        return _model

    errors = []

    loaders = [
        ("utils.inference.RFDETRInference", _try_load_from_utils_inference),
        ("installed rfdetr package", _try_load_from_rfdetr_package),
    ]

    for name, loader in loaders:
        try:
            _model = loader()
            _model_error = None
            print(f"[RF-DETR API] Model loaded using: {name}")
            return _model
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    _model_error = "\n".join(errors)
    print("[RF-DETR API] Model loading failed:")
    print(_model_error)
    print(traceback.format_exc())

    # Do not crash server. Keep API alive and return clear error on /detect.
    _model = None
    return None


def get_model():
    if _model is None:
        load_model()

    if _model is None:
        raise RuntimeError(
            "RF-DETR model is not loaded.\n\n"
            "Fix options:\n"
            "1. Put this API folder inside your real RF-DETR project root.\n"
            "2. Make sure this file exists: utils/inference.py\n"
            "3. Make sure it contains class RFDETRInference.\n"
            "4. Or install/use your RF-DETR package and update api/core/model.py adapter.\n\n"
            f"Original loading errors:\n{_model_error}"
        )

    return _model


def get_model_status():
    return {
        "loaded": _model is not None,
        "error": _model_error,
        "expected_notebook_import": "from utils.inference import RFDETRInference",
        "expected_file": str(ROOT_DIR / "utils" / "inference.py"),
        "project_root": str(ROOT_DIR),
    }
