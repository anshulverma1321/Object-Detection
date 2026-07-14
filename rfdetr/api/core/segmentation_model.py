from typing import Optional

from utils.mask2former_inference import Mask2FormerInference

_model: Optional[Mask2FormerInference] = None
_model_error = None


def load_segmentation_model():
    global _model, _model_error

    if _model is not None:
        return _model

    try:
        _model = Mask2FormerInference()
        _model_error = None

        print(
            "[SEGMENTATION API] Mask2Former model loaded successfully"
        )

        return _model

    except Exception as e:
        _model_error = str(e)
        print(
            f"[SEGMENTATION API] Failed to load model: {_model_error}"
        )
        return None


def get_segmentation_model():

    if _model is None:
        load_segmentation_model()

    if _model is None:
        raise RuntimeError(
            f"Mask2Former model not loaded.\n\nError:\n{_model_error}"
        )

    return _model


def get_segmentation_status():
    return {
        "loaded": _model is not None,
        "error": _model_error,
    }