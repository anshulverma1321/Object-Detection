import os
import json
import inspect
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class RFDETRInference:
    """
    Wrapper class used by notebook/API:

        from utils.inference import RFDETRInference
        model_wrapper = RFDETRInference(size="M", config_path="../configs/config.json")
        detections = model_wrapper.predict(image, threshold=0.5)

    Supports:
    - cv2 image / numpy image
    - PIL image
    - optional config.json
    - RF-DETR package class auto selection
    """

    def __init__(self, size="M", config_path=None, **kwargs):
        self.size = str(size).upper().strip()
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.model = self._load_model(**kwargs)

    def _load_config(self, config_path):
        """
        Config optional hai.
        Agar config file missing hai to crash nahi karega.
        """
        if not config_path:
            return {}

        possible_paths = []

        # As given
        possible_paths.append(Path(config_path))

        # Relative to current working directory
        possible_paths.append(Path.cwd() / config_path)

        # Relative to project root
        project_root = Path(__file__).resolve().parents[1]
        possible_paths.append(project_root / config_path)

        # Common configs/config.json
        possible_paths.append(project_root / "configs" / "config.json")

        for path in possible_paths:
            try:
                path = path.resolve()
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass

        return {}

    def _get_rfdetr_class(self):
        """
        RF-DETR package ke different class names auto handle karega.
        """

        try:
            import rfdetr
        except Exception as e:
            raise ImportError(
                "RF-DETR package import nahi ho raha. Pehle install karo:\n"
                "pip install rfdetr\n\n"
                f"Original error: {repr(e)}"
            )

        size_map = {
            "N": ["RFDETRNano", "RFDETRNanoModel"],
            "NANO": ["RFDETRNano", "RFDETRNanoModel"],

            "S": ["RFDETRSmall", "RFDETRSmallModel"],
            "SMALL": ["RFDETRSmall", "RFDETRSmallModel"],

            "M": ["RFDETRMedium", "RFDETRBase", "RFDETR"],
            "MEDIUM": ["RFDETRMedium", "RFDETRBase", "RFDETR"],

            "B": ["RFDETRBase", "RFDETR"],
            "BASE": ["RFDETRBase", "RFDETR"],

            "L": ["RFDETRLarge", "RFDETRBase", "RFDETR"],
            "LARGE": ["RFDETRLarge", "RFDETRBase", "RFDETR"],

            "XL": ["RFDETRLarge", "RFDETRBase", "RFDETR"],
        }

        candidates = size_map.get(self.size, size_map["M"])

        for class_name in candidates:
            if hasattr(rfdetr, class_name):
                return getattr(rfdetr, class_name)

        # Try submodules also
        possible_modules = [
            "rfdetr",
            "rfdetr.models",
            "rfdetr.model",
            "rfdetr.detr",
        ]

        for module_name in possible_modules:
            try:
                module = __import__(module_name, fromlist=["*"])
                for class_name in candidates:
                    if hasattr(module, class_name):
                        return getattr(module, class_name)
            except Exception:
                continue

        available = [name for name in dir(rfdetr) if "RF" in name or "DETR" in name]

        raise ImportError(
            "RF-DETR class nahi mili installed rfdetr package me.\n"
            f"Tried: {candidates}\n"
            f"Available RF/DETR names: {available}"
        )

    def _build_model_kwargs(self, model_class, extra_kwargs):
        """
        Config se weights/checkpoint agar available ho to model constructor me pass karega.
        Sirf wahi arguments pass karega jo constructor accept karta hai.
        """

        config = self.config or {}
        model_kwargs = {}

        possible_weight_keys = [
            "pretrain_weights",
            "pretrained_weights",
            "checkpoint_path",
            "weights_path",
            "model_path",
            "weights",
        ]

        for key in possible_weight_keys:
            if key in config and config[key]:
                model_kwargs[key] = config[key]

        # User provided kwargs priority
        model_kwargs.update(extra_kwargs)

        try:
            sig = inspect.signature(model_class.__init__)
            accepted_params = set(sig.parameters.keys())

            # Agar constructor **kwargs accept karta hai to sab pass karo
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            if accepts_kwargs:
                return model_kwargs

            filtered = {
                k: v for k, v in model_kwargs.items()
                if k in accepted_params
            }

            return filtered

        except Exception:
            return model_kwargs

    def _load_model(self, **kwargs):
        model_class = self._get_rfdetr_class()
        model_kwargs = self._build_model_kwargs(model_class, kwargs)

        try:
            model = model_class(**model_kwargs)
            print(f"Loaded RF-DETR model: {model_class.__name__}")
            return model

        except TypeError:
            # Agar kwargs issue kare to no-argument constructor try karo
            model = model_class()
            print(f"Loaded RF-DETR model without kwargs: {model_class.__name__}")
            return model

        except Exception as e:
            raise RuntimeError(
                f"RF-DETR model load failed using {model_class.__name__}.\n"
                f"Model kwargs: {model_kwargs}\n"
                f"Original error: {repr(e)}"
            )

    def _to_pil_rgb(self, image):
        """
        API/video/image input ko PIL RGB me convert karta hai.
        """

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            # Grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(image)

            # BGRA
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(image)

            # BGR from cv2 to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Use PIL.Image or numpy/cv2 image."
        )

    def predict(self, image, threshold=0.5):
        """
        Returns RF-DETR detections.
        Usually this is supervision.Detections object having:
        - xyxy
        - confidence
        - class_id
        """

        image = self._to_pil_rgb(image)

        if threshold is None:
            threshold = 0.5

        threshold = float(threshold)

        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")

        if not hasattr(self.model, "predict"):
            raise AttributeError(
                "Loaded RF-DETR model does not have predict() method."
            )

        try:
            return self.model.predict(image, threshold=threshold)

        except TypeError:
            # Some versions may use confidence instead of threshold
            try:
                return self.model.predict(image, confidence=threshold)
            except TypeError:
                return self.model.predict(image)

    def __call__(self, image, threshold=0.5):
        return self.predict(image, threshold=threshold)