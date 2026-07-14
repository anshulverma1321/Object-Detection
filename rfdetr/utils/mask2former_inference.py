import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


class Mask2FormerInference:
    """
    API Wrapper for Mask2Former

    Usage:

        model = Mask2FormerInference()

        result = model.segment(image)

    Input:
        PIL.Image
        numpy/cv2 image

    Output:
        segmented overlay image (numpy array)
    """

    def __init__(
        self,
        model_name="facebook/mask2former-swin-small-ade-semantic",
        target_size=384,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = target_size

        print(f"[Mask2Former] Loading model on {self.device}...")

        torch.set_grad_enabled(False)

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name
        )

        self.model = (
            Mask2FormerForUniversalSegmentation
            .from_pretrained(model_name)
            .to(self.device)
        )

        if self.device == "cuda":
            self.model = self.model.half()

        self.model.eval()

        np.random.seed(42)
        self.colors = np.random.randint(
            0,
            255,
            (150, 3),
            dtype=np.uint8
        )

        print("[Mask2Former] Model loaded successfully")

    def _to_cv2(self, image):

        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    def segment(self, image):

        frame = self._to_cv2(image)

        orig_h, orig_w = frame.shape[:2]

        frame_small = cv2.resize(
            frame,
            (self.target_size, self.target_size)
        )

        rgb = cv2.cvtColor(
            frame_small,
            cv2.COLOR_BGR2RGB
        )

        inputs = self.processor(
            images=rgb,
            return_tensors="pt"
        )

        if self.device == "cuda":
            inputs = {
                k: v.to(self.device).half()
                for k, v in inputs.items()
            }
        else:
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        result = (
            self.processor
            .post_process_semantic_segmentation(
                outputs,
                target_sizes=[
                    (
                        self.target_size,
                        self.target_size
                    )
                ],
            )[0]
            .cpu()
            .numpy()
        )

        mask = self.colors[result]

        mask = cv2.resize(
            mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        overlay = cv2.addWeighted(
            frame,
            0.6,
            mask,
            0.4,
            0,
        )

        return overlay

    def __call__(self, image):
        return self.segment(image)