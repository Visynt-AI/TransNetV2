import os
import json
import logging
import tempfile
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    scenes: List[List[int]]
    single_frame_predictions: List[float]
    all_frame_predictions: List[float]
    frame_count: int


class TransNetPredictor:
    _instance: Optional["TransNetPredictor"] = None
    _model = None
    _device = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, weights_path: str, device: str = "cpu"):
        if self._model is not None:
            return

        self.weights_path = weights_path
        self.device = device

        self._load_model()

    def _load_model(self):
        from inference_pytorch.transnetv2_pytorch import TransNetV2

        logger.info(f"Loading TransNetV2 model from {self.weights_path}")
        logger.info(f"Using device: {self.device}")

        self._model = TransNetV2()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(self.device)

        logger.info("Model loaded successfully")

    def extract_frames(self, video_path: str) -> np.ndarray:
        import ffmpeg

        logger.info(f"Extracting frames from {video_path}")

        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        frames = np.frombuffer(out, np.uint8).reshape([-1, 27, 48, 3])
        logger.info(f"Extracted {len(frames)} frames")
        return frames

    def predict_frames(
        self, frames: np.ndarray, batch_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        no_padded_frames_start = 25
        no_padded_frames_end = (
            25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
        )

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start
            + [frames]
            + [end_frame] * no_padded_frames_end,
            0,
        )

        predictions_single = []
        predictions_all = []

        all_tensor = torch.from_numpy(padded_inputs).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                input_tensor = all_tensor[:, ptr : ptr + 100]

                single_frame_pred, all_frame_pred = self._model(input_tensor)

                single_frame_pred = (
                    torch.sigmoid(single_frame_pred).cpu().numpy()[0, 25:75, 0]
                )
                all_frame_pred = (
                    torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()[0, 25:75, 0]
                )

                predictions_single.append(single_frame_pred)
                predictions_all.append(all_frame_pred)

                ptr += 50

        single_frame_predictions = np.concatenate(predictions_single)[: len(frames)]
        all_frame_predictions = np.concatenate(predictions_all)[: len(frames)]

        return single_frame_predictions, all_frame_predictions

    @staticmethod
    def predictions_to_scenes(
        predictions: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def visualize_predictions(
        self,
        frames: np.ndarray,
        single_frame_predictions: np.ndarray,
        all_frame_predictions: np.ndarray,
    ) -> Image.Image:
        from PIL import Image, ImageDraw

        predictions = [single_frame_predictions, all_frame_predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames_padded = np.pad(
            frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)]
        )

        predictions_padded = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames_padded) // width

        img = frames_padded.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(
            np.split(np.concatenate(np.split(img, height), axis=2)[0], width), axis=2
        )[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i, pred in enumerate(zip(*predictions_padded)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)

        return img

    def predict_video(self, video_path: str) -> PredictionResult:
        frames = self.extract_frames(video_path)
        single_frame_pred, all_frame_pred = self.predict_frames(frames)
        scenes = self.predictions_to_scenes(single_frame_pred)

        return PredictionResult(
            scenes=scenes.tolist(),
            single_frame_predictions=single_frame_pred.tolist(),
            all_frame_predictions=all_frame_pred.tolist(),
            frame_count=len(frames),
        )

    def predict_video_with_visualization(
        self, video_path: str
    ) -> Tuple[PredictionResult, Image.Image]:
        frames = self.extract_frames(video_path)
        single_frame_pred, all_frame_pred = self.predict_frames(frames)
        scenes = self.predictions_to_scenes(single_frame_pred)

        visualization = self.visualize_predictions(
            frames, single_frame_pred, all_frame_pred
        )

        result = PredictionResult(
            scenes=scenes.tolist(),
            single_frame_predictions=single_frame_pred.tolist(),
            all_frame_predictions=all_frame_pred.tolist(),
            frame_count=len(frames),
        )

        return result, visualization
