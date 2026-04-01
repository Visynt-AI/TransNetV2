import glob
import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    scenes: List[List[int]]
    frame_count: int


class TransNetPredictor:
    _instance: Optional["TransNetPredictor"] = None

    _model: Optional[nn.Module] = None
    _device: Optional[str] = None
    _weights_path: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, weights_path: str, device: str = "cpu"):
        if self.__class__._model is not None:
            if weights_path != self.__class__._weights_path or device != self.__class__._device:
                raise RuntimeError(
                    f"TransNetPredictor already initialized with "
                    f"weights_path={self.__class__._weights_path!r}, device={self.__class__._device!r}. "
                    f"Cannot reinitialize with different parameters."
                )
            self.weights_path = self.__class__._weights_path
            self.device = self.__class__._device
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

        self.__class__._weights_path = self.weights_path
        self.__class__._device = self.device

        logger.info("Model loaded successfully")

    def extract_frames(self, video_path: str) -> np.ndarray:
        import ffmpeg

        logger.info(f"Extracting frames from {video_path}")

        # probe = ffmpeg.probe(video_path)
        # video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        # width = int(video_info["width"])
        # height = int(video_info["height"])

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
                many_hot = all_frame_pred.get("many_hot") if isinstance(all_frame_pred, dict) else None
                if many_hot is None:
                    raise RuntimeError(
                        "Model output missing 'many_hot' key; got keys: "
                        f"{list(all_frame_pred.keys()) if isinstance(all_frame_pred, dict) else type(all_frame_pred)}"
                    )
                all_frame_pred = torch.sigmoid(many_hot).cpu().numpy()[0, 25:75, 0]

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

        if len(predictions) == 0:
            return np.empty((0, 2), dtype=np.int32)

        scenes = []
        t, t_prev, start = -1, 0, 0
        last_index = len(predictions) - 1
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t == 0:
            scenes.append([start, last_index])

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

    def predict_video(
        self, video_path: str, threshold: float = 0.5
    ) -> PredictionResult:
        frames = self.extract_frames(video_path)
        single_frame_pred, all_frame_pred = self.predict_frames(frames)
        scenes = self.predictions_to_scenes(single_frame_pred, threshold=threshold)

        return PredictionResult(
            scenes=scenes.tolist(),
            frame_count=len(frames),
        )

    def predict_video_with_visualization(
        self, video_path: str, threshold: float = 0.5
    ) -> Tuple[PredictionResult, Image.Image]:
        frames = self.extract_frames(video_path)
        single_frame_pred, all_frame_pred = self.predict_frames(frames)
        scenes = self.predictions_to_scenes(single_frame_pred, threshold=threshold)

        visualization = self.visualize_predictions(
            frames, single_frame_pred, all_frame_pred
        )

        result = PredictionResult(
            scenes=scenes.tolist(),
            frame_count=len(frames),
        )

        return result, visualization

    def extract_frame_image(
        self, video_path: str, frame_index: int, output_path: str
    ) -> None:
        extracted_frames = self.extract_frame_images(
            video_path, [frame_index], os.path.dirname(output_path)
        )
        extracted_path = extracted_frames.get(frame_index)
        if extracted_path is None:
            raise RuntimeError(f"Failed to extract frame {frame_index} from {video_path}")
        if extracted_path != output_path:
            os.replace(extracted_path, output_path)

    def extract_frame_images(
        self, video_path: str, frame_indices: List[int], output_dir: str
    ) -> dict[int, str]:
        import ffmpeg

        unique_frame_indices = sorted(set(frame_indices))
        if not unique_frame_indices:
            return {}

        os.makedirs(output_dir, exist_ok=True)
        max_frames_per_batch = 64
        extracted_frames: dict[int, str] = {}
        extraction_id = uuid.uuid4().hex

        for batch_start in range(0, len(unique_frame_indices), max_frames_per_batch):
            batch_indices = unique_frame_indices[
                batch_start : batch_start + max_frames_per_batch
            ]
            batch_prefix = f"frame-{extraction_id}-{batch_start:06d}"
            output_pattern = os.path.join(output_dir, f"{batch_prefix}-%06d.png")
            select_expr = "+".join(
                f"eq(n,{frame_index})" for frame_index in batch_indices
            )

            try:
                (
                    ffmpeg.input(video_path)
                    .filter("select", select_expr)
                    .output(
                        output_pattern,
                        format="image2",
                        vcodec="png",
                        fps_mode="vfr",
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                stderr = (
                    exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                )
                raise RuntimeError(
                    f"Failed to extract frames {batch_indices} from {video_path}: {stderr}"
                ) from exc

            extracted_paths = sorted(
                glob.glob(os.path.join(output_dir, f"{batch_prefix}-*.png"))
            )
            if len(extracted_paths) != len(batch_indices):
                raise RuntimeError(
                    "Extracted frame count does not match request: "
                    f"requested={len(batch_indices)} extracted={len(extracted_paths)}"
                )

            extracted_frames.update(dict(zip(batch_indices, extracted_paths)))

        return extracted_frames
