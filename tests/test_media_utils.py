import pytest

from app.media_utils import build_scene_sampling_plan, probe_video_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(duration="10.0", avg_frame_rate="25/1", r_frame_rate="25/1"):
    return {
        "streams": [
            {
                "codec_type": "video",
                "duration": duration,
                "avg_frame_rate": avg_frame_rate,
                "r_frame_rate": r_frame_rate,
            }
        ],
        "format": {"duration": duration},
    }


# ---------------------------------------------------------------------------
# probe_video_metadata
# ---------------------------------------------------------------------------

class TestProbeVideoMetadata:
    def test_normal_25fps(self):
        duration, fps = probe_video_metadata(_make_probe("10.0", "25/1"), 250)
        assert duration == pytest.approx(10.0)
        assert fps == pytest.approx(25.0)

    def test_normal_30fps(self):
        duration, fps = probe_video_metadata(_make_probe("5.0", "30/1"), 150)
        assert duration == pytest.approx(5.0)
        assert fps == pytest.approx(30.0)

    def test_fractional_fps(self):
        # 24000/1001 ≈ 23.976
        duration, fps = probe_video_metadata(_make_probe("10.0", "24000/1001"), 240)
        assert fps == pytest.approx(24000 / 1001, rel=1e-4)

    def test_duration_falls_back_to_format(self):
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "avg_frame_rate": "24/1",
                    "r_frame_rate": "24/1",
                }
            ],
            "format": {"duration": "5.0"},
        }
        duration, fps = probe_video_metadata(probe, 120)
        assert duration == pytest.approx(5.0)
        assert fps == pytest.approx(24.0)

    def test_fps_computed_from_frame_count_when_rate_is_zero(self):
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "10.0",
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "0/0",
                }
            ],
            "format": {"duration": "10.0"},
        }
        duration, fps = probe_video_metadata(probe, 300)
        assert fps == pytest.approx(30.0)  # 300 frames / 10 s

    def test_fps_computed_from_frame_count_when_rate_missing(self):
        probe = {
            "streams": [{"codec_type": "video", "duration": "8.0"}],
            "format": {"duration": "8.0"},
        }
        _, fps = probe_video_metadata(probe, 200)
        assert fps == pytest.approx(25.0)  # 200 / 8

    def test_missing_duration_raises(self):
        probe = {
            "streams": [{"codec_type": "video", "avg_frame_rate": "25/1"}],
            "format": {},
        }
        with pytest.raises(RuntimeError, match="duration"):
            probe_video_metadata(probe, 0)

    def test_uses_r_frame_rate_when_avg_is_zero(self):
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "4.0",
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "4.0"},
        }
        _, fps = probe_video_metadata(probe, 120)
        assert fps == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# build_scene_sampling_plan
# ---------------------------------------------------------------------------

class TestBuildSceneSamplingPlan:
    def test_zero_frame_count_returns_empty(self):
        result = build_scene_sampling_plan([[0, 100]], 0, 5.0, 2.0)
        assert result == []

    def test_zero_duration_returns_empty(self):
        result = build_scene_sampling_plan([[0, 100]], 100, 0.0, 2.0)
        assert result == []

    def test_single_scene_short_gets_one_sample(self):
        # 25fps * 5s max = 125 frames per sample; scene has 50 frames → 1 sample
        result = build_scene_sampling_plan([[0, 49]], 50, 2.0, 5.0)
        assert len(result) == 1
        scene = result[0]
        assert scene["scene_index"] == 0
        assert scene["start_frame"] == 0
        assert scene["end_frame"] == 49
        assert scene["sample_count"] == 1
        assert len(scene["sampled_frames"]) == 1

    def test_sampled_frame_within_scene_bounds(self):
        result = build_scene_sampling_plan([[10, 59]], 60, 2.0, 5.0)
        for sf in result[0]["sampled_frames"]:
            assert 10 <= sf["frame_id"] <= 59

    def test_sample_index_sequence(self):
        result = build_scene_sampling_plan([[0, 999]], 1000, 10.0, 2.0)
        indices = [sf["sample_index"] for sf in result[0]["sampled_frames"]]
        assert indices == list(range(len(indices)))

    def test_multiple_scenes_correct_count(self):
        scenes = [[0, 49], [50, 99], [100, 149]]
        result = build_scene_sampling_plan(scenes, 150, 6.0, 10.0)
        assert len(result) == 3
        for i, s in enumerate(result):
            assert s["scene_index"] == i
            assert s["start_frame"] == scenes[i][0]
            assert s["end_frame"] == scenes[i][1]

    def test_long_scene_gets_multiple_samples(self):
        # avg_fps = 100fps, max_interval = 2s → max 200 frames/sample
        # scene has 1000 frames → ceil(1000/200) = 5 samples
        result = build_scene_sampling_plan([[0, 999]], 1000, 10.0, 2.0)
        assert result[0]["sample_count"] == 5
        assert len(result[0]["sampled_frames"]) == 5

    def test_each_sample_falls_in_its_segment(self):
        # 30fps * 3s = 90 frames per sample; scene has 300 frames → ceil(300/90) = 4 samples
        result = build_scene_sampling_plan([[0, 299]], 300, 10.0, 3.0)
        frames = [sf["frame_id"] for sf in result[0]["sampled_frames"]]
        # sampled frames must be unique (each segment picks its own middle frame)
        assert len(set(frames)) == len(frames)

    def test_minimum_one_sample_per_scene(self):
        # even a 1-frame scene must produce exactly 1 sample
        result = build_scene_sampling_plan([[5, 5]], 100, 10.0, 1.0)
        assert result[0]["sample_count"] == 1
        assert result[0]["sampled_frames"][0]["frame_id"] == 5
