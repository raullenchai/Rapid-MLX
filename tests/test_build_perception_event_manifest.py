import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/research/build_perception_event_manifest.py"
SPEC = importlib.util.spec_from_file_location("event_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_preserves_boundaries_and_roles(tmp_path):
    annotations = {
        "video_1": {
            "metadata": {"video_id": "video_1", "num_frames": 11},
            "action_localisation": [{
                "id": 4, "label_id": 21,
                "label": "Lifting something and placing it back down",
                "parent_objects": [2], "timestamps": [100, 200], "frame_ids": [1, 10],
            }],
        }
    }
    prior = [{"id": "video_1", "split": "train"}, {"id": "video_2", "split": "dev"}]
    result = MODULE.build(annotations, prior, [], [tmp_path])
    assert result["cases"][0]["segments"][0]["timestamps"] == [100, 200]
    assert result["cases"][0]["role"] == "train"
    assert result["statistics"]["by_role"]["dev"]["videos_without_action_annotations"] == 1
    assert result["statistics"]["pickup_putdown_segments_by_role"]["train"] == {
        "compound_pickup_putdown": 1
    }
    assert result["statistics"]["pickup_putdown_videos_by_role"]["train"] == {
        "compound_pickup_putdown": 1
    }
    assert "not generated" in result["boundary_contract"]["negative_windows"]
