import numpy as np

from omnirt.models.quicktalk.runtime_v2 import QuickTalkRebuild


def _worker() -> QuickTalkRebuild:
    worker = object.__new__(QuickTalkRebuild)
    worker.batch_size = 99

    def prepare_batch(img_batch, rep_batch, frame_batch, coords_batch, affines_batch):
        return img_batch, rep_batch, frame_batch, coords_batch, affines_batch

    worker._prepare_batch = prepare_batch
    return worker


def _frame(marker: int) -> np.ndarray:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = marker
    return frame


def _face(marker: int) -> np.ndarray:
    face = np.zeros((2, 2, 3), dtype=np.uint8)
    face[:, :, 2] = marker
    return face


def _face_results(count: int):
    return [(_face(index), [0, 0, 2, 2], np.eye(2, 3, dtype=np.float32)) for index in range(count)]


def test_quicktalk_datagen_loops_short_source_video_forward() -> None:
    frames = [_frame(index) for index in range(3)]
    reps = [np.asarray([index], dtype=np.float32) for index in range(7)]

    _, _, selected_frames, _, _ = next(_worker().datagen(frames, reps, _face_results(len(frames))))

    assert [int(frame[0, 0, 0]) for frame in selected_frames] == [0, 1, 2, 0, 1, 2, 0]


def test_quicktalk_datagen_trims_long_source_video_to_audio_frames() -> None:
    frames = [_frame(index) for index in range(5)]
    reps = [np.asarray([index], dtype=np.float32) for index in range(2)]

    _, _, selected_frames, _, _ = next(_worker().datagen(frames, reps, _face_results(len(reps))))

    assert [int(frame[0, 0, 0]) for frame in selected_frames] == [0, 1]
