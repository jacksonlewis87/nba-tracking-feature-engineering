from preprocessing.preprocessing_config import PreprocessingConfig


def test_preprocessing_config():
    data_path = "some-data-path"
    min_event_duration = 1
    target_frame_rate = 2.0
    training_frame_rate = 3.0
    game_ids = ["game_id_0", "game_id_1"]

    result = PreprocessingConfig(
        data_path=data_path,
        min_event_duration=min_event_duration,
        target_frame_rate=target_frame_rate,
        training_frame_rate=training_frame_rate,
        game_ids=game_ids,
    )

    assert result.data_path == data_path
    assert result.min_event_duration == min_event_duration
    assert result.target_frame_rate == target_frame_rate
    assert result.training_frame_rate == training_frame_rate
    assert result.game_ids == game_ids
