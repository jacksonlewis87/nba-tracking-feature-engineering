from local_constants import DATA_PATH
from preprocessing.preprocessing_config import PreprocessingConfig
from preprocessing.preprocessing import preprocess_tracking_data


def do_work():
    config = PreprocessingConfig(
        data_path=DATA_PATH,
        min_event_duration=5,
        target_frame_rate=25.0,
        training_frame_rate=5.0,
        game_ids=None,
    )

    preprocess_tracking_data(config=config)


if __name__ == "__main__":
    do_work()
