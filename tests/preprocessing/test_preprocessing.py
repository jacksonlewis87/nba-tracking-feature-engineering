import pytest
import torch
from unittest.mock import Mock, call, patch

from nba_tracking_data_commons.dto.tracking import Event, Frame

from constants import DataPaths
from preprocessing.preprocessing import (
    create_coordinate_from_tracking_row,
    process_tracking_event,
    load_game,
    filter_events,
    down_sample_events,
    get_player_coordinates_tensor_from_frames,
    convert_event_to_tensors,
    get_game_ids,
    preprocess_tracking_data,
    write_tensors,
)
from preprocessing.preprocessing_config import PreprocessingConfig


@patch("preprocessing.preprocessing.Coordinate")
def test_create_coordinate_from_tracking_row(mock_coordinate):
    # Setup
    row = [0, 0, 1.0, 2.0, 3.0]
    expected_x, expected_y, expected_z = 1.0, 2.0, 3.0

    # Mock the Coordinate instance
    mock_coordinate_instance = Mock()
    mock_coordinate.return_value = mock_coordinate_instance

    # Call the function under test
    result = create_coordinate_from_tracking_row(row)

    # Assertions
    mock_coordinate.assert_called_once_with(x=expected_x, y=expected_y, z=expected_z)
    assert result == mock_coordinate_instance


@patch("preprocessing.preprocessing.create_coordinate_from_tracking_row")
@patch("preprocessing.preprocessing.Frame")
def test_process_tracking_event(mock_frame, mock_create_coordinate_from_tracking_row):
    # Setup
    tracking_event = {
        "eventId": "123",
        "moments": [
            [
                1,
                5000,
                1000,
                None,
                None,
                [
                    [-1, -1, 0, 0, 0, 0],  # Ball
                    [0, 1, 1, 0, 0, 0],  # Player 1 from Team 0
                    [0, 2, 2, 0, 0, 0],  # Player 2 from Team 0
                    [0, 3, 2, 0, 0, 0],  # Player 3 from Team 0
                    [0, 4, 2, 0, 0, 0],  # Player 4 from Team 0
                    [0, 5, 2, 0, 0, 0],  # Player 5 from Team 0
                    [1, 6, 1, 0, 0, 0],  # Player 1 from Team 1
                    [1, 7, 2, 0, 0, 0],  # Player 2 from Team 1
                    [1, 8, 2, 0, 0, 0],  # Player 3 from Team 1
                    [1, 9, 2, 0, 0, 0],  # Player 4 from Team 1
                    [1, 10, 2, 0, 0, 0],  # Player 5 from Team 1
                ],
            ],
            [
                1,
                6000,
                2000,
                None,
                None,
                [
                    [-1, -1, 0, 0, 0, 0],  # Ball
                    [0, 1, 1, 0, 0, 0],  # Player 1 from Team 0
                    [0, 2, 2, 0, 0, 0],  # Player 2 from Team 0
                    [0, 3, 2, 0, 0, 0],  # Player 3 from Team 0
                    [0, 4, 2, 0, 0, 0],  # Player 4 from Team 0
                    [0, 5, 2, 0, 0, 0],  # Player 5 from Team 0
                    [1, 6, 1, 0, 0, 0],  # Player 1 from Team 1
                    [1, 7, 2, 0, 0, 0],  # Player 2 from Team 1
                    [1, 8, 2, 0, 0, 0],  # Player 3 from Team 1
                    [1, 9, 2, 0, 0, 0],  # Player 4 from Team 1
                    [1, 10, 2, 0, 0, 0],  # Player 5 from Team 1
                ],
            ],
        ],
    }

    # Mock objects
    mock_event_instance = Mock(spec=Event)
    mock_frame_instance_0 = Mock(spec=Frame)
    mock_frame_instance_1 = Mock(spec=Frame)

    # Configure mocks
    mock_create_coordinate_from_tracking_row.return_value = Mock()
    mock_frame.side_effect = [mock_frame_instance_0, mock_frame_instance_1]

    # Expected Event object
    expected_event = Event(
        game_id="game_id",
        event_id=123,
        period=1,
        game_clock_start=1000,
        game_clock_end=2000,
        wall_clock_start=5000,
        wall_clock_end=6000,
        team_0_id=0,
        team_1_id=1,
        team_0_players=[1, 2, 3, 4, 5],
        team_1_players=[6, 7, 8, 9, 10],
        frames=[mock_frame_instance_0, mock_frame_instance_1],
    )

    # Call the function under test
    result_event = process_tracking_event(tracking_event=tracking_event, game_id="game_id")

    assert result_event == expected_event
    mock_create_coordinate_from_tracking_row.assert_has_calls(
        [
            call(row=[-1, -1, 0, 0, 0, 0]),  # Ball
            call(row=[0, 1, 1, 0, 0, 0]),  # Player 1 from Team 0
            call(row=[0, 2, 2, 0, 0, 0]),  # Player 2 from Team 0
            call(row=[0, 3, 2, 0, 0, 0]),  # Player 3 from Team 0
            call(row=[0, 4, 2, 0, 0, 0]),  # Player 4 from Team 0
            call(row=[0, 5, 2, 0, 0, 0]),  # Player 5 from Team 0
            call(row=[1, 6, 1, 0, 0, 0]),  # Player 1 from Team 1
            call(row=[1, 7, 2, 0, 0, 0]),  # Player 2 from Team 1
            call(row=[1, 8, 2, 0, 0, 0]),  # Player 3 from Team 1
            call(row=[1, 9, 2, 0, 0, 0]),  # Player 4 from Team 1
            call(row=[1, 10, 2, 0, 0, 0]),  # Player 5 from Team 1
        ]
    )


@patch("preprocessing.preprocessing.create_coordinate_from_tracking_row")
@patch("preprocessing.preprocessing.Frame")
@patch("preprocessing.preprocessing.Event")
def test_process_tracking_event_invalid_data(mock_event, mock_frame, mock_create_coordinate_from_tracking_row):
    tracking_event = {
        "eventId": "123",
        "moments": [
            [
                1,
                5000,
                1000,
                None,
                None,
                [
                    [0, 1, 1, 0, 0, 0],  # Player 1 from Team 0 (no ball)
                    [0, 2, 2, 0, 0, 0],  # Player 2 from Team 0
                    [1, 1, 1, 0, 0, 0],  # Player 1 from Team 1
                    [1, 2, 2, 0, 0, 0],  # Player 2 from Team 1
                ],
            ],
        ],
    }

    result_event = process_tracking_event(tracking_event, "game_id")

    assert result_event is None


@patch("preprocessing.preprocessing.load_json")
@patch("preprocessing.preprocessing.process_tracking_event")
@patch("preprocessing.preprocessing.os.path.join")
def test_load_game(mock_path_join, mock_process_tracking_event, mock_load_json):
    mock_path_join.side_effect = lambda *args: "/".join(args)

    game_id = "game123"
    data_path = "test_data_path"

    mock_tracking_data = {
        "events": [{"some_key": "some_value_1"}, {"some_key": "some_value_2"}, {"some_key": "some_value_3"}]
    }

    mock_load_json.return_value = mock_tracking_data

    event1 = Mock()
    event2 = Mock()
    event3 = Mock()

    event1.game_clock_start = 1000
    event1.game_clock_end = 2000

    event2.game_clock_start = 1000
    event2.game_clock_end = 2000

    event3.game_clock_start = 2000
    event3.game_clock_end = 3000

    mock_process_tracking_event.side_effect = [event1, event2, event3]
    expected_events = [event1, event3]

    events = load_game(data_path, game_id)

    assert events == expected_events
    mock_load_json.assert_called_once_with(f"{data_path}/{DataPaths.RAW.value}/{game_id}.json")
    mock_process_tracking_event.assert_has_calls(
        calls=[
            call(tracking_event={"some_key": "some_value_1"}, game_id=game_id),
            call(tracking_event={"some_key": "some_value_2"}, game_id=game_id),
            call(tracking_event={"some_key": "some_value_3"}, game_id=game_id),
        ]
    )


@patch("preprocessing.preprocessing.Event")
@patch("preprocessing.preprocessing.PreprocessingConfig")
def test_filter_events(mock_config, mock_event):
    mock_config_instance = Mock(spec=PreprocessingConfig)
    mock_config_instance.min_event_duration = 1
    mock_config_instance.target_frame_rate = 50

    mock_event1 = Mock(spec=Event)
    mock_event2 = Mock(spec=Event)
    mock_event3 = Mock(spec=Event)

    mock_event1.frames = [Mock()] * 50  # 50 frames
    mock_event1.wall_clock_start = 1000
    mock_event1.wall_clock_end = 2000

    mock_event2.frames = [Mock()] * 49  # 49 frames
    mock_event2.wall_clock_start = 1000
    mock_event2.wall_clock_end = 2000

    mock_event3.frames = [Mock()] * 60  # 60 frames
    mock_event3.wall_clock_start = 1000
    mock_event3.wall_clock_end = 2247

    events = [mock_event1, mock_event2, mock_event3]
    expected_filtered_events = [mock_event1, mock_event3]

    filtered_events = filter_events(events, mock_config_instance)

    assert filtered_events == expected_filtered_events


@patch("preprocessing.preprocessing.Event")
def test_down_sample_events(mock_event):
    target_frame_rate = 30.0
    training_frame_rate = 15.0
    mock_event1 = Mock()
    mock_event1.frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mock_event2 = Mock()
    mock_event2.frames = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    events = [mock_event1, mock_event2]

    down_sample_events(events=events, target_frame_rate=target_frame_rate, training_frame_rate=training_frame_rate)

    expected_frames1 = [0, 2, 4, 6, 8]
    expected_frames2 = [10, 12, 14, 16, 18]
    assert mock_event1.frames == expected_frames1
    assert mock_event2.frames == expected_frames2


@patch("preprocessing.preprocessing.Frame")
def test_get_player_coordinates_tensor_from_frames(mock_frame):
    player_id = 1
    mock_frame1 = Mock()
    mock_frame2 = Mock()
    mock_ball1 = Mock()
    mock_ball2 = Mock()
    mock_player1 = Mock()
    mock_player2 = Mock()
    mock_frame1.ball = mock_ball1
    mock_frame2.ball = mock_ball2
    mock_frame1.players = {player_id: mock_player1}
    mock_frame2.players = {player_id: mock_player2}
    mock_ball1.to_row.return_value = [1, 2, 3]
    mock_ball2.to_row.return_value = [4, 5, 6]
    mock_player1.to_row.return_value = [7, 8, 9]
    mock_player2.to_row.return_value = [10, 11, 12]
    frames = [mock_frame1, mock_frame2]

    # Test for ball (player_id = -1)
    tensor_ball = get_player_coordinates_tensor_from_frames(-1, frames)
    expected_tensor_ball = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.equal(tensor_ball, expected_tensor_ball)

    # Test for player
    tensor_player = get_player_coordinates_tensor_from_frames(player_id, frames)
    expected_tensor_player = torch.tensor([[7, 8, 9], [10, 11, 12]])
    assert torch.equal(tensor_player, expected_tensor_player)


@patch("preprocessing.preprocessing.get_player_coordinates_tensor_from_frames")
@patch("preprocessing.preprocessing.Event")
def test_convert_event_to_tensors(mock_event, mock_get_player_coordinates_tensor_from_frames):
    mock_event_instance = Mock()
    mock_event_instance.frames = [Mock(), Mock()]
    mock_event_instance.team_0_players = [0, 1]
    mock_event_instance.team_1_players = [2, 3]
    tensor_ball = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor_player_0 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    tensor_player_1 = torch.tensor([[13, 14, 15], [16, 17, 18]])
    tensor_player_2 = torch.tensor([[19, 20, 21], [22, 23, 24]])
    tensor_player_3 = torch.tensor([[25, 26, 27], [28, 29, 30]])
    mock_get_player_coordinates_tensor_from_frames.side_effect = [
        tensor_ball,
        tensor_player_0,
        tensor_player_1,
        tensor_player_2,
        tensor_player_3,
    ]
    expected_tensor = torch.stack(
        [tensor_ball, tensor_player_0, tensor_player_1, tensor_player_2, tensor_player_3], dim=0
    ).permute(1, 0, 2)

    result_tensor = convert_event_to_tensors(mock_event_instance)

    assert torch.equal(result_tensor, expected_tensor)


@pytest.mark.parametrize("game_ids", [["game_id_0", "game_id_1"], None])
@patch("preprocessing.preprocessing.list_files_in_directory")
@patch("preprocessing.preprocessing.os.path.join")
def test_get_game_ids(mock_os_path_join, mock_list_files_in_directory, game_ids):
    config = Mock()
    config.game_ids = game_ids

    result = get_game_ids(config=config)

    if game_ids:
        assert result == game_ids
    else:
        mock_os_path_join.assert_called_once_with(config.data_path, DataPaths.RAW.value)
        mock_list_files_in_directory.assert_called_once_with(path=mock_os_path_join.return_value, suffix=".json")
        assert result == mock_list_files_in_directory.return_value


@patch("preprocessing.preprocessing.DataClassEncoder")
@patch("preprocessing.preprocessing.get_game_ids")
@patch("preprocessing.preprocessing.load_game")
@patch("preprocessing.preprocessing.filter_events")
@patch("preprocessing.preprocessing.down_sample_events")
@patch("preprocessing.preprocessing.write_json")
@patch("preprocessing.preprocessing.os.path.join")
def test_preprocess_tracking_data(
    mock_path_join,
    mock_write_json,
    mock_down_sample_events,
    mock_filter_events,
    mock_load_game,
    mock_get_game_ids,
    mock_dataclass_encoder,
):
    config = Mock(spec=PreprocessingConfig)
    config.data_path = "test_data_path"
    config.target_frame_rate = 30
    config.training_frame_rate = 60
    mock_get_game_ids.return_value = ["game1", "game2"]

    preprocess_tracking_data(config)

    mock_get_game_ids.assert_called_once_with(config=config)
    mock_load_game.assert_has_calls(
        calls=[
            call(data_path=config.data_path, game_id="game1"),
            call(data_path=config.data_path, game_id="game2"),
        ]
    )
    mock_filter_events.assert_has_calls(
        calls=[
            call(events=mock_load_game.return_value, config=config),
            call(events=mock_load_game.return_value, config=config),
        ]
    )
    mock_down_sample_events.assert_has_calls(
        calls=[
            call(
                events=mock_filter_events.return_value,
                target_frame_rate=config.target_frame_rate,
                training_frame_rate=config.training_frame_rate,
            ),
            call(
                events=mock_filter_events.return_value,
                target_frame_rate=config.target_frame_rate,
                training_frame_rate=config.training_frame_rate,
            ),
        ]
    )
    mock_path_join.assert_has_calls(
        calls=[
            call(config.data_path, DataPaths.PREPROCESSED.value, "game1.json"),
            call(config.data_path, DataPaths.PREPROCESSED.value, "game2.json"),
        ]
    )
    mock_write_json.assert_has_calls(
        calls=[
            call(
                path=mock_path_join.return_value,
                json_object=mock_down_sample_events.return_value,
                encoder=mock_dataclass_encoder,
            ),
            call(
                path=mock_path_join.return_value,
                json_object=mock_down_sample_events.return_value,
                encoder=mock_dataclass_encoder,
            ),
        ]
    )


@patch("preprocessing.preprocessing.list_files_in_directory")
@patch("preprocessing.preprocessing.load_json")
@patch("preprocessing.preprocessing.convert_event_to_tensors")
@patch("preprocessing.preprocessing.torch")
@patch("preprocessing.preprocessing.os.path.join")
@patch("preprocessing.preprocessing.Event")
def test_write_tensors(
    mock_event, mock_path_join, mock_torch, mock_convert_event_to_tensors, mock_load_json, mock_list_files_in_directory
):
    data_path = "test_data_path"
    mock_list_files_in_directory.return_value = ["file1", "file2"]
    mock_event_list1 = [{"event": "event_data1"}, {"event": "event_data2"}, {"event": "event_data3"}]
    mock_event_list2 = [{"event": "event_data4"}]
    mock_load_json.side_effect = [mock_event_list1, mock_event_list2]
    mock_tensor1 = Mock()
    mock_tensor1.size.return_value = 10
    mock_tensor2 = Mock()
    mock_tensor2.size.return_value = 20
    mock_convert_event_to_tensors.side_effect = [mock_tensor1, None, mock_tensor2, mock_tensor1]

    write_tensors(data_path=data_path)

    mock_path_join.assert_has_calls(
        calls=[
            call(data_path, DataPaths.PREPROCESSED.value),
            call(data_path, DataPaths.PREPROCESSED.value, "file1.json"),
            call(data_path, DataPaths.TENSORS.value, "file1.pt"),
            call(data_path, DataPaths.PREPROCESSED.value, "file2.json"),
            call(data_path, DataPaths.TENSORS.value, "file2.pt"),
        ]
    )
    mock_list_files_in_directory.assert_called_once_with(path=mock_path_join.return_value, suffix=".json")
    mock_load_json.assert_has_calls(
        calls=[
            call(path=mock_path_join.return_value),
            call(path=mock_path_join.return_value),
        ]
    )
    mock_event.from_dict.assert_has_calls(
        calls=[
            call(data={"event": "event_data1"}),
            call(data={"event": "event_data2"}),
        ]
    )
    mock_convert_event_to_tensors.assert_has_calls(
        calls=[
            call(event=mock_event.from_dict.return_value),
            call(event=mock_event.from_dict.return_value),
        ]
    )
    mock_torch.cat.assert_has_calls(
        calls=[
            call([mock_tensor1, mock_tensor2], dim=0),
            call([mock_tensor1], dim=0),
        ]
    )
    mock_torch.tensor.assert_has_calls(calls=[call([10, 20]), call([10])])
    mock_torch.save.assert_has_calls(
        calls=[
            call((mock_torch.cat.return_value, mock_torch.tensor.return_value), mock_path_join.return_value),
            call((mock_torch.cat.return_value, mock_torch.tensor.return_value), mock_path_join.return_value),
        ]
    )
