import os
import torch

from nba_tracking_data_commons.dto.tracking import Coordinate, Frame, Event
from nba_tracking_data_commons.utils import DataClassEncoder, load_json, list_files_in_directory, write_json

from constants import DataPaths
from preprocessing.preprocessing_config import PreprocessingConfig


def create_coordinate_from_tracking_row(row: list):
    return Coordinate(x=row[2], y=row[3], z=row[4])


def process_tracking_event(tracking_event: dict, game_id: str):
    try:
        event_id = int(tracking_event["eventId"])
        period = None
        game_clock_start = None
        game_clock_end = None
        wall_clock_start = None
        wall_clock_end = None
        teams = {}
        frames = []
        for frame in tracking_event["moments"]:

            # period
            if period is None:
                period = frame[0]

            # clocks
            if game_clock_start is None:
                game_clock_start = frame[2]
            if wall_clock_start is None:
                wall_clock_start = frame[1]
            game_clock_end = frame[2]
            wall_clock_end = frame[1]

            # players
            ball = None
            players = {}
            for player in frame[5]:
                # ball
                if player[0] == -1 or player[1] == -1:
                    if ball is not None:
                        raise ValueError("duplicate ball")

                    ball = create_coordinate_from_tracking_row(row=player)
                else:
                    team_id = player[0]
                    player_id = player[1]
                    if team_id not in teams:
                        if len(teams.keys()) >= 2:
                            raise ValueError("more than two teams event")

                        teams[team_id] = []

                    if player_id not in teams[team_id]:
                        if len(teams[team_id]) >= 5:
                            raise ValueError("more than 5 players on a team within event")

                        teams[team_id] += [player_id]

                    if player_id in players:
                        raise ValueError("duplicate player in frame")

                    players[player_id] = create_coordinate_from_tracking_row(row=player)

            if ball is None:
                raise ValueError("missing ball")

            frames += [
                Frame(
                    ball=ball,
                    players=players,
                )
            ]

        team_ids = list(teams.keys())
        if len(team_ids) != 2:
            raise ValueError("wrong number of teams")

        if len(teams[team_ids[0]]) != 5 or len(teams[team_ids[1]]) != 5:
            raise ValueError("wrong number of players per team")

        return Event(
            game_id=game_id,
            event_id=event_id,
            period=period,
            game_clock_start=game_clock_start,
            game_clock_end=game_clock_end,
            wall_clock_start=wall_clock_start,
            wall_clock_end=wall_clock_end,
            team_0_id=team_ids[0],
            team_1_id=team_ids[1],
            team_0_players=teams[team_ids[0]],
            team_1_players=teams[team_ids[1]],
            frames=frames,
        )

    except ValueError as e:
        print(e)
        return None


def load_game(data_path: str, game_id: str):
    tracking_data = load_json(f"{os.path.join(data_path, DataPaths.RAW.value, game_id)}.json")

    events = []
    for tracking_event in tracking_data["events"]:
        event = process_tracking_event(tracking_event=tracking_event, game_id=game_id)
        if event is not None and (
            len(events) == 0
            or events[-1].game_clock_start != event.game_clock_start
            or events[-1].game_clock_end != event.game_clock_end
        ):
            events += [event]

    return events


def filter_events(events: list[Event], config: PreprocessingConfig):
    filtered_events = []
    for event in events:
        if (
            len(event.frames) >= config.min_event_duration * config.target_frame_rate
            and abs(
                (1000.0 / config.target_frame_rate)
                - ((event.wall_clock_end - event.wall_clock_start) / len(event.frames))
            )
            < 0.8
        ):
            filtered_events += [event]

    return filtered_events


def down_sample_events(events: list[Event], target_frame_rate: float, training_frame_rate: float):
    factor = int(target_frame_rate / training_frame_rate)
    for event in events:
        event.frames = event.frames[::factor]

    return events


def get_player_coordinates_tensor_from_frames(player_id: int, frames: list[Frame]):
    if player_id == -1:
        # ball
        return torch.tensor([frame.ball.to_row() for frame in frames])
    else:
        # player
        return torch.tensor([frame.players[player_id].to_row() for frame in frames])


def convert_event_to_tensors(event: Event):
    try:
        list_of_tensors = [get_player_coordinates_tensor_from_frames(player_id=-1, frames=event.frames)]

        for player_id in event.team_0_players:
            list_of_tensors += [get_player_coordinates_tensor_from_frames(player_id=player_id, frames=event.frames)]

        for player_id in event.team_1_players:
            list_of_tensors += [get_player_coordinates_tensor_from_frames(player_id=player_id, frames=event.frames)]

        return torch.stack(list_of_tensors, dim=0).permute(1, 0, 2)
    except Exception as e:
        print(f"Error in {event.game_id}_{event.event_id}: {e}")
        return None


def get_game_ids(config: PreprocessingConfig):
    if config.game_ids:
        return config.game_ids
    else:
        return list_files_in_directory(path=os.path.join(config.data_path, DataPaths.RAW.value), suffix=".json")


def preprocess_tracking_data(config: PreprocessingConfig):
    game_ids = get_game_ids(config=config)

    for game_id in game_ids:
        events = load_game(data_path=config.data_path, game_id=game_id)

        events = filter_events(events=events, config=config)
        events = down_sample_events(
            events=events,
            target_frame_rate=config.target_frame_rate,
            training_frame_rate=config.training_frame_rate,
        )

        write_json(
            path=os.path.join(config.data_path, DataPaths.PREPROCESSED.value, f"{game_id}.json"),
            json_object=events,
            encoder=DataClassEncoder,
        )


def write_tensors(data_path: str):
    event_files = list_files_in_directory(path=os.path.join(data_path, DataPaths.PREPROCESSED.value), suffix=".json")

    for event_file in event_files:
        event_list = load_json(path=os.path.join(data_path, DataPaths.PREPROCESSED.value, f"{event_file}.json"))
        tensor_list = []
        tensor_size_list = []
        for event_dict in event_list:
            event = Event.from_dict(data=event_dict)
            tensor = convert_event_to_tensors(event=event)
            if tensor is not None:
                tensor_list.append(tensor)
                tensor_size_list.append(tensor.size(0))

        tensor_list = torch.cat(tensor_list, dim=0)
        tensor_size_list = torch.tensor(tensor_size_list)
        torch.save(
            (tensor_list, tensor_size_list), os.path.join(data_path, DataPaths.TENSORS.value, f"{event_file}.pt")
        )
