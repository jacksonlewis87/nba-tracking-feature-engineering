from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PreprocessingConfig:
    data_path: str
    min_event_duration: int
    target_frame_rate: float
    training_frame_rate: float
    game_ids: Optional[List[str]] = None
