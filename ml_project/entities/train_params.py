
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=12314)
    criterion: str = field(default="entropy")
    n_estimators: int = field(default=100)
    C: float = field(default=1)

