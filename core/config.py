from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import yaml

class AppConfig(BaseModel):
    streams: List[str]
    cam_ids: List[int]
    detection_confidence: float
    number_detections_to_send: int
    backend_adress: Optional[str] = None
    path_to_detect_model: str
    path_to_read_model: str
    detector_framework: Optional[str] = "tf"
    align: Optional[bool] = False
    gpu_number: Optional[str] = "0"
    from_top: Optional[int] = 0
    repeat: Optional[int] = 1
    debug: Optional[str] = "0"
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    id_to_char: Optional[Dict[int, str]] = None

def load_config(path: str = "config/config.yaml") -> AppConfig:
    """
    Загрузить конфиг из YAML-файла и провалидировать через pydantic.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        config = AppConfig(**raw_data)
        return config
    except FileNotFoundError:
        raise RuntimeError(f"Config file {path} not found.")
    except ValidationError as ve:
        raise RuntimeError(f"Ошибка валидации config.yaml:\n{ve}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке config.yaml: {e}")

# Пример использования (удобно тестировать отдельно)
if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
