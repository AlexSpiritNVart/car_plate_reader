import requests
import cv2
import numpy as np
from io import BytesIO
from typing import Any, Tuple, Optional
from utils_in_code.logger import log as default_logger

class TelegramSender:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        logger: Optional[Any] = None
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        # use provided logger or fallback to default logger from utils_in_code
        self.logger = logger or default_logger
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    def send_plate_data(
        self,
        image: np.ndarray,
        plate: str,
        coords: Tuple[int, int, int, int],
        cam_id: int,
        ts: str
    ):
        """Отправка фото и текста в Телеграм."""
        try:
            # Кодируем картинку в jpeg
            is_success, buffer = cv2.imencode(".jpg", image)
            if not is_success:
                raise Exception("Ошибка кодирования изображения.")
            img_bytes = BytesIO(buffer.tobytes())
            caption = f"🚘 Plate: {plate}\nКамера: {cam_id}\nКоординаты: {coords}\nВремя: {ts}"
            files = {'photo': ('plate.jpg', img_bytes, 'image/jpeg')}
            data = {'chat_id': self.chat_id, 'caption': caption}
            response = requests.post(self.api_url, data=data, files=files)
            if response.status_code != 200:
                self.logger.error(f"Ошибка отправки в Telegram: {response.text}")
            else:
                self.logger.info(f"Успешно отправлено: {plate}")
        except Exception as e:
            self.logger.error(f"TelegramSender.send_plate_data — ошибка: {e}")
