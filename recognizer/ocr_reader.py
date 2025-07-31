import numpy as np
# from recognizer.model_maker import build_ocr_model
from typing import Any, Dict, Optional
from try1 import build_ocr_model

class PlateRecognizer:
    def __init__(
        self,
        model_path: str,
        id_to_char: dict,
        logger=None
    ):
        self.model_path = model_path
        self.id_to_char = id_to_char
        self.logger = logger
        self.model = None
        try:
            self.model = self._load_model()
        except Exception as e:
            if self.logger:
                self.logger.critical(f"Ошибка загрузки OCR-модели: {e}")
            # Можно тут raise, чтобы остановить программу,
            # либо оставить, чтобы объект существовал, но model=None


    def _load_model(self):
        try:
            self.model = build_ocr_model(input_shape=(32, 306, 1), nclass=len(self.id_to_char))

            self.model.load_weights(self.model_path)

            if self.model is None:
                self.logger.critical("OCR-модель не загрузилась!")
            else:
                self.logger.info(f"Модель загружена: {self.model}")

            if self.logger:
                self.logger.info(f"OCR-модель успешно загружена: {self.model_path}")
            return self.model
        except Exception as e:
            if self.logger:
                self.logger.critical(f"Ошибка загрузки OCR-модели: {e}")
            raise

    def read(self, image: np.ndarray) -> str:
        """
        Распознаёт текст с изображения номера.
        На вход — np.ndarray (обрезанное изображение номера).
        На выход — строка (распознанный номер).
        """
        if self.model is None:
            if self.logger:
                self.logger.critical("OCR-модель не инициализирована")
            return ""

        try:
            # Преобразование (resize, normalize) под твою модель
            img = image.copy()
            img = self.preprocess(img)
            # Модель ждёт батч, даже если 1 картинка
            preds = self.model.predict(np.expand_dims(img, axis=0))
            # Декодинг (argmax + перевод индексов в символы)
            text = self.decode(preds)
            if self.logger:
                self.logger.debug(f"OCR результат: {text}")
            return text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ошибка при OCR: {e}")
            return ""

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Готовит изображение для подачи в OCR-модель: resize, grayscale, norm.
        (Настроить параметры под твою сеть!)
        """
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (306, 32))  # (w, h) — поменять на твои размеры!
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # (h, w, 1)
        return img

    def decode(self, preds: np.ndarray) -> str:
        """
        Перевод предсказания модели (логиты или вероятности) в строку.
        """
        if preds is None:
            self.logger.critical("Предсказание preds = None")
            return ""
        # Для CTC-based моделей обычно берём argmax по таймстепам
        best_path = preds.argmax(axis=-1)[0]  # (seq_len,)
        text = ''
        prev_idx = None
        for idx in best_path:
            if idx != prev_idx and idx in self.id_to_char:
                char = self.id_to_char.get(idx, '')
                if char != '*':  # Например, "*" — спецсимвол "пусто"
                    text += char
            prev_idx = idx
        return text
