from core.group_manager import GroupManager
from utils_in_code.align import align_plate
from typing import Any, List, Tuple, Optional, Callable
import time

class DetectReadPipeliner:
    def __init__(
        self,
        detector: Any,
        reader: Any,
        cam_ids: list,
        sender: Optional[Callable] = None,    # callable: (image, plate, coords, cam_id, ts)
        detection_confidence: float = 0.5,
        num_detect_to_send: int = 2,
        align: bool = False,
        group_time_thresh: float = 10.0,
        dist_thresh: float = 2.5,
        logger: Any = None
    ):
        self.detector = detector
        self.reader = reader
        self.cam_ids = cam_ids
        self.sender = sender   # теперь любой callable: хоть Telegram, хоть backend, хоть запись в БД
        self.detection_confidence = detection_confidence
        self.num_detect_to_send = num_detect_to_send
        self.align = align
        self.logger = logger

        # Группировщик номеров для каждого канала отдельно
        self.group_managers = {cam_id: GroupManager(dist_thresh, group_time_thresh) for cam_id in cam_ids}

    def process_frame(self, frame, cam_id) -> List[Tuple[Any, str, Any, int, str]]:
        """
        Обработка одного кадра: детекция → выравнивание → OCR → группировка → выдача best_plate.
        Возвращает список готовых к отправке (img, plate, coords, cam_id, ts).
        """
        results = []
        try:
            boxes = self.detector.detect(frame)
            if self.logger:
                self.logger.debug(f"Cam {cam_id}: найдено {len(boxes)} объектов.")
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                crop = frame[y1:y2, x1:x2]
                if self.align:
                    crop = align_plate(crop)
                text = self.reader.read(crop)
                coords = (x1, y1, x2, y2)
                self.group_managers[cam_id].add_number(text, crop, coords)
            # Проверяем, есть ли группы для отправки
            to_send = self.group_managers[cam_id].flush_reduce_groups(cam_id, self.num_detect_to_send)
            for img, plate, coords, cam_id, last_time in to_send:
                if self.sender:
                    try:
                        self.sender(img, plate, coords, cam_id, last_time)
                        if self.logger:
                            self.logger.info(f"Отправлен номер: {plate}, камера {cam_id}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Ошибка отправки результата: {e}")
                results.append((img, plate, coords, cam_id, last_time))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ошибка пайплайна на камере {cam_id}: {e}")
        return results

    def process_batch(self, frames: List[Any], cam_ids: List[int]) -> List[List[Tuple[Any, str, Any, int, str]]]:
        """
        Обработка батча кадров (один кадр на каждую камеру).
        Возвращает список списков результатов для каждого кадра/камеры.
        """
        batch_results = []
        for frame, cam_id in zip(frames, cam_ids):
            batch_results.append(self.process_frame(frame, cam_id))
        return batch_results

    def reset(self):
        """
        Сброс всех группировщиков (например, для тестов или ночного рестарта).
        """
        for gm in self.group_managers.values():
            gm.groups.clear()
        if self.logger:
            self.logger.info("Сброшены все группы номеров.")

