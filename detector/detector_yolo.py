import torch
import numpy as np

class PlateDetector:
    def __init__(self, model_path, confidence=0.3, device=None, logger=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            self.model.conf = confidence
            self.model.iou = 0.45
            self.model.to(self.device)
            if self.logger:
                self.logger.info(f"YOLOv5 модель успешно загружена ({model_path}), device={self.device}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ошибка загрузки YOLOv5 модели: {e}")
            raise

    def detect(self, image):
        try:
            results = self.model(image)
            detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
            # Фильтруем только "номера" (обычно класс 0, если иначе — подправь условие!)
            boxes = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.model.conf and int(cls) == 0:
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
            if self.logger:
                self.logger.debug(f"YOLOv5 обнаружено объектов: {len(boxes)}")
            return boxes
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ошибка в detect: {e}")
            return []
