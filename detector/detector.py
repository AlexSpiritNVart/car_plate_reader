import torch
import numpy as np
from typing import List, Tuple


from fpnssd.net import FPNSSD512
from fpnssd.box_coder import FPNSSDBoxCoder # Импортируй свой класс нейронки

def remove_module_prefix(state_dict):
    """Удаляет префикс 'module.' из ключей state_dict (после обучения с DataParallel)."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class PlateDetectorTorch:
    def __init__(self, model_path: str, confidence: float = 0.5, device: str = "cuda" if torch.cuda.is_available() else "cpu", logger=None):
        self.model_path = model_path
        self.confidence_threshold = confidence
        self.device = device
        self.logger = logger
        self.model = None
        self._initialize()
        self.box_coder = FPNSSDBoxCoder()

    def _initialize(self):
        try:
            self.logger.info(f"Загружаем PyTorch модель: {self.model_path}")
            self.model = FPNSSD512(num_classes=2)  # Настрой под свой класс
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint['net']
            state_dict = remove_module_prefix(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info("PyTorch PlateDetector инициализирован.")
        except Exception as e:
            self.logger.critical(f"Ошибка загрузки модели: {e}")
            raise

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
               Возвращает: список (x1, y1, x2, y2, conf) для каждого найденного номера
        """
        try:
            img = self.preprocess(image)
            loc_preds, cls_preds = self.model(img)
            loc_preds = loc_preds[0]
            cls_preds = cls_preds[0]
            # Декодируем боксы
            boxes, labels, scores = self.box_coder.decode(loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45)
            results = []
            for box, label, score in zip(boxes, labels, scores):
                # Оставляем только таргетный класс (например, 1 = "plate")
                if label == 1 and score > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    results.append((x1, y1, x2, y2, float(score)))
            return results
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ошибка в detect: {e}")
            return []

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Базовая препроцессинг функция (резайз, нормализация, etc). Настроить под свою модель!"""
        from torchvision import transforms
        img = image[..., :3]  # если вдруг RGBA
        img = np.ascontiguousarray(img)
        # Подбери правильный размер и нормировку!
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # если у тебя другой input, поменяй!
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        img = transform(img).unsqueeze(0).to(self.device)
        return img

    def postprocess(self, outputs, original_shape):
        # outputs — это (loc_preds, cls_preds)
        loc_preds, cls_preds = outputs
        # Только первый в батче (B=1)
        loc_preds = loc_preds[0]
        cls_preds = cls_preds[0]
        # Декодируем
        boxes, labels, scores = self.box_coder.decode(loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45)
        # Приводим координаты к int и фильтруем только нужный класс (например, 1)
        results = []
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.tolist())
                results.append((x1, y1, x2, y2, float(score)))
        return results
