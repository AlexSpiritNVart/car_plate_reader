import easyocr
import numpy as np

class PlateRecognizerEasy:
    def __init__(self, lang='en'):
        self.reader = easyocr.Reader([lang])

    def read(self, image: np.ndarray) -> str:
        results = self.reader.readtext(image)
        texts = [text for (_, text, prob) in results if prob > 0.5]
        print(f"Распознанный текст{texts}")
        return max(texts, key=len) if texts else ''
