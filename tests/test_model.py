import os
import sys
import types
import numpy as np
import pytest

# Ensure package root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub ocr_model_maker to avoid heavy deps
ocr_stub = types.ModuleType("ocr_model_maker")

class DummyModel:
    def load_weights(self, path):
        pass
    def predict(self, batch):
        seq_len = 3
        nclass = 4
        preds = np.zeros((1, seq_len, nclass))
        preds[0, 0, 1] = 1.0
        preds[0, 1, 2] = 1.0
        preds[0, 2, 3] = 1.0
        return preds

def build_ocr_model(input_shape=(32, 306, 1), nclass=4):
    return DummyModel()

ocr_stub.build_ocr_model = build_ocr_model
sys.modules['ocr_model_maker'] = ocr_stub

from recognizer.ocr_reader import PlateRecognizer


def test_plate_recognizer_returns_string():
    id_to_char = {1: 'A', 2: 'B', 3: 'C'}
    pr = PlateRecognizer(model_path="dummy", id_to_char=id_to_char)
    img = np.zeros((32, 306, 3), dtype=np.uint8)
    text = pr.read(img)
    assert isinstance(text, str)
    assert text != ""

