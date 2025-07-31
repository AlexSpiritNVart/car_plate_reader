import cv2
import numpy as np
from math import atan2, degrees
from scipy.spatial.distance import euclidean
from PIL import Image
from typing import Optional

def get_angle_between_points(p1, p2):
    """Вычисляет угол между двумя точками (в градусах)."""
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return degrees(atan2(y_diff, x_diff))

def align_plate(
    plate: np.ndarray,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Пытается "выпрямить" номер (выравнивание по горизонтали).
    Возвращает новый кадр (np.ndarray) или None, если не удалось.
    """
    img = np.array(plate).copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    try:
        edges = cv2.Canny(gray, 10, 100)
    except Exception as e:
        print(f'Ошибка в Canny: {e}, gray.shape={gray.shape}')
        return None

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(edges, cmap="gray")
        plt.show()

    # Разбиваем картинку пополам по вертикали
    edges_bottom = edges[img.shape[0]//2 :]
    edges_top = edges[:img.shape[0]//2]

    # Ищем линии Хафа на верхней и нижней половине
    lines_bottom = cv2.HoughLinesP(edges_bottom, 1, np.pi/180, 80, minLineLength=30, maxLineGap=50)
    lines_top = cv2.HoughLinesP(edges_top, 1, np.pi/180, 80, minLineLength=30, maxLineGap=50)

    # Берём самую длинную линию вверху и внизу
    try:
        bot_lane = sorted(lines_bottom, key=lambda x: -euclidean((x[0][0],x[0][1]), (x[0][2],x[0][3])))[0][0]
        bot_lane[1] += img.shape[0]//2
        bot_lane[3] += img.shape[0]//2
    except Exception:
        bot_lane = [img.shape[1], img.shape[0]-img.shape[0]//2, img.shape[1], img.shape[0]-img.shape[0]//2]

    try:
        top_lane = sorted(lines_top, key=lambda x: -euclidean((x[0][0],x[0][1]), (x[0][2],x[0][3])))[0][0]
    except Exception:
        top_lane = [0, 0, 0, 0]

    # Считаем углы
    angle_bot = get_angle_between_points((bot_lane[0], bot_lane[1]), (bot_lane[2], bot_lane[3]))
    angle_top = get_angle_between_points((top_lane[0], top_lane[1]), (top_lane[2], top_lane[3]))
    mean_angle = (angle_bot + angle_top) / (2 if angle_bot and angle_top else 1)

    # Вращаем изображение
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), mean_angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # Получаем новые границы
    pts = np.array([[bot_lane[:2], bot_lane[2:], top_lane[:2], top_lane[2:]]])
    pts = np.int32(cv2.transform(pts, M))
    pts[pts < 0] = 0
    bot_border = max(pts[0][0][1], pts[0][1][1])
    top_border = min(pts[0][2][1], pts[0][3][1])

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(img_rot[top_border:bot_border])
        plt.show()

    try:
        result = img_rot[top_border:bot_border]
        return result
    except Exception:
        return plate  # Если что-то пошло не так, возвращаем исходник
