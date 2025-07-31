import cv2
import queue
import threading
import time
from typing import Optional

class VideoStreamReader:
    """
    Читает видео (RTSP или файл) в отдельном потоке и кладёт кадры в очередь.
    Использовать .read() чтобы получать новые кадры.
    """
    def __init__(self, stream_url: str, max_queue: int = 5, sleep: float = 0.02, logger=None):
        self.stream_url = stream_url
        self.queue = queue.Queue(max_queue)
        self.sleep = sleep
        self.stopped = False
        self.logger = logger
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        # cap = cv2.VideoCapture(self.stream_url)
        cap = cv2.VideoCapture('C:\\Users\\Alexander\\PycharmProjects\\car_plate_reader_best_ever\\video_2025-06-22_19-51-47.mp4')

        if not cap.isOpened():
            if self.logger:
                self.logger.error(f"Не удалось открыть поток: {self.stream_url}")
            return
        if self.logger:
            self.logger.info(f"Старт чтения потока: {self.stream_url}")
        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(self.stream_url)
                continue
            try:
                if not self.queue.full():
                    self.queue.put(frame, timeout=1)
                else:
                    # Перезаписываем старый кадр если очередь полная
                    self.queue.get_nowait()
                    self.queue.put(frame, timeout=1)
            except queue.Full:
                pass
            time.sleep(self.sleep)
        cap.release()
        if self.logger:
            self.logger.info(f"Остановлен поток: {self.stream_url}")

    def read(self) -> (bool, Optional[any]):
        """Возвращает (True, кадр) или (False, None) если очередь пуста."""
        try:
            frame = self.queue.get(timeout=0.2)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        """Останавливает чтение и поток."""
        self.stopped = True
        self.thread.join(timeout=2)
        if self.logger:
            self.logger.info("Чтение потока остановлено.")
