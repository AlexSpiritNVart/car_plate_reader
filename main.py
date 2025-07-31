import yaml
from core.config import AppConfig
from detector.detector import PlateDetectorTorch
from detector.detector_yolo import PlateDetector
# from recognizer.ocr_reader import PlateRecognizer
from recognizer.easyocr_reader import PlateRecognizerEasy
from core.pipeline import DetectReadPipeliner
# from utils_in_code.align import align_plate
from utils_in_code.image_queue import VideoStreamReader
from sender.telegram_sender import TelegramSender
from utils_in_code.logger import log as logger # Подключаем логгер

def load_config(path='config/config.yaml'):
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфига: {e}")
        raise

def main():
    # 1. Загрузка конфига
    try:
        config = load_config()
    except Exception:
        logger.critical("Не удалось загрузить конфиг. Завершаем работу.")
        return

    # 2. Инициализация компонентов
    try:
        detector = PlateDetector(config.path_to_detect_model, config.detection_confidence, logger=logger)
        # recognizer = PlateRecognizer(config.path_to_read_model, config.id_to_char,logger=logger)
        recognizer = PlateRecognizerEasy()
        sender = TelegramSender(config.telegram_bot_token, config.telegram_chat_id,logger=logger)
        stream_reader = VideoStreamReader(config.streams[0],logger=logger)
        pipeliner = DetectReadPipeliner(
            detector=detector,
            reader=recognizer,
            cam_ids=config.cam_ids,
            detection_confidence=config.detection_confidence,
            num_detect_to_send=config.number_detections_to_send,
            align=config.align,
            logger=logger
        )
    except Exception as e:
        logger.critical(f"Ошибка инициализации компонентов: {e}")
        return

    logger.info('Starting main loop...')
    try:
        while True:
            try:
                status, frame = stream_reader.read()
                if not status or frame is None:
                    continue
            except Exception as e:
                logger.warning(f"Ошибка чтения кадра: {e}")
                continue

            try:
                results = pipeliner.process_frame(frame, config.cam_ids[0])
            except Exception as e:
                logger.error(f"Ошибка обработки кадра в pipeline: {e}")
                continue

            for img, plate, coords, cam_id, ts in results:
                try:
                    sender.send_plate_data(img, plate, coords, cam_id, ts)
                except Exception as e:
                    logger.error(f"Ошибка отправки данных: {e}")
    except KeyboardInterrupt:
        logger.info('Завершение по Ctrl+C')
    except Exception as e:
        logger.critical(f"Неизвестная ошибка: {e}")
    finally:
        stream_reader.stop()
        logger.info('Поток видеочтения остановлен.')


if __name__ == '__main__':
    main()
