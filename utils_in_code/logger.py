import datetime

class Logger:
    def _write(self, message: str, level: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("logs.txt", "a") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
        print(f"[{timestamp}] [{level}] {message}\n")

    def info(self, message: str):
        self._write(message, "INFO")

    def warning(self, message: str):
        self._write(message, "WARNING")

    def error(self, message: str):
        self._write(message, "ERROR")

    def critical(self, message: str):
        self._write(message, "CRITICAL")

    def debug(self, message: str):
        self._write(message, "DEBUG")
# Экземпляр логгера
log = Logger()