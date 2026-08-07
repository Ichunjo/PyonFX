from __future__ import annotations

__all__: list[str] = []

import datetime
import functools
import logging
import sys
from abc import ABC, ABCMeta
from collections.abc import Callable
from enum import IntEnum
from threading import Lock
from typing import Any, ClassVar, NoReturn, overload, override


class LogLevel(IntEnum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    USER_WARNING = 60
    USER_INFO = 70


logging.addLevelName(LogLevel.TRACE, "TRACE")
logging.addLevelName(LogLevel.SUCCESS, "SUCCESS")
logging.addLevelName(LogLevel.USER_WARNING, "USER WARNING")
logging.addLevelName(LogLevel.USER_INFO, "USER INFO")


class PyonFXFormatter(logging.Formatter):
    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).astimezone()
        return ct.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @override
    def format(self, record: logging.LogRecord) -> str:
        is_user = getattr(record, "user", False)
        logger_level = getattr(record, "logger_level", LogLevel.ERROR)
        if is_user and record.levelno >= LogLevel.USER_WARNING and logger_level >= LogLevel.ERROR:
            return record.getMessage()

        message = record.getMessage()
        asctime = self.formatTime(record)
        levelname = record.levelname
        name = record.name
        module = record.module
        func_name = record.funcName
        lineno = record.lineno

        res = f"{asctime} | {levelname:<12} | {name}:{module}:{func_name}:{lineno} - {message}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not res.endswith("\n"):
                res += "\n"
            res += record.exc_text
        return res


class SingletonMeta(ABCMeta):
    _instances: ClassVar[dict[object, Any]] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(ABC, metaclass=SingletonMeta): ...


class Logger(Singleton):
    __slots__ = ("__level", "_handler", "_logger")

    def __init__(self) -> None:
        self.__level: int = LogLevel.ERROR
        self._logger = logging.getLogger("pyonfx")
        self._logger.setLevel(LogLevel.TRACE)
        self._logger.propagate = False
        self._logger.handlers.clear()

        self._handler = logging.StreamHandler(sys.stderr)
        self._handler.setLevel(self.__level)
        self._handler.setFormatter(PyonFXFormatter())
        self._logger.addHandler(self._handler)

    def set_level(self, level: int) -> None:
        self.__level = level
        self._handler.setLevel(level)

    def _log(self, level: int, message: Any, depth: int = 1, user: bool = False) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(
                level,
                str(message),
                stacklevel=depth + 2,
                extra={"user": user, "logger_level": self.__level},
            )

    def trace(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.TRACE, message, depth=depth, user=False)

    def debug(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.DEBUG, message, depth=depth, user=False)

    def info(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.INFO, message, depth=depth, user=False)

    def success(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.SUCCESS, message, depth=depth, user=False)

    def warning(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.WARNING, message, depth=depth, user=False)

    def error(self, message: Any, /) -> NoReturn:
        if sys.exc_info()[0] is not None:
            self._logger.error(
                str(message),
                exc_info=True,  # noqa: LOG014
                stacklevel=2,
                extra={"user": False, "logger_level": self.__level},
            )
        else:
            self._logger.error(str(message), stacklevel=2, extra={"user": False, "logger_level": self.__level})
        sys.exit(1)

    def user_warning(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.USER_WARNING, message, depth=depth, user=True)

    def user_info(self, message: Any, /, depth: int = 1) -> None:
        self._log(LogLevel.USER_INFO, message, depth=depth, user=True)

    @overload
    def catch[**P, R](self, func: Callable[P, R], /) -> Callable[P, R]: ...
    @overload
    def catch[**P, R](self, /, *, force_exit: bool = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
    def catch[**P, R](
        self, func: Callable[P, R] | None = None, /, *, force_exit: bool = False
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator(f: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return f(*args, **kwargs)
                except Exception:
                    self._logger.exception("An exception occurred", stacklevel=2, extra={"user": False, "logger_level": self.__level})
                    if force_exit:
                        sys.exit(1)
                    return None

            return wrapper

        if func is None:
            return decorator
        return decorator(func)


logger: Logger = Logger()
