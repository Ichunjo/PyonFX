from __future__ import annotations

import pytest

from pyonfx._logging import LogLevel, logger


def test_log_level_values() -> None:
    assert LogLevel.TRACE.value == 5
    assert LogLevel.DEBUG.value == 10
    assert LogLevel.INFO.value == 20
    assert LogLevel.SUCCESS.value == 25
    assert LogLevel.WARNING.value == 30
    assert LogLevel.ERROR.value == 40
    assert LogLevel.CRITICAL.value == 50
    assert LogLevel.USER_WARNING.value == 60
    assert LogLevel.USER_INFO.value == 70


def test_logger_set_level() -> None:
    logger.set_level(LogLevel.DEBUG)
    logger.debug("Debug test message")
    logger.info("Info test message")
    logger.set_level(LogLevel.ERROR)


def test_logger_catch_decorator() -> None:
    @logger.catch
    def failing_func() -> None:
        raise ValueError("Test error")

    # Should catch and return None
    assert failing_func() is None


def test_logger_catch_with_force_exit() -> None:
    @logger.catch(force_exit=True)
    def failing_func_exit() -> None:
        raise ValueError("Fatal error")

    with pytest.raises(SystemExit):
        failing_func_exit()
