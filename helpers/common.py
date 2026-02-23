import logging
from termcolor import colored


def float_list_formatter(num):
    def formatter(float_list):
        return [round(x, num) for x in float_list]
    return formatter


# Define a new log level `OK`
OK_LEVEL = 25
logging.addLevelName(OK_LEVEL, "OK")
def ok(self, message, *args, **kwargs):
    if self.isEnabledFor(OK_LEVEL):
        self._log(OK_LEVEL, message, args, **kwargs)
logging.Logger.ok = ok


class ConsoleFormatter(logging.Formatter):
    """Formatter class of logging for console logging."""
    MESSAGE_STR = "%(message)s (%(filename)s:%(lineno)d)"

    def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, module_name = ""):
        """Initialize the formatter"""
        super().__init__(fmt, datefmt, style, validate, defaults=None)
        if module_name:
            module_name = f"{module_name} "
        head = f"%(asctime)s - [{module_name}%(levelname)s] "
        self.FORMATS = {
            logging.DEBUG: colored(head, "magenta", attrs=["bold"]) + self.MESSAGE_STR,
            logging.INFO: colored(head, "blue", attrs=["bold"]) + self.MESSAGE_STR,
            OK_LEVEL: colored(head, "green", attrs=["bold"]) + self.MESSAGE_STR,
            logging.WARNING: colored(head, "yellow", attrs=["bold"]) + self.MESSAGE_STR,
            logging.ERROR: colored(head, "red", attrs=["bold"]) + self.MESSAGE_STR,
            logging.CRITICAL: colored(head, "red", attrs=["bold", "reverse"]) + self.MESSAGE_STR,
        }

    def format(self, record):
        """Apply custom fomatting on LogRecord object record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger:
    def __init__(self, logger_name="AC-DiT", console_logging_level="INFO", module_name=""):
        """Initialize the logger"""
        self.logger_name = logger_name
        logger = logging.getLogger(self.logger_name)
        if console_logging_level is not None:
            ch = logging.StreamHandler()
            ch.setLevel(console_logging_level)
            console_formatter = ConsoleFormatter(module_name=module_name)
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)
            logger.setLevel(console_logging_level)

    def get_logger(self):
        """Get the logger whose name equals self.logger_name"""
        logger = logging.getLogger(self.logger_name)
        return logger


ACDIT_DEFAULT_LOGGER = CustomLogger().get_logger()
