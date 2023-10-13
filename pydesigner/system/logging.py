

"""Loggin configuration for PyDesigner"""
import logging

# Define ANSI text color codes
ANSI_COLORS = {
    "maroon": "\x1b[38;5;1m",
    "green": "\x1b[38;5;2m",
    "purple": "\x1b[38;5;5m",
    "silver": "\x1b[38;5;7m",
    "grey": "\x1b[38;5;8m",
    "red": "\x1b[38;5;9m",
    "orange": "\x1b[38;5;202m",
    "lime": "\x1b[38;5;10m",
    "yellow": "\x1b[38;5;11m",
    "blue": "\x1b[38;5;12m",
    "white": "\x1b[38;5;15m",
}

# Define ANSI text format codes
ANSI_FORMAT = {"bold": "\x1b[1m", "underline": "\x1b[4m", "reverse": "\x1b[4m"}

# Define ANSI reset code to reset text formatting
ANSI_RESET = "\x1b[0m\n"

# Define console output format
LOG_FORMAT = "%(name)s (%(filename)s:%(lineno)d) %(levelname)s - %(message)s"


class ColorLogging(logging.Formatter):
    """
    A logging class to define colors used in printing of various messages. Parse
    this class into logging's StreamHandler()'s setFormatter method.

    Example:
        Enable color logging by parsing this class into setFormatter method

        .. code-block:: python
            # Configure logging
            log = logging.getLogger(__name__)
            log.setLevel(logging.DEBUG)
            # Create a console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # Parse ColorLogging as an input
            console_handler.setFormatter(ColorLogging())  # parse class here
            log.addHandler(ch)
    """

    LOGFORMAT = {
        logging.DEBUG: ANSI_COLORS["grey"] + LOG_FORMAT + ANSI_RESET,
        logging.INFO: ANSI_COLORS["silver"] + LOG_FORMAT + ANSI_RESET,
        logging.WARNING: ANSI_COLORS["orange"] + LOG_FORMAT + ANSI_RESET,
        logging.ERROR: ANSI_COLORS["red"] + LOG_FORMAT + ANSI_RESET,
        logging.CRITICAL: ANSI_FORMAT["bold"]
        + ANSI_COLORS["red"]
        + LOG_FORMAT
        + ANSI_RESET,
    }

    def format(self, record):
        log_fmt = self.LOGFORMAT.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
