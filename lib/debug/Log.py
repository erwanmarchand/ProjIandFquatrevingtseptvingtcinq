# -*- coding: utf-8 -*-
"""
    A simple log mechanism styled after PEP 282.

    Inspired by python/cpython log library
    Adapted by Thibault PIANA
"""
import sys
import time
import platform

DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4
FATAL = 5

CSI = "\x1B["
DEBUG_ACTIVATED = True


class Log:
    def __init__(self):
        pass

    @staticmethod
    def _log(level, msg, indentation):
        msg = ("    " * indentation) + msg

        if level not in (DEBUG, INFO, WARN, ERROR, FATAL):
            raise ValueError('%s wrong log level' % str(level))

        stream = sys.stderr if level in (WARN, ERROR, FATAL) else sys.stdout

        if stream.errors == 'strict':
            # emulate backslashreplace error handler
            encoding = stream.encoding
            msg = msg.encode(encoding, "backslashreplace").decode(encoding)

        if platform.system() == "Windows":
            stream.write('[' + str(round(time.clock(), 2)) + ']' + " %s\n" % msg)
        else:
            COLOR = "31m" if level in [WARN, ERROR, FATAL] else "0m"
            COLOR = "34m" if level in [INFO] else "0m"
            COLOR = "32m" if level in [DEBUG] else "0m"

            stream.write(CSI + COLOR + '[' + str(round(time.clock(), 2)) + ']' + CSI + "0m" + " %s\n" % msg)

        stream.flush()

    @staticmethod
    def log(level, msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        Log._log(level, msg, indentation)

    @staticmethod
    def debug(msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        if DEBUG_ACTIVATED:
            Log._log(DEBUG, msg, indentation)

    @staticmethod
    def info(msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        Log._log(INFO, msg, indentation)

    @staticmethod
    def warn(msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        Log._log(WARN, msg, indentation)

    @staticmethod
    def error(msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        Log._log(ERROR, msg, indentation)

    @staticmethod
    def fatal(msg, *args):
        indentation = args[0] if len(args) > 0 else 0
        Log._log(FATAL, msg, indentation)
