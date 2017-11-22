# -*- coding: utf-8 -*-
"""
    A simple log mechanism styled after PEP 282.

    Inspired by python/cpython log library
    Adapted by Thibault PIANA
"""

DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4
FATAL = 5

CSI = "\x1B["

import sys
import time


class Log:
    @staticmethod
    def _log(level, msg, args):
        if level not in (DEBUG, INFO, WARN, ERROR, FATAL):
            raise ValueError('%s wrong log level' % str(level))

        COLOR = "31m" if level in [WARN, ERROR, FATAL] else "0m"
        COLOR = "34m" if level in [INFO] else "0m"
        COLOR = "32m" if level in [DEBUG] else "0m"

        msg = msg % args if args else msg
        stream = sys.stderr if level in (WARN, ERROR, FATAL) else sys.stdout

        if stream.errors == 'strict':
            # emulate backslashreplace error handler
            encoding = stream.encoding
            msg = msg.encode(encoding, "backslashreplace").decode(encoding)

        stream.write(CSI + COLOR + '[' + str(round(time.clock(), 2)) + ']' + CSI + "0m" + " %s\n" % msg)
        stream.flush()

    @staticmethod
    def log(level, msg, *args):
        Log._log(level, msg, args)

    @staticmethod
    def debug(msg, *args):
        Log._log(DEBUG, msg, args)

    @staticmethod
    def info(msg, *args):
        Log._log(INFO, msg, args)

    @staticmethod
    def warn(msg, *args):
        Log._log(WARN, msg, args)

    @staticmethod
    def error(msg, *args):
        Log._log(ERROR, msg, args)

    @staticmethod
    def fatal(msg, *args):
        Log._log(FATAL, msg, args)
