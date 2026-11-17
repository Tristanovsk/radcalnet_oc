'''
Logging of radcalnet_oc.

Each module logs on its own logger (``radcalnet_oc.process``, ``radcalnet_oc.lut``...), children of the
package logger ``radcalnet_oc``; importing the package does not configure any handler or level.

- With the Python API, the messages follow the configuration of the calling application, e.g.
  ``logging.basicConfig(level=logging.INFO)``, or ``logging.getLogger('radcalnet_oc').setLevel(logging.WARNING)``
  to silence the package only. Without any configuration, Python prints the warnings and errors on the
  standard error.
- :py:func:`set_logging` is a shortcut that writes the messages of the package on the standard error
  (and optionally in a file); the ``radcalnet_oc`` command uses it (options ``--log_level``, ``--log_file``).

Levels: ``DEBUG`` for the internal steps (loading of the tables, preparation of the LUTs), ``INFO`` for the
progress of a run (inputs, options, timing, output file), ``WARNING`` for the inputs out of range or
replaced by defaults and for the lower-accuracy fallbacks, ``ERROR`` for the failures.
'''

import logging
import sys

PACKAGE_LOGGER = 'radcalnet_oc'
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s | %(message)s'
LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

# marker of the handlers installed by set_logging (replaced on a new call)
_HANDLER_TAG = '_radcalnet_oc_handler'


def set_logging(level='INFO', log_file=None, fmt=LOG_FORMAT, stream=sys.stderr):
    '''
    Write the messages of radcalnet_oc on `stream` (standard error by default) and, optionally, in
    `log_file`.

    The handlers are attached to the package logger ``radcalnet_oc``, which then stops passing its messages
    to the root logger (no duplicates with a handler of the application); a new call replaces the handlers
    of the previous one.

    :param level: minimum level: ``DEBUG``, ``INFO``, ``WARNING`` or ``ERROR`` (name or number)
    :param log_file: file the messages are appended to (in addition to `stream`), or None
    :param fmt: format of the messages (:py:class:`logging.Formatter`)
    :param stream: stream of the messages, or None for the file only
    :return: the package logger
    '''
    if isinstance(level, str):
        if level.upper() not in LEVELS:
            raise ValueError(f'log level must be one of {", ".join(LEVELS)}, not {level!r}')
        level = level.upper()

    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
        logger.removeHandler(handler)
        handler.close()

    handlers = []
    if stream is not None:
        handlers.append(logging.StreamHandler(stream))
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    formatter = logging.Formatter(fmt)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = not handlers
    return logger
