import importlib
import io
import logging

import pytest

import radcalnet_oc
from radcalnet_oc import set_logging
from radcalnet_oc.log import PACKAGE_LOGGER


@pytest.fixture
def package_logger():
    '''Package logger restored after the test (handlers, level, propagation).'''
    logger = logging.getLogger(PACKAGE_LOGGER)
    state = (list(logger.handlers), logger.level, logger.propagate)
    yield logger
    for handler in logger.handlers:
        if handler not in state[0]:
            handler.close()
    logger.handlers[:], logger.level, logger.propagate = state


def test_import_leaves_root_logger():
    # importing the package does not configure the logging of the application
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    root.setLevel(logging.ERROR)
    try:
        importlib.reload(radcalnet_oc)
        assert root.level == logging.ERROR
        assert root.handlers == handlers
    finally:
        root.setLevel(level)


def test_module_loggers_are_children_of_the_package():
    from radcalnet_oc import process, lut, config
    for module in (process, lut, config):
        assert module.logger.name.startswith(PACKAGE_LOGGER + '.')


def test_set_logging_level_and_format(package_logger):
    stream = io.StringIO()
    set_logging('warning', stream=stream)
    child = logging.getLogger(PACKAGE_LOGGER + '.lut')
    child.info('hidden')
    child.warning('shown')
    text = stream.getvalue()
    assert 'hidden' not in text
    assert 'WARNING radcalnet_oc.lut | shown' in text
    assert package_logger.propagate is False


def test_set_logging_replaces_its_handlers(package_logger, tmp_path):
    log_file = tmp_path / 'run.log'
    set_logging('INFO', stream=io.StringIO())
    set_logging('DEBUG', log_file=log_file, stream=io.StringIO())
    ours = [h for h in package_logger.handlers if getattr(h, '_radcalnet_oc_handler', False)]
    assert len(ours) == 2
    logging.getLogger(PACKAGE_LOGGER + '.process').debug('step')
    for handler in ours:
        handler.flush()
    assert 'DEBUG radcalnet_oc.process | step' in log_file.read_text()


def test_set_logging_rejects_unknown_level(package_logger):
    with pytest.raises(ValueError):
        set_logging('verbose')


def test_cli_rejects_unknown_level(package_logger, tmp_path):
    from radcalnet_oc import run
    assert run.main([str(tmp_path / 'missing.nc'), '--log_level', 'loud']) == 1
