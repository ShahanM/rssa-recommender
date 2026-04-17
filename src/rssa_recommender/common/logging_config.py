"""Logging configuration for structlog in AWS Lambda environment."""

import logging
import os
import sys

import structlog


def setup_logging():
    """Configures structlog to wrap the standard logging module.

    This should be called once on application startup.
    """
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ and not structlog.is_configured():
        log_level = logging.INFO

        if 'AWS_LAMBDA_RUNTIME_API' not in os.environ:
            log_level = logging.DEBUG
            logging.getLogger('src').setLevel(logging.DEBUG)
            print('INFO: Local debugging logging enabled at DEBUG level.')

        logging.basicConfig(
            level=log_level,
            format='%(message)s',
            stream=sys.stdout,
            force=True,
        )

        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

        structlog.configure(
            processors=processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
            ],
        )

        root_logger = logging.getLogger()

        if root_logger.handlers:
            root_logger.handlers[0].setFormatter(formatter)

        root_logger.setLevel(log_level)

        log = structlog.stdlib.get_logger(__name__)
        log.info(
            'Structured logging initialized',
            function=os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'local-run'),
            region=os.environ.get('AWS_REGION', 'local-region'),
            log_level=logging.getLevelName(log_level),
        )
