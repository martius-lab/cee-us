import logging

default_basic_logging_params = dict(
    formatter=dict(
        fmt='%(asctime)s (%(levelname)s) [%(name)s] > %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ),
    level=logging.WARNING,
)
