import logging

def setup_logger(name: str, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger