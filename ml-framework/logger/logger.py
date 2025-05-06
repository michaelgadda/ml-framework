import logging 
from pathlib import Path
from colorama import Fore

class stdoutFormatter(logging.Formatter):
    format = '%(asctime)s - %(levelname)s - %(message)s'
    FORMATS = {
        logging.DEBUG: Fore.CYAN + format + Fore.RESET,
        logging.INFO: Fore.WHITE + format + Fore.RESET,
        logging.WARNING: Fore.YELLOW + format + Fore.RESET,
        logging.ERROR: Fore.LIGHTRED_EX + format + Fore.RESET,
        logging.CRITICAL: Fore.RED + format + Fore.RESET
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)
    

class fileFormatter(logging.Formatter):
    format = '%(asctime)s - %(levelname)s - %(message)s'

    def format(self, record):
        formatter = logging.Formatter(format)
        return formatter.format(record) 
    




def create_logger(logger_path: Path):
    logger = logging.getLogger(logger_path.stem)
    #fileHandler = logging.FileHandler(f'{logger_path}', mode='w')
    #fileHandler.setLevel(logging.DEBUG)
    #fileHandler.setFormatter(stdoutFormatter())
    logging.basicConfig(filename=logger_path, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(stdoutFormatter())
    logger.addHandler(streamHandler)
    #logger.addHandler(fileHandler)


    return logger


