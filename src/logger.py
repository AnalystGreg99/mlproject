import logging
import os
import sys
from datetime import datetime

# Create a log file name based on the current datetime
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
print(f"Log file path: {LOG_FILE_PATH}") #Debug print

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)

# Example logging to test 
logger = logging.getLogger(__name__) 
logger.info("Logging is set up correctly!")
if __name__=="__main__":

    try:
        a=1/0

    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e,sys)