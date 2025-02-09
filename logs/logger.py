import logging
import os
from pathlib import Path
 

def main():
    logging_config(logs_save=False)
    
def logging_config(logs_save=True, amount_samples=4, transform=True, model_type="EncoderDecoder", window_size=[64,64]):
    if logs_save:
        target_dir_path = Path(f"logs")
        target_dir_path.mkdir(parents=True, exist_ok=True)
        files=os.listdir(target_dir_path)
        if transform:
            transform="TR"
        else:
            transform="NO"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s -- [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M::%S",
            filename=f"logs/training_{len(files)+1:03d}_{window_size[0]}x{window_size[1]}_{amount_samples}_{model_type}_{transform}.log",
            #filename=f"logs/test.log",
            filemode='w'
            )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s -- [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M::%S"        
            )    

    logging.info("Logging Configuration Loaded.")



if __name__ == "__main__":
    main()
