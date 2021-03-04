import sys
import os
sys.path.append(os.path.abspath('..'))

import config as conf

def print2file(text):
    print(text)
    
    if conf.log_file is not None:
        with open(conf.log_file, "a") as f:
            f.write(f"{text}\n")
