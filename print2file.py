import config as conf

def print2file(text):
    print(text)
    
    if conf.log_file is not None:
        with open(conf.log_file, "a") as f:
            f.write(f"{text}\n")
