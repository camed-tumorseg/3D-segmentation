import config as conf

def print2file(text):
    print(text)
    
    if conf.output_file is not None:
        with open(conf.output_file, "a") as f:
            f.write(f"{text}\n")
