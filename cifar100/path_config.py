import configparser
import os

cfd = os.path.dirname(os.path.realpath(__file__))
ini_path = os.path.join(os.path.dirname(cfd), "config.ini")

config = configparser.ConfigParser()
config.read(ini_path)
data_dir = config["DATASETS"]["Cifar100"]
