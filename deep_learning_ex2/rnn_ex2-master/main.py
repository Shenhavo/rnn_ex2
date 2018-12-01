
from os import getcwd
import ptb.reader as reader
def main():
    print("hola mundo")
    # import
    import tensorflow as tf
    with tf.device('/gpu:1'):
      current_path = getcwd() + "\Dataset"
      train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(current_path)
      print ("el fin")

if __name__ == "__main__":
  main()