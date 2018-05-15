import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from data import get_data_set
from model import model


test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()
label_names = get_data_set("label_names")


_BATCH_SIZE = 128
_CLASS_SIZE = 10
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


saver = tf.train.Saver()
sess = tf.Session()


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def main():
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    print(correct)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
    print(len(correct))

    visited=[]
    i = 0

    while(i<10):
        c_i = random.randint(0,len(correct))
        if(correct[c_i]==False):
            index = np.asscalar(np.where(test_y[c_i]==1)[0])
            if index not in visited:
                print(index)
                visited.append(index)
                images = test_x[c_i]
                images_reshape = images.reshape((32,32,3))
                ax = plt.subplot2grid((2,5),(i //5 , i % 5))
                ax.set_title(label_names[predicted_class[c_i]])
                plt.imshow(images_reshape)
                i = i+1

                if i >= 10:
                    break
    plt.show()
    plt.suptitle("Misclassified images")

if __name__ == "__main__":
    main()


sess.close()
