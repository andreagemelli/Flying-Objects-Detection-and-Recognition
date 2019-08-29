
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

from model import model
from utils import *

if __name__ == '__main__':

    # DATA LOAD

    if not os.isdir("data"):
        os.mkdir("data")

    if not os.isdir("video_test"):
        os.mkdir("video_test")

    if not os.isdir("weights"):
        os.mkdir("weights")

    X_train_orig, Y_train_orig, X_val_orig, Y_val_orig, X_test_orig, Y_test_orig = load_dataset(src_path='data')
    num_classes = 1

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_val = X_val_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, num_classes)
    Y_val = convert_to_one_hot(Y_val_orig, num_classes)
    Y_test = convert_to_one_hot(Y_test_orig, num_classes)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    # HYPERPARAMS
    img_height, img_width = 40, 40
    batch_size = 300
    lr = 0.0001
    epochs = 50
    weights_path = 'weights/vgg19.h5'

    tb = TensorBoard(log_dir='./logs')
    update_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    # MODEL AND TRAIN

    model = model(img_height, img_width, num_classes, lr)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, Y_val),
              shuffle=True, callbacks=[checkpoint, update_lr, tb])

    # MODEL TEST

    model.load_weights(weights_path)
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
