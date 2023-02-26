from keras.callbacks import Callback

class CustomLogger(Callback):
    def __init__(self, every=100, logpath=""):
        self.logpath = logpath
        self.every = every

    def on_epoch_end(self, epoch, logs=[dict, None]):
        if(int(epoch) % self.every) == 0:
            if 'val_loss' in logs:
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{logs['loss']:.5e}" + " | Valid loss: " + f"{logs['val_loss']:.5e}")
            else:
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{logs['loss']:.5e}")

class RMSELogger(Callback):
    def __init__(self, every=100, logpath=""):
        self.logpath = logpath
        self.every = every

    def on_epoch_end(self, epoch, X_train, fn, logs=[dict, None]):
        if(int(epoch) % self.every) == 0:
            if 'val_loss' in logs:
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{logs['loss']:.5e}" + " | Valid loss: " + f"{logs['val_loss']:.5e}")
            else:
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{logs['loss']:.5e}")