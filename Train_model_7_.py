import matplotlib.pyplot as plt
import matplotlib
from Save_model_json_2 import SaveModelJson
from Save_model_keras import SaveModelKeras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
class Trainmodel():

    def trainmodel(self, x_train, y_train,model_,unit1,namepic, epoch, batch_size):
        if not os.path.exists("models_keras"):
            os.mkdir("models_keras")
        name = "epoch_" + str(epoch) + "_unit1_" + str(unit1)
        namepic =str(namepic) +"epoch_" + str(epoch) + "_unit1_" + str(unit1)

        logdir = "logs/" + name
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        #model_.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
        print(x_train)
        #print(len(y_train))
        filepath="models_keras/weights.best.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True,mode='max')
        #callbacks_list = [checkpoint]
        #history =model_.fit(x_train,y_train, batch_size=batch_size, epochs=epoch, verbose=2, callbacks=callbacks_list, shuffle=True)
        history=model_.fit(x_train,y_train, batch_size=batch_size, epochs=epoch, verbose=2, shuffle=True)

        print(history.history.keys())
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("logs/"+namepic+"accuracy.png")
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig("logs/"+namepic+"loss.png")
        #save keras model
        saveTokeras_model = SaveModelKeras()
        saveTokeras_model.save_model_keras(model_,name)
        #save json model
        saveTojason_model=SaveModelJson()
        saveTojason_model.save_model_json(model_,name)
        return model_


