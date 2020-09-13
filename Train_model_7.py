import matplotlib.pyplot as plt
import matplotlib
from Save_model_json_2 import SaveModelJson
from Save_model_keras import SaveModelKeras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
class Trainmodel():

    def trainmodel(self, x_train, y_train,x_validation, y_validation,model_, epoch, batch_size):
        if not os.path.exists("models_keras"):
            os.mkdir("models_keras")
        #model_.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
        print(x_train)
        print(len(y_train))

        filepath="models_keras/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True,mode='max')
        callbacks_list = [checkpoint]
        history =model_.fit(x_train, np.array(y_train),validation_data=(x_validation, np.array(y_validation)), batch_size=batch_size, epochs=epoch, verbose=2, callbacks=callbacks_list, shuffle=True)
        fresult=open("logs/result.txt", "w", encoding="utf8")
        fresult.write("history.history.keys:   "+str(history.history.keys()) + "\n")
        fresult.write("history.history.values:   " + str(history.history.values()) + "\n")
        print(history.history.keys())
        print(history.history.values())
############################################
#dict_keys(['val_loss', 'val_crf_viterbi_accuracy', 'loss', 'crf_viterbi_accuracy'])
        # Plot the graph
        plt.style.use('ggplot')

        def plot_history(history):

            accuracy = history.history['crf_viterbi_accuracy']
            val_acc = history.history['val_crf_viterbi_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            x = range(1, len(accuracy) + 1)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(x, accuracy, 'b', label='Training acc')
            plt.plot(x, val_acc, 'r', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, loss, 'b', label='Training loss')
            plt.plot(x, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

        plot_history(history)

        #save keras model
        saveTokeras_model = SaveModelKeras()
        saveTokeras_model.save_model_keras(model_,"mykeras")
        #save json model
        saveTojason_model=SaveModelJson()
        saveTojason_model.save_model_json(model_,"myjson")
        return model_