from src.utils.data_mgmt import get_data

import pickle

import tensorflow as tf
import os



def trainig123(path):

    x_train_scaled, y_train, x_test_scaled, y_test = get_data(path)

    print("Get_Data Called")

    model = tf.keras.Sequential(name = "Main_Container")

    # adding the the input layer
    model.add(tf.keras.layers.Input(shape = [8],name = "Input_layer"))

    # adding the dense layers
    model.add(tf.keras.layers.Dense(units = 9, activation = "elu", kernel_initializer = "he_normal", name = "Hidden_Layer_1"))

    model.add(tf.keras.layers.Dense(units = 8, activation = "elu", kernel_initializer = "he_normal", name = "Hidden_Layer_2"))

    model.add(tf.keras.layers.Dense(units = 4, activation = "elu", kernel_initializer = "he_normal", name = "Hidden_Layer_3"))

    # building an output layer
    model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid", kernel_initializer = "he_normal", name = "Output_layer"))

    model.compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
            )
    
    history = model.fit(
                  x_train_scaled,
                  y_train,
                  verbose = True,
                  batch_size = 16,
                  validation_split = 0.2,
                  epochs = 100
                )
    
    return model
   
def save_model(model):
    print("We are at save model")
    filename = "trained_model.sav"
    model.save("model.h5")

    
    
if __name__ == "__main__":
    print("We are at Main")
    try:
        path = os.path.join(os.getcwd(),"diabetes.csv")
        #path = "C:\\Users\\DELL\\Desktop\\LLIVE_DEMO\\Diabetic_prediction_ANN\\diabetes.csv"
        model = trainig123(path)
        save_model(model)

        print("#######################Training Completed##############################")
    except Exception as e:
        print(e)


    
