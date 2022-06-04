from keras import layers 
from keras import models 
from keras.datasets import mnist
from keras.utils import to_categorical
import wandb 
from wandb.keras import WandbCallback

wandb.init(project="ConvNet Simple project", entity= "sunkeunjo")

def model_():
    # 컨브넷 모델링
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation = "relu", input_shape = (28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))

    # 분류기 추가
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation = "relu"))
    model.add(layers.Dense(10,activation = "softmax"))
    
    return model

if __name__ == "__main__":
    # 모델 정의 
    model = model_()
    print(model.summary())

    # 데이터 전처리 
    (train_images,train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000,28,28,1))
    train_images = train_images.astype("float32")/255
    train_labels = to_categorical(train_labels)
    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images.astype("float32")/255
    test_labels = to_categorical(test_labels)
    # 모델 학습 
    wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
    }
    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy",metrics =['accuracy'],)
    model.fit(train_images,train_labels, validation_data= (test_images,test_labels),epochs = 100, batch_size = 128 ,callbacks =[WandbCallback()])

    


