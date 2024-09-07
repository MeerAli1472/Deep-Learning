from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd

def main(data,modelName,plotName,eta,epochs):
    # DataFrame of OR data
    df_OR = pd.DataFrame(data)

    #prepare data for Training
    X,y = prepare_data(df_OR)

    model_OR = Perceptron(eta =eta, epochs = epochs)
    model_OR.fit(X,y)
    _ = model_OR.total_loss()
    model_OR.save(filename=modelName, model_dir="model")

    #Save plot
    save_plot(df_OR,model_OR,filename=plotName)


if __name__ == "__main__":

    #Data for OR Gate
    OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
    "y":  [0,1,1,1]
    }

    ETA = 0.1
    EPOCHS = 10

    main(data=OR,modelName="or.model",plotName="or.png",eta=ETA,epochs=EPOCHS)

