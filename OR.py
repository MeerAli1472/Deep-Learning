from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
#Data for OR Gate
OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
   "y":  [0,1,1,1]
}
# DataFrame of OR data
df_OR = pd.DataFrame(OR)


#prepare data for Training
X,y = prepare_data(df_OR)

ETA = 0.1
EPOCHS = 10

model_OR = Perceptron(eta = ETA, epochs = EPOCHS)
model_OR.fit(X,y)
_ = model_OR.total_loss()

model_OR.save(filename="OR.model", model_dir="model_or")

#Save plot
save_plot(df_OR,model_OR,filename="OR.png")