<a><img alt = 'python' src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'spyder' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
<a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>

![model_loss](static/covid.jpg)

# Forecastive Time-Series Model using Deep Learning of LSTM for Covid-19 Cases in Malaysia.
 Trained nearly 700 official data provided by Ministry of Health Malaysia to forecast Covid-19 cases.

## Description
1. The project's objective is to forecast the Malaysia's Covid-19 cases trend.
2. The data is provided through an offical GitHub page by Ministry of Health (MoH) Malaysia. The link to the GitHub page will be provided in the credit section below.
3. The dataset contains anomalies such as 2 different types of NaNs (which imputed by using df.interpolate()), but no duplicate data.
4. The layers used for the deep learning model are consist of only three layers, input layer, LSTM layer and output layer.

### Deep learning model images
![model_architecture](static/model.png)

## Results
### Training Loss:

![model_loss](static/loss.png)

### Training MAPE:

![model_mape](static/mape.png)

### Model Score:

![model_score](static/score.PNG)

### Malaysia's Covid-19 Cases Trendline:

![model_score](static/trendline.png)

## Discussion
1. The model is able to predict the trend of the Covid-19 cases in Malaysia.
2. Mean absolute error(MAE) and mean squared error(MSE) report 4.18% and 0.47% respectively when tested using the testing dataset. 
3. Based on the Loss graph displayed using Tensorboard, loss occured during the dataset training is nearly 0% with a high amount of epochs (1000).

![tensorboard](static/tensorboard.png)

4. The deep learning model used only 3 layers; input layer, LSTM layer and output layer. The number of nodes was set to 15, and the dropout rate was set to 0.03. Rectified linear unit (ReLU) was used as an activation function.
5. Based on the mean absolute percentage error(MAPE) which around 0.08%, this model can be considered as successful as it can predicted the trendline of Covid-19 cases in Malaysia.

## Credits:
The source of the dataset is obtained from GitHub page of Ministry of Health (MoH) Malaysia. Check out the latest dataset by clicking the link below. :smile:
### Dataset link
[Covid-19 Cases in Malaysia](https://github.com/MoH-Malaysia/covid19-public)

