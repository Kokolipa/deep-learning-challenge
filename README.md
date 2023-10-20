# Deep Learning - Neural Network
## Project Overview:
Developing a binary classifier that can predict whether applicants will be successful, based on their chance of success in Alphabet Soup's (non-profit foundation) vertures, if funded by Alphabet. The historical data collected from Alphabet, CSV, containing more than 34,000 organisations that have received funding from Alphabet Soup over the years.
<br>

**Dataset columns:**
* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organisation classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organisation type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


## Project Results:
This project contains 2 deep learning models:
``` yml
│   ├── Deep_Models
│   |   ├── AlphabetSoupCharity.h5 # The original model
│   |   ├── AlphabetSoupCharity_Optimisation.h5 # The optimised learning model
```
### Data preprocessing
- Sense we want to predict weather applicants will be succesfull out target variable is the 'IS_SUCCESSFUL' column as it indicates on the result of the historical applications in Alphabet Soup. 
- The features for our model therefore are: 
    * APPLICATION_TYPE—Alphabet Soup application type
    * AFFILIATION—Affiliated sector of industry
    * CLASSIFICATION—Government organisation classification
    * USE_CASE—Use case for funding
    * ORGANIZATION—Organisation type
    * STATUS—Active status
    * INCOME_AMT—Income classification
    * SPECIAL_CONSIDERATIONS—Special considerations for application
    * ASK_AMT—Funding amount requested
- The 'EIN' and 'NAME' columns should be removed from the input data as they are neigher targets or features that can be considered relevant or could potentially have impact on the performance of the model. For the model, it doesn't matter what is the applicants name or ID.

### Compiling, Training, and Evaluating the Model: 
**Five models were evaluated:**
1. **The Original Model** ('AlphabetSoupCharity.h5').
2. The **second model** is trying to optimise the accuracy score by increasing the number of values for each bin & removing columns.
    * Our original model includes 43 features and inputs which may introduce high variance and low bias for our dataset, hance, manipulating the dataset by increasing the amount of values in the bins may be necessary and valueable to fine-tunning our model by reducing the amount of inputs. 
3. The **third model** tries to optimise the accuracy score by increasing the amount of apochs the model is going to itterate through. 
    * Increasing the amount of apochs results more time for learning, fixing privious errors, and deep understanding of the metric to be predicted.
4. The **fourth model** tries to optimise the accuracy score by increasing the amount of hidden layers and neurons. 
    * Here I assume the model is complex and require more layers to understand and predict the result, given the historical data, and by that increase the accuracy score. 
5. The **fifth model** tries to optimise the accuracy score by modifying the activation functions. 
    * Here I try to fine-tune the result with differnt activation functions and outline the results.

**Questions:** 
<br>

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**
<br>

* **Model 1, 2, 3, and 5:** 2 hidden layers, 80 neurons for the first hidden layer, 30 neurons for the second hidden layer and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid. 
    * Determining the number of hidden layers is dependent of the complexity of the model, in our case, one hidden layer won't suffice for the model to learn complex relashionships and pattern there for two layers are good starting point.
    * The number of neurons for each layer most of the time, the rule of thumb is that the amount of neurons should be twice as big as the number of features/inputs the model received. So 80 is a good starting point. I reduce the amount of neurons to 30 to allow the network to distill and focus on essential features as it progresses.
    * The output layer is used with sigmoid function as the model try to predict a binary result (true/false). 
    * Relu and tanh are being used for the first and second hidden layers. Relu is used for faster learning and simplied output while tanh is used to classify the data into two distinguished classes (good vs bad OR successful / unsuccessful, in our circumstances). 
* **Model 4**: 5 hidden layers, 100 neurons for the first hidden layer, 40 neurons for the second hidden layer, 10 neurons for the third layer, 3 for the fourth layer, and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid.
    * This model doesn't differ substantially from the other models, we just increase the amount of neurons and hidden layers for the same reasons mentioned above, assuming the problem to be predicted is more complex. 
<br>

**Was the target model performance achieved?** 
- Non of the models (1-5) above were able to achieve a target score higher than 75%. The third model received the highest score, **0.7481**, and was saved as 'AlphabetSoupCharity_Optimisation.h5'. 

**What steps were taken to increase the model performance?**
As mentiond above, the below steps were taken: 
1. Optimise the accuracy score by increasing the number of values for each bin & removing columns.
2. Optimise the accuracy score by increasing the amount of apochs.
3. Optimise the accuracy score by increasing the amount of hidden layers and neurons.
4. Optimise the accuracy score by modifying the activation functions.

## Summary - Table :
| **Model**| **Description**| **Accuracy Score**|**Model Loss**|**Hidden Layers Quan**|**Activation Function**|**Neurons**|
|:-|:-|:-|:-|:-|:-|:-|
|**(1)**|Original model|0.7475|0.5231|2|['relu', 'tanh', 'sigmoid']|
|**(2)**|Optimise the accuracy score by increasing the number of values for each bin & removing columns|0.7433|0.5291|2|['relu', 'tanh', 'sigmoid']|
|**(3)**|Optimise the accuracy score by increasing the amount of apochs.|0.7481|0.5186|2|['relu', 'tanh', 'sigmoid']|
|**(4)**|Optimise the accuracy score by increasing the amount of hidden layers and neurons|0.7470|0.5208|4|['relu', 'tanh', 'sigmoid', 'leaky_relu']|
|**(5)**|Optimise the accuracy score by modifying the activation functions|0.7307|0.5583|2|['relu', 'tanh', 'sigmoid']|


logistic regression 

#### Folder structure
``` yml
.
│   ├── Starter_Code 
│   |   ├── Resources   
│   |   |   ├── crypto_market_data.csv    
│   |   ├── Crypto_Clustering.ipynb              
|___README.md    
|.gitignore          
``` 