# Deep Learning - Neural Network
## Project Overview:
Developing a binary classifier that can predict whether applicants will succeed based on their chance of success in Alphabet Soup's (non-profit foundation) ventures if funded by Alphabet. The historical data collected from Alphabet, CSV, contains over 34,000 organisations that have received funding from Alphabet Soup over the years.
<br>

**Dataset columns:**
* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organisation classification
* USE_CASE—Use case for funding
* ORGANISATION—Organisation type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


## Project Results:
This project contains 2 deep-learning models:
``` yml
│   ├── Deep_Models
│   |   ├── AlphabetSoupCharity.h5 # The original model
│   |   ├── AlphabetSoupCharity_Optimisation.h5 # The optimised learning model
```
### Data preprocessing
- Since we want to predict whether applicants will be successful, our target variable is the 'IS_SUCCESSFUL' column, as it indicates the result of the historical applications in Alphabet Soup. 
- The features of our model, therefore are: 
    * APPLICATION_TYPE—Alphabet Soup application type
    * AFFILIATION—Affiliated sector of industry
    * CLASSIFICATION—Government organisation classification
    * USE_CASE—Use case for funding
    * ORGANISATION—Organisation type
    * STATUS—Active status
    * INCOME_AMT—Income classification
    * SPECIAL_CONSIDERATIONS—Special considerations for application
    * ASK_AMT—Funding amount requested
- The 'EIN' and 'NAME' columns should be removed from the input data as they are neither targets nor features that can be considered relevant or could potentially impact the model's performance. For the model, the applicant's name or ID doesn't matter.

### Compiling, Training, and Evaluating the Model: 
**Five models were evaluated:**
1. **The Original Model** ('AlphabetSoupCharity.h5').
2. The **second model** is trying to optimise the accuracy score by increasing the number of values for each bin & removing columns.
    * Our original model includes 43 features and inputs, which may introduce high variance and low bias for our dataset. Hence, manipulating the dataset by increasing the number of values in the bins may be necessary and valuable to fine-tune our model by reducing the amount of inputs. 
3. The **third model** tries to optimise the accuracy score by increasing the number of epochs the model is going to iterate through. 
    * Increasing the number of epochs results in more time for learning, fixing previous errors, and a deep understanding of the metric to be predicted.
4. The **fourth model** tries to optimise the accuracy score by increasing the amount of hidden layers and neurons. 
    * Here, I assume the model is complex and requires more layers to understand and predict the result, given the historical data, and by that, increase the accuracy score. 
5. The **fifth model** tries to optimise the accuracy score by modifying the activation functions. 
    * Here, I try to fine-tune the result with different activation functions and outline the results.

**Questions:** 
<br>

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**
<br>

* **Models 1, 2, 3, and 5:** 2 hidden layers, 80 neurons for the first hidden layer, 30 neurons for the second hidden layer, and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid. 
    * Determining the number of hidden layers is dependent of the complexity of the model, in our case, one hidden layer won't suffice for the model to learn complex relationships and patterns so two layers are a good starting point.
    * The number of neurons for each layer most of the time. The rule of thumb is that the number of neurons should be twice as big as the number of features/inputs the model received. So 80 is a good starting point. I reduce the number of neurons to 30 to allow the network to distil and focus on essential features as it progresses.
    * The output layer is used with a sigmoid function as the model tries to predict a binary result (true/false). 
    * Relu and tanh are being used for the first and second hidden layers. Relu is used for faster learning and simplified output, while tanh is used to classify the data into two distinguished classes (good vs bad OR successful/unsuccessful, in our circumstances). 
* **Model 4**: 5 hidden layers, 100 neurons for the first hidden layer, 40 neurons for the second hidden layer, 10 neurons for the third layer, 3 for the fourth layer, and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid.
    * This model doesn't differ substantially from the other models; we just increase the number of neurons and hidden layers for the same reasons mentioned above, assuming the problem to be predicted is more complex. 
<br>

**Was the target model performance achieved?** 
- None of the models (1-5) above achieved a target score higher than 75%. The third model received the highest score, **0.7481**, and was saved as 'AlphabetSoupCharity_Optimisation.h5'. 

**What steps were taken to increase the model performance?**
As mentioned above, the below steps were taken: 
1. Optimise the accuracy score by increasing the number of values for each bin & removing columns.
2. Optimise the accuracy score by increasing the number of epochs.
3. Optimise the accuracy score by increasing the amount of hidden layers and neurons.
4. Optimise the accuracy score by modifying the activation functions.

## Summary - Table :
| **Model**| **Accuracy Score**|**Model Loss**|**Hidden Layers Quan**|**Activation Function**|**Neurons**|
|:-|:-|:-|:-|:-|:-|
|**(1)**|0.7475|0.5231|2|['relu', 'tanh', 'sigmoid']|80, 30|
|**(2)**|0.7433|0.5291|2|['relu', 'tanh', 'sigmoid']|80, 30|
|**(3)**|**0.7481**|**0.5186**|**2**|**['relu', 'tanh', 'sigmoid']**|**80, 30**|
|**(4)**|0.7470|0.5208|4|['relu', 'tanh', 'sigmoid', 'leaky_relu']|100, 40, 10, 3|
|**(5)**|0.7307|0.5583|2|['relu', 'tanh', 'sigmoid']|80, 30|

### Recommendation
- [x] Given that our model tries to predict a binary result, the **logistic regression** model can potentially be more effective in solving Alphabet Soup's problem statement as it estimates the probability of an event occurring, such as successful/unsuccessful, based on a given dataset of independent variables. Logistic regression deals well with features with linear relationships but can also perform well with features without linear relationships.

### Libraries Used
1. sklearn
2. pandas
3. numpy
4. tensorflow 
5. keras_tuner

#### Folder structure
``` yml
.
│   ├── Deep_Models 
│   |   ├── AlphabetSoupCharity_Optimisation.h5  
│   |   ├── AlphabetSoupCharity.h5   
│   ├── fine_tunining_trails
│   |   ├── trial_0000  
│   |   ├── trial_0001
│   |   ├── trial_0002  
│   |   ├── trial_0003
│   |   ├── oracle.json  
│   |   ├── tuner0.json 
│   ├── AlphabetSoupCharity_Optimisation.ipynb 
│   ├── Starter_Code.ipynb
|___README.md    
|.gitignore          
``` 