# DL ASSIGNMENT 3 : Implementation of Encoder and Decoder based models :
This Repository consists for several files which are implementation based on the Deep learning Encoder/Decoder based models which takes a sample of the Aksharantar dataset released by AI4Bharat. This dataset contains pairs of the following form: ajanabee,अजनबी

The Model is trained on the Hindi dataset, but can be used to train other languges as well. The model provides various parameters such as Modeltype(LSTM, GRU , RNN) , bidirectional type, beam length , teacher forcing , number of epochs etc.

Following are the customizable parameters which are implementing in the code :

embedding_size = embedding size for the model (INT)

hidden_size = hidden layer size (INT)

Layers = layer size for encoder and decoder (INT)

module_type = Type of model to be used (LSTM, GRU , RNN) (STRING)(PROVIDE INPUT IN CAPS ONLY)

dropout = dropout probability (ranging from 0 to 1)(FLOAT)

learning_rate = learning rate for gradiend descent algorithms (FLOAT)

batch_size = batchsize for the model (INT)

num_epochs = Number of iterations for training (INT)

optimizer = Possible optimizers for training (STRING)('adam', 'sgd', 'rmsprop', 'nadam', 'adagrad')

beam_width = beam with for the beam search (INT)

bidirectional_type = parameter to enable bidirectional in model (True/False)(BOOL)

length_penalty = length penalty (ranging from 0 to 1)(FLOAT)

teacher_forcing = teacher forcing ratio (ranging from 0 to 1)(FLOAT)


# Running the files :

The Repo contains 5 files namely 
