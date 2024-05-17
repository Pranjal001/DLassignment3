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

The Repo contains 4 files and one folder namely :

predictions_attention (folder)
DLA_A3_VANILLA_WANDBattention.ipynb : Wandb implemented file for attention based model. can open and add your Wandb key in the first cell and hypertune parameters on you wandb account(ipynb file).
DLA_A3_VANILLA_WANDB.ipynb : Wandb implemented file for Vanilla model. can open and add your Wandb key in the first cell and hypertune parameters on you wandb account (ipynb file).
DLA3_vanilla_train.py : .py file that work with command line arguments(for vanilla model).
DLA3_attention_train.py : .py file that work with command line arguments(for attention based model).

By default the py files are initialized with best parameters as default parameters.
The following is the command line syntax to run the .py files:

('-dp', '--data_path', type=str, help='Path to the data folder')

('-ln', '--lng', type=str, help=' provide which language you want it to be trained on')

('-es', '--embedding_size', type=int, help='Embedding size')

('-hs', '--hidden_size', type=int, help='Hidden size')

('-l', '--layers', type=int,  help='layers size')

('-mt', '--module_type', type=str,  choices=['RNN', 'LSTM', 'GRU'], help='Module type (RNN, LSTM, GRU)')

('-dt', '--dropout', type=float,  help='Dropout rate')

('-lr', '--learning_rate', type=float,  help='Learning rate')

('-bs', '--batch_size', type=int, help='Batch size')

('-e', '--num_epochs', type=int,  help='Number of epochs')

('-opt', '--optimizer', type=str,  choices=['adam', 'sgd', 'rmsprop', 'nadam', 'adagrad'], help='Optimizer (adam, sgd, rmsprop, nadam, adagrad)')

('-bw', '--beam_search_width', type=int,  help='Beam search width')

('-lp', '--length_penalty', type=float,  help='Length penalty')

('-tf', '--teacher_forcing', type=float, help='Teacher forcing ratio')

('-bi', '--bidirectional_type', help='Use bidirectional_type encoder')

command line code syntax:

!pyhton [pyhton file location] -dp [data path(add data path up to languages list folder only)(eg-> /kaggle/input/aksharantar-sampled)] -ln [add language you wish to train on(hin/mar/mal/asm etc.)] [..any parameter from the above list with appropriate values]

remove [content] from the above and add appropriate input there.

you can also use -help to view list of possible arguments/parameters.
