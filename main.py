import matplotlib
from Existing_model_MADDPG import MADDPG
from Existing_model_TADPG import TADPG
from proposed_DQN import proposed
from save_load import load, save
matplotlib.use('TkAgg', force=True)
from Data_gen import *
from Existing_model_DDQN import *
from Existing_model_LSTM_MCTS import *
from MHAT_LSTM import *
from QI_PSO_MOGA import *
from plot_result import *
def full_analysis():
    datagen()
    MHAT_LSTM()
    QI_PSO_MOGA()

    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')

    for i in range(2):

        #PROPOSED
        met= proposed(X_train[i], X_test[i],  y_train[i],y_test[i])
        save('proposed_met',met)

        #DDQN
        met = DDQN(X_train[i], X_test[i], y_train[i], y_test[i])
        save('DDQN_met', met)

        #LSTM_MCTS
        cm =LSTM_MCTS(X_train[i], X_test[i], y_train[i], y_test[i])
        save('LSTM_MCTS_met',cm)

        #MADDPG
        cm = MADDPG(X_train[i], X_test[i], y_train[i], y_test[i])
        save('MADDPG_met', cm)

        #TADPG
        cm =TADPG(X_train[i], X_test[i], y_train[i], y_test[i])
        save('TADPG_met', cm)



a =0
if a == 1:
    full_analysis()

plot_res()
validation_loss_graph()