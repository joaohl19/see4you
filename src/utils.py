import matplotlib.pyplot as plt
import numpy as np

def write_loss_graph(loss_train_list : list, loss_test_list : list):
        plt.style.use('ggplot')
        
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(loss_train_list, color = "red")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Epochs")
        title = "Epoch x Loss in train"
        ax[0].set_title(title)

        ax[1].plot(loss_test_list, color = "blue")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epochs")
        title = "Epoch x Loss in val"
        ax[1].set_title(title)
        plt.show()