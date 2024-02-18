import matplotlib.pyplot as plt

# Using readlines()
import numpy as np

# base_path = './logs/superpoint_retina_1024/'
# f = open(base_path + 'training_output.txt', 'r')
# f = open('./logs/superpoint_retina/training_output.txt', 'r')
f = open('./training_output.txt', 'r')
lines = f.readlines()
f.close()

# Strips the newline character
train_loss_lst = []
val_loss_lst = []
x = []

for line in lines:
    if 'train loss detector:' in line:
        train_loss = float(line.split(' ')[-1])
        train_loss_lst.append(train_loss)
    elif 'val loss detector:' in line:
        val_loss = float(line.split(' ')[-1])
        val_loss_lst.append(val_loss)
    elif 'epoch' in line:
        epoch = int(line.split(' ')[-1])
        x.append(epoch)
#
# x_train_loss = list(range(len(train_loss_lst)))
# x_val_loss = list(range(len(val_loss_lst)))
# first_k = int(25/5)
first_k = 225
plt.plot(x[:first_k], train_loss_lst[:first_k], label='train_detector_loss')
plt.plot(x[:first_k], val_loss_lst[:first_k], label='val_detector_loss')
plt.legend()
# plt.savefig(base_path + 'detector_loss_vs_epoch_0_76.png')
plt.savefig('detector_loss_vs_epoch_0_225.png')
plt.show()

# fig, ax = plt.subplots(2, 3)
# ax[0, 0].set_title('Train loss')
# ax[0, 0].plot(x_train_loss, train_loss_lst)
# ax[0, 1].set_title('Train Precision')
# ax[0, 1].plot(x_train_precision, train_precision_lst)
# ax[0, 2].set_title('Train Recall')
# ax[0, 2].plot(x_train_recall, train_recall_lst)
#
# ax[1, 0].set_title('Val loss')
# ax[1, 0].plot(x_val_loss, val_loss_lst)
# ax[1, 1].set_title('Val Precision')
# ax[1, 1].plot(x_val_precision, val_precision_lst)
# ax[1, 2].set_title('Val Recall')
# ax[1, 2].plot(x_val_recall, val_recall_lst)
#
# plt.tight_layout()
# plt.show()


