import matplotlib.pyplot as plt
import re

# define empty lists for storing loss and ce_loss values
loss = []
loss_ce = []
epochs = []

# read in the log file
with open('downloads/train_log.txt', 'r') as f:
    # skip the first line
    next(f)
    i = 7
    for line in f:
        if i != 93:
            i += 1 
            continue

        i = 0
        # Use regular expressions to match the values of the loss and loss_ce
        match_loss = re.search(r'loss : ([\d.]+),', line)
        match_loss_ce = re.search(r'loss_ce: ([\d.]+)', line)
        if match_loss and match_loss_ce:
            # Extract the values of the loss and loss_ce
            loss.append(float(match_loss.group(1)))
            loss_ce.append(float(match_loss_ce.group(1)))

# Plot the values of the loss and loss_ce
plt.figure()
plt.plot(loss, label='Soft Dice Loss')
plt.plot(loss_ce, label='CE Loss')
plt.title("Training loss Vs. Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

