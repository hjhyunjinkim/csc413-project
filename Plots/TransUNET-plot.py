import re
import matplotlib.pyplot as plt

loss = []
loss_ce = []

# Open the text file
with open('downloads/train_log.txt') as f:
    # Read the file line by line
    for line in f:
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
plt.title("Training loss Vs. Iterations")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
