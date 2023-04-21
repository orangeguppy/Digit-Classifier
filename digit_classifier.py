import torch
from torch import nn, save, load
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define the Digit Classifier model
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # First layer: Image is black and white, only 1 channel
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(), # Remove linearity

            # Second Layer
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(), # Remove linearity

            # Last hidden layer
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(), # Remove linearity

            # Flatten
            nn.Flatten(),
            nn.Linear(64 * (28-6) * (28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Download data from the dataset (images of handwritten digits from 0-9)
training_data = datasets.MNIST(root="data/training", train=True, download=True, transform=ToTensor())
testing_data = datasets.MNIST(root="data/testing", train=False, download=True, transform=ToTensor())

training_dataloader = DataLoader(training_data, 32)
testing_dataloader = DataLoader(testing_data, 32)

# Create the model, and then define the optimiser and loss functions
neuralnet = DigitClassifier().to('cpu') # running this on a CPU. Use 'cuda' for GPU
optimiser = SGD(neuralnet.parameters(), lr=0.01) # Use Stochastic Gradient Descent, and a suggested default learning rate
loss_function = nn.CrossEntropyLoss()

# TRAIN THE MODEL:D
# for epoch in range(11): # Train for 11 epochs
#     for batch in training_dataloader:
#         x,y = batch 
#         x, y = x.to('cpu'), y.to('cpu')
#         predicted_val = neuralnet(x)
#         loss = loss_function(predicted_val, y)

#         # Use backpropagation
#         optimiser.zero_grad()
#         loss.backward() 
#         optimiser.step()
#     print("Epoch done", loss.item())

# # Save the model weights
# with open('trained_weights.pt', 'wb') as f: 
#         save(neuralnet.state_dict(), f)

# TEST THE MODEL
with open('trained_weights.pt', 'rb') as f: 
        neuralnet.load_state_dict(load(f))

# Store the total number of entries and correctly-predicted output
num_entries = len(testing_data)
num_correct = 0

# Print out the first batch only
first_batch_printed = False

for batch in testing_dataloader:
    # Extract values from the batch
    x,actual_values = batch 

    # Run the input into the neural network
    predicted_val = neuralnet(x) # Raw data from the network
    predicted_digits = [] # Store an array of the predicted digits for this batch
    
    # For output
    for result in predicted_val: # Each 'result' is an array of 10 values, for instance the first element of result
                                 # stores the probability that the image has the digit '0', index 1 for P(digit is 1), etc
        predicted_dig = torch.argmax(result).item()
        predicted_digits.append(predicted_dig)

    # Compare the input and output
    if (first_batch_printed is False): print("-----------------FIRST BATCH STARTED-----------------")
    for i in range(len(actual_values)):
        predicted_output = predicted_digits[i]
        actual_output = actual_values[i].item()

        # Check if the output is correct
        if (predicted_output == actual_output):
            num_correct += 1
        
        # Print out the results if it's the first batch
        if (first_batch_printed is False):
            print("Predicted Actual: ", predicted_output, actual_output)
    
    if (first_batch_printed is False): print("-----------------FIRST BATCH ENDED, ONLY THIS BATCH IS SHOWN-----------------")
    first_batch_printed = True

# Print the results of the test
print("NUMBER OF ENTRIES: ", num_entries)
print("NUMBER OF CORRECT ENTRIES: ", num_correct)
print("ACCURACY RATE: ", num_correct / num_entries)
