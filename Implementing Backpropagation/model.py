import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Define all possible inputs for the model
x = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

y_true = x # We want the fianal output to be the same as the input


#Define sigmoid function and the derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1-x)

# Then define the training function for the model
def train_model(
    x,
    y_true, #Data 
    input_size=8, #Size of model
    hidden_size=3,
    output_size=8,
    epochs=100000, # Iterations
    learning_rate=0.1, # Learning rate,
    random_seed=100
):
    # set the random seed for reproducibility
    np.random.seed(random_seed)
    # data size
    n = len(x)

    # Initialize weights randomly
  
    W1 = np.random.randn(hidden_size, input_size)
    W2 = np.random.randn(output_size, hidden_size)

    # Initialize biases as 0
    b1 = np.zeros(hidden_size)
    b2 = np.zeros(output_size)

    # Store results
    a_full = []
    y_full = []
    W1_full = []
    W2_full = []
    b1_full = []
    b2_full = []
    errors = []

    for epoch in range(epochs): # For all iterations 

        delta2 = [] #Save results for each iteration
        delta1 = []
        dW2 = []
        dW1 = []
        error = []
        a_epoch = []
        y_epoch = []

        for m in range(n): # For all data points
            # calculate hidden layer and output (Forward propagation)
            a = sigmoid(np.dot(W1, x[m]) + b1) 
            y = sigmoid(np.dot(W2, a) + b2)

            a_epoch.append(a)
            y_epoch.append(y)

            # Calculate error 
            error.append(0.5 * np.sum((y_true[m] - y)**2))
            
            # Calculate delta (Backpropagation)
            delta2_m = (
                sigmoid_derivative(y)
                * (y-y_true[m])
            )
            delta1_m = (
                np.dot(W2.T, delta2_m)
                * sigmoid_derivative(a)
            )

            delta2.append(delta2_m)
            delta1.append(delta1_m)

            # Derive partial derivatives for delta
            dW2.append(np.outer(delta2_m, a))
            dW1.append(np.outer(delta1_m, x[m]))

        # Update weights
        W2 = W2 - learning_rate * np.mean(dW2, axis=0)
        W1 = W1 - learning_rate * np.mean(dW1, axis=0)

        # Update biases (derivative with respect to bias is just 1 so we can just use the delta)
        b2 = b2 - learning_rate * np.mean(delta2, axis=0)
        b1 = b1 - learning_rate * np.mean(delta1, axis=0)

        # Store training history
        errors.append(np.mean(error))

        a_full.append(a_epoch)
        y_full.append(y_epoch)
        W1_full.append(W1.copy())
        W2_full.append(W2.copy())
        b1_full.append(b1.copy())
        b2_full.append(b2.copy())

    # Return trained model and history
    return {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2,
        "errors": errors,
        "a_full": a_full,
        "y_full": y_full,
        "W1_full": W1_full,
        "W2_full": W2_full,
        "b1_full": b1_full,
        "b2_full": b2_full
    }

#Interpretate the results of the model

result = train_model(x, y_true)
baseline_errors = result["errors"].copy()

# Convert recorded parameter lists to NumPy arrays

W1_hist = np.array(result["W1_full"])  # Shape (epochs, 3, 8)
W2_hist = np.array(result["W2_full"])  # Shape (epochs, 8, 3)
b1_hist = np.array(result["b1_full"])  # Shape (epochs, 3)
b2_hist = np.array(result["b2_full"])  # Shape (epochs, 8)
errors = result["errors"]
y_full = np.array(result["y_full"])  # Shape (epochs, 8, 8)
a_full = np.array(result["a_full"])  # Shape (epochs, 8, 3)

print("Final output after training:")
print(np.round(y_full[-1], 2))

print("Final hidden layer activations after training:")
print(np.round(a_full[-1], 2))


epochs_range = range(1, len(W1_hist) + 1)

################################ PLOT RESULTS ################################
## error X Epochs Graph ##

plt.figure(figsize=(9, 4.5))
plt.plot(range(1, len(errors) + 1), errors, linewidth=1.5)

plt.title("Training Error", fontsize=11, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.grid(True, which="both", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()



## Hidden-Layer Weights ##

fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig1.suptitle(
    "Hidden-Layer Weights",
    fontsize=12,
    fontweight="bold",
)

for h in range(3):
    for i in range(8):
        axes1[h].plot(
            epochs_range,
            W1_hist[:, h, i],
            linewidth=1.5,
            label=f"Input {i+1}",
        )

    axes1[h].set_title(f"Hidden Neuron {h+1}", fontsize=10, fontweight="bold")

    axes1[h].set_ylabel("Weight Value")
    axes1[h].grid(True, linestyle=":", alpha=0.6)

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(
    handles,
    labels,
    title="Input",
    loc="center right",
    bbox_to_anchor=(0.99, 0.5),
    fontsize=8,
)

axes1[2].set_xlabel("Epoch")

plt.tight_layout(rect=(0, 0, 0.86, 0.95))
plt.show()


## Output-Layer Weights ##

fig2, axes2 = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
fig2.suptitle(
    "Output-Layer Weights",
    fontsize=12,
    fontweight="bold",
)

axes2_flat = axes2.flatten()

for o in range(8):
    for h in range(3):
        axes2_flat[o].plot(
            epochs_range,
            W2_hist[:, o, h],
            linewidth=1.5,
            label=f"Hidden {h+1}",
        )

    axes2_flat[o].set_title(f"Output Neuron {o+1}", fontsize=10, fontweight="bold")

    axes2_flat[o].set_ylabel("Weight Value")
    axes2_flat[o].grid(True, linestyle=":", alpha=0.6)

handles, labels = axes2_flat[0].get_legend_handles_labels()
fig2.legend(
    handles,
    labels,
    title="Hidden Unit",
    loc="center right",
    bbox_to_anchor=(0.99, 0.5),
    fontsize=8,
)
axes2_flat[6].set_xlabel("Epoch")
axes2_flat[7].set_xlabel("Epoch")

plt.tight_layout(rect=(0, 0, 0.86, 0.95))
plt.show()


## Biases b1 & b2 (2 Graphs on the Same Page) ##

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(
    "Biases",
    fontsize=12,
    fontweight="bold",
)

# Hidden-layer biases
for h in range(3):
    axes3[0].plot(
        epochs_range,
        b1_hist[:, h],
        linewidth=2,
        label=f"Neuron {h+1}",
    )

axes3[0].set_title("Hidden Layer", fontsize=11, fontweight="bold")

axes3[0].set_xlabel("Epoch")
axes3[0].set_ylabel("Bias Value")
axes3[0].grid(True, linestyle=":", alpha=0.6)
axes3[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=3
)

# Output-layer biases
for o in range(8):
    axes3[1].plot(
        epochs_range,
        b2_hist[:, o],
        linewidth=1.5,
        label=f"Neuron {o+1}",
    )

axes3[1].set_title("Output Layer", fontsize=11, fontweight="bold")

axes3[1].set_xlabel("Epoch")
axes3[1].set_ylabel("Bias Value")
axes3[1].grid(True, linestyle=":", alpha=0.6)
axes3[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=4
)

plt.tight_layout(rect=(0, 0.12, 1, 0.9))
plt.show()


########################## EXPERIEMENTS ###########################

## Run the model 100 times with different random seeds (different starting weights)
# Plot each run's training-error history to compare how the models learn.
num_trials = 100
test_lr = 0.1
#test_epochs = 100000 # Actual run 
test_epochs = 1000 # to not take too long

plt.figure(figsize=(10, 6))
final_errors = []
final_predictions = []

for trial in range(1, num_trials + 1):
    res = train_model(
        x, y_true,
        epochs=test_epochs,
        learning_rate=test_lr,
        random_seed=trial
    )
    plt.plot(range(1, test_epochs + 1), res["errors"], alpha=0.6)

    hidden_outputs = sigmoid(np.dot(x, res["W1"].T) + res["b1"])
    predictions = sigmoid(np.dot(hidden_outputs, res["W2"].T) + res["b2"])
    final_error = np.mean(0.5 * np.sum((y_true - predictions) ** 2, axis=1))
    final_errors.append(final_error)
    final_predictions.append(predictions.copy())

plt.title(f"Training Error Across {num_trials} Random Initializations")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.boxplot(final_errors)
plt.title(f"Final Training Error Across {num_trials} Random Initializations")
plt.ylabel("Training error")
plt.xticks([1], ["Final error"])
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(final_errors, bins="auto", edgecolor="black", alpha=0.75)
plt.title(f"Distribution of Final Training Errors ({num_trials} Runs)")
plt.xlabel("Final training error")
plt.ylabel("Number of runs")
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Compare each model's final outputs with the average outputs across all runs.
prediction_array = np.array(final_predictions)
mean_predictions = np.mean(prediction_array, axis=0)
run_differences = np.mean(np.abs(prediction_array - mean_predictions), axis=(1, 2))

plt.figure(figsize=(6, 5))
plt.boxplot(run_differences)
plt.title("Variation in Final Outputs Across Runs")
plt.ylabel("Average absolute difference from mean outputs")
plt.xticks([1], ["All runs"])
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()


############ Change of hyper parameters: Learning rate, Epochs ############

# Test 1: 10,000 epochs, learning rate 0.1
test_result = train_model(x, y_true, epochs=10000, learning_rate=0.1)
print("y1:")
print(np.round(test_result["y_full"][-1], 2))
print("a1:")
print(np.round(test_result["a_full"][-1], 2))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(baseline_errors) + 1), baseline_errors, label="Starting model")
plt.plot(range(1, len(test_result["errors"]) + 1), test_result["errors"], label="10,000 epochs, LR 0.1")
plt.title("Test 1: 10,000 Epochs, Learning Rate 0.1")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Test 2: 100,000 epochs, learning rate 1
test_result = train_model(x, y_true, epochs=100000, learning_rate=1)
print("y2:")
print(np.round(test_result["y_full"][-1], 2))
print("a2:")
print(np.round(test_result["a_full"][-1], 2))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(baseline_errors) + 1), baseline_errors, label="Starting model (LR 0.1, 100,000 epochs)")
plt.plot(range(1, len(test_result["errors"]) + 1), test_result["errors"], label="LR 1")
plt.title("Test 2: 100,000 Epochs, Learning Rate 1")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Test 3: 10,000 epochs, learning rate 1
test_result = train_model(x, y_true, epochs=10000, learning_rate=1)
print("y3:")
print(np.round(test_result["y_full"][-1], 2))
print("a3:")
print(np.round(test_result["a_full"][-1], 2))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(baseline_errors) + 1), baseline_errors, label="Starting model")
plt.plot(range(1, len(test_result["errors"]) + 1), test_result["errors"], label="10,000 epochs, LR 1")
plt.title("Test 3: 10,000 Epochs, Learning Rate 1")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Test 4: 2 neurons in the hidden layer
test_result = train_model(x, y_true, hidden_size=2)
print("y4 (2 hidden neurons):")
print(np.round(test_result["y_full"][-1], 2))
print("a4 (2 hidden neurons):")
print(np.round(test_result["a_full"][-1], 2))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(baseline_errors) + 1), baseline_errors, label="Starting model (3 hidden neurons)")
plt.plot(range(1, len(test_result["errors"]) + 1), test_result["errors"], label="2 hidden neurons")
plt.title("Test 4: 2 Hidden Neurons")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Test 5: 4 neurons in the hidden layer
test_result = train_model(x, y_true, hidden_size=4)
print("y5 (4 hidden neurons):")
print(np.round(test_result["y_full"][-1], 2))
print("a5 (4 hidden neurons):")
print(np.round(test_result["a_full"][-1], 2))

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(baseline_errors) + 1), baseline_errors, label="Starting model (3 hidden neurons)")
plt.plot(range(1, len(test_result["errors"]) + 1), test_result["errors"], label="4 hidden neurons")
plt.title("Test 5: 4 Hidden Neurons")
plt.xlabel("Epoch")
plt.ylabel("Training error")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
