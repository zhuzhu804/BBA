import numpy as np

# Function to calculate the entropy loss
def entropy_loss(prob_dist, epsilon=1e-9):
    # Entropy loss is the sum of p(x) * log(p(x)) for all probabilities in the distribution
    return -np.sum(prob_dist * np.log(prob_dist + epsilon))

# Function to calculate the diversity loss
def diversity_loss(prob_dist, epsilon=1e-9):
    # Diversity loss is the negative entropy of the mean probability
    mean_prob = np.mean(prob_dist)
    return -np.sum(mean_prob * np.log(mean_prob + epsilon))

# Function to calculate the information maximization loss
def information_maximization_loss(prob_dist, epsilon=1e-9):
    ent_loss = entropy_loss(prob_dist, epsilon)
    div_loss = diversity_loss(prob_dist, epsilon)
    return ent_loss + div_loss

# Given probability distributions
# p_dist = np.array([0.4, 0.6, 0, 0, 0, 0, 0, 0])
# q_dist = np.array([0, 0, 0, 0.4, 0.6, 0, 0, 0])
# q_dist = np.array([0, 0, 0, 0.8, 0, 0.2, 0, 0])

p_dist = np.array([1, 0])
q_dist = np.array([0.4, 0.6])


# Calculate the information maximization losses
p_im_loss = information_maximization_loss(p_dist)
q_im_loss = information_maximization_loss(q_dist)

# Print the results
print("Information Maximization Loss for p_dist:", p_im_loss)
print("Information Maximization Loss for q_dist:", q_im_loss)
