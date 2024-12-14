# import numpy as np
# from scipy.optimize import minimize
#
# # Given probability distribution p
# p_dist = np.array([0.4, 0.6, 0, 0, 0, 0, 0, 0])
#
# # Define the cross-entropy function
# def cross_entropy(p, q):
#     return -np.sum(p * np.log(q + 1e-9))  # Adding a small epsilon to avoid log(0)
#
# # Define the objective function to minimize
# def objective(Aq):
#     A = Aq[:8]  # Extracting A distribution from Aq
#     q = Aq[8:]  # Extracting q distribution from Aq
#     # Calculate cross-entropy between p and A, and q and A
#     ce_p_A = cross_entropy(p_dist, A)
#     ce_q_A = cross_entropy(q, A)
#     # The objective is to minimize the absolute difference between cross-entropy results
#     return abs(ce_p_A - ce_q_A)
#
# # Constraints for probabilities to ensure precision of 0.1
# def constraints(Aq):
#     A = Aq[:8]
#     q = Aq[8:]
#     # A probabilities should sum up to 1
#     sum_A = np.sum(A)
#     # q probabilities should sum up to 1 and have precision of 0.1
#     sum_q = np.sum(q)
#     q_precision = np.all(np.isclose(q % 0.1, 0))  # Check if q values have precision of 0.1
#     return [sum_A - 1, sum_q - 1, q_precision]
#
# # Initial guess for A distribution and q distribution
# A_guess = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Initial guess for A
# q_dist = np.round(p_dist, 1)  # q starts with rounded p values to ensure 0.1 precision
# Aq_guess = np.concatenate((A_guess, q_dist))  # Concatenating A and q to optimize together
#
# # Bounds for probabilities (between 0 and 1)
# bounds = [(0, 1)] * 8  # Bounds for A probabilities
# bounds += [(0.1, 0.9)] * 8  # Bounds for q probabilities with 0.1 precision
#
# # Perform optimization to find A and q with constraints
# result = minimize(objective, Aq_guess, bounds=bounds, constraints={'type': 'eq', 'fun': constraints})
#
# # Extract optimized A and q distributions from the result
# A_optimized = result.x[:8]
# q_optimized = result.x[8:]
#
# # Print optimized distributions
# print("Optimized A distribution:", A_optimized)
# print("Optimized q distribution:", q_optimized)


import numpy as np

# Define the cross-entropy function
def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-9))  # Adding a small epsilon to avoid log(0)

# Given probability distributions
# p_dist = np.array([0.4, 0.6, 0, 0, 0, 0, 0, 0])
# q_dist = np.array([0, 0, 0, 0.4, 0.6, 0, 0, 0])
# ref_dist = np.array([0, 0, 0.9, 0, 0, 0.1, 0, 0])

# p_dist = np.array([0, 0.1, 0.8, 0, 0.1, 0, 0, 0])
# q_dist = np.array([0, 1, 0, 0, 0, 0, 0, 0])
# ref_dist = np.array([0, 0.1, 0.1, 0, 0.8, 0, 0, 0])
p_dist = np.array([1, 0])
q_dist = np.array([0.4, 0.6])
ref_dist = np.array([0.9, 0.1])


# Calculate cross-entropy between p and reference distribution
ce_p_ref = cross_entropy(p_dist, ref_dist)
print("Cross-entropy between p and reference distribution:", ce_p_ref)

# Calculate cross-entropy between q and reference distribution
ce_q_ref = cross_entropy(q_dist, ref_dist)
print("Cross-entropy between q and reference distribution:", ce_q_ref)


# import numpy as np
#
# # Given probability distributions
# p_dist = np.array([0.4, 0.6, 0, 0, 0, 0, 0, 0])
# q_dist = np.array([0, 0, 0, 0.4, 0.6, 0, 0, 0])
#
# # Define the cross-entropy function
# def cross_entropy(p, q):
#     return -np.sum(p * np.log(q + 1e-9))  # Adding a small epsilon to avoid log(0)
#
# # Define the objective function to minimize
# def objective(A):
#     # Calculate cross-entropy between p and A, and q and A
#     ce_p_A = cross_entropy(p_dist, A)
#     ce_q_A = cross_entropy(q_dist, A)
#     # Return the absolute difference between cross-entropy results
#     return abs(ce_p_A - ce_q_A)
#
# # Generate all possible combinations of A with precision of 0.1
# A_options = np.arange(0, 1.1, 0.1)
#
# # Initialize list to store all valid A distributions
# valid_A_distributions = []
#
# # Iterate over all possible combinations of A
# for a1 in A_options:
#     for a2 in A_options:
#         for a3 in A_options:
#             for a4 in A_options:
#                 for a5 in A_options:
#                     for a6 in A_options:
#                         for a7 in A_options:
#                             for a8 in A_options:
#                                 A = np.array([a1, a2, a3, a4, a5, a6, a7, a8])
#                                 if np.sum(A) == 1:  # Ensure sum of probabilities is 1
#                                     if objective(A) < 0.01:  # Check if within desired objective value
#                                         print("Valid A distribution:", A)
#                                         valid_A_distributions.append(A)
#
# # Print all valid A distributions
# for A in valid_A_distributions:
#     print("Valid A distribution:", A)
#
#
#
#
#
#
#
#
#
#
