import numpy as np


def p_matrix():
    p = np.zeros((5, 5), dtype=float)

    p[0] = [0.9915, 0.005, 0.0025, 0, 0.001]
    p[1] = [0, 0.986, 0.005, 0.004, 0.005]
    p[2] = [0, 0, 0.992, 0.003, 0.005]
    p[3] = [0, 0, 0, 0.991, 0.009]
    p[4] = [0, 0, 0, 0, 1.0]
    return p


P = p_matrix()

accepted_lifetimes = []
required_samples = 1000

while len(accepted_lifetimes) < required_samples:
    state = 0
    months = 0
    reappeared_in_first_12 = False

    while state != 4:  # Loop until Death
        state = np.random.choice(5, p=P[state])
        months += 1

        # Check for cancer reappearance within the first 12 months
        if months <= 12 and state in [1, 2, 3]:
            reappeared_in_first_12 = True

    # Acceptance Criteria
    if months > 12 and reappeared_in_first_12:
        accepted_lifetimes.append(months)

# Calculate expected conditional lifetime
expected_lifetime = np.mean(accepted_lifetimes)
print(f"Expected lifetime (conditional): {expected_lifetime:.2f} months")
