import numpy as np


def compute_dtw_distance(spec1, spec2):
    """
    Computes the DTW distance between two Mel Spectrograms.
    Follows the recursive dynamic programming formula:
    D(i, j) = d(i, j) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    """
    # N and M are the number of frames (time columns) in each recording
    N = spec1.shape[1]
    M = spec2.shape[1]

    # Initialize the Cumulative Cost Matrix with infinity
    cost_matrix = np.full((N, M), np.inf)

    # Boundary Condition: Calculate local distance for the first cell (0,0)
    cost_matrix[0, 0] = np.linalg.norm(spec1[:, 0] - spec2[:, 0])

    # Fill the first row (only horizontal movement possible)
    for i in range(1, N):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + np.linalg.norm(spec1[:, i] - spec2[:, 0])

    # Fill the first column (only vertical movement possible)
    for j in range(1, M):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + np.linalg.norm(spec1[:, 0] - spec2[:, j])

    # Fill the rest of the matrix using the recursive formula
    for i in range(1, N):
        for j in range(1, M):
            # Euclidean distance between current frame vectors
            local_dist = np.linalg.norm(spec1[:, i] - spec2[:, j])

            # Add local distance to the minimum cost of reaching this cell from neighbors
            cost_matrix[i, j] = local_dist + min(cost_matrix[i - 1, j],  # Insertion
                                                 cost_matrix[i, j - 1],  # Deletion
                                                 cost_matrix[i - 1, j - 1])  # Match

    # The final DTW cost is the value at the top-right corner
    total_cost = cost_matrix[N - 1, M - 1]

    # Normalizing by the total path length (Section 3.g.ii)
    return total_cost / (N + M)


def build_distance_matrix(test_data, db_data):
    """
    Constructs the 40x11 distance matrix required for the project (Section 3.d).
    Compares every recording in the test set against the reference DB.
    """
    speakers = list(test_data.keys())
    db_keys = list(db_data.keys())  # Expected: 0-9 + banana

    # Calculate total number of test files across all speakers
    num_test_files = sum(len(test_data[spk]) for spk in speakers)
    dist_matrix = np.zeros((num_test_files, len(db_keys)))

    row_idx = 0
    for spk in speakers:
        # Loop through each word recorded by the speaker
        for word in sorted(test_data[spk].keys()):
            test_spec = test_data[spk][word]
            # Compare current file to every entry in the reference DB
            for col_idx, db_word in enumerate(db_keys):
                ref_spec = db_data[db_word]
                dist_matrix[row_idx, col_idx] = compute_dtw_distance(ref_spec, test_spec)
            row_idx += 1

    return dist_matrix