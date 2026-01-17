import numpy as np


def compute_dtw_distance(spectrogram1, spectrogram2, normalize=True):
    """
        Computes the Dynamic Time Warping (DTW) distance between two Mel Spectrograms.

        The spectrograms are 2D matrices where:
        - Rows: Frequency bins (Mel filter banks)
        - Columns: Time frames (feature vectors)

        This function aligns the sequences by finding the optimal path that minimizes
        the cumulative cost using the recursive formula:
        D(i, j) = d(i, j) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

        Args:
            spectrogram1 (np.ndarray): The reference spectrogram from the DB.
            spectrogram2 (np.ndarray): The test spectrogram to be classified.
            normalize (bool): If True, divides the final cost by (N + M) to account for
                              different audio lengths (Section 3.g.ii).

        Returns:
            float: The calculated DTW distance (cost) between the two signals.
        """
    # the number of frames (time columns) in each recording
    num_frames_ref, num_frames_test = spectrogram1.shape[1], spectrogram2.shape[1]

    # Initialize the Cumulative Cost Matrix with infinity
    cost_matrix = np.full((num_frames_ref, num_frames_test), np.inf)

    # Boundary Condition: Calculate local distance for the first cell (0,0)
    cost_matrix[0, 0] = np.linalg.norm(spectrogram1[:, 0] - spectrogram2[:, 0])

    # Fill the first row (only horizontal movement possible)
    for i in range(1, num_frames_ref):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + np.linalg.norm(spectrogram1[:, i] - spectrogram2[:, 0])

    # Fill the first column (only vertical movement possible)
    for j in range(1, num_frames_test):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + np.linalg.norm(spectrogram1[:, 0] - spectrogram2[:, j])

    # Fill the rest with the matrix using the recursive formula
    for i in range(1, num_frames_ref):
        for j in range(1, num_frames_test):
            local_dist = np.linalg.norm(spectrogram1[:, i] - spectrogram2[:, j])

            # Add local distance to the minimum cost of reaching this cell from neighbors
            cost_matrix[i, j] = local_dist + min(cost_matrix[i - 1, j],  # Insertion
                                                 cost_matrix[i, j - 1],  # Deletion
                                                 cost_matrix[i - 1, j - 1])  # Match

    # The final DTW cost is the value in the top-right corner
    total_cost = cost_matrix[num_frames_ref - 1, num_frames_test - 1]

    # Normalizing by the total path length (Section 3.g.ii)
    return total_cost / (num_frames_ref + num_frames_test) if normalize else total_cost


def build_distance_matrix(test_data, db_data, db_keys):
    """
    Constructs the 40x11 distance matrix required for the project (Section 3.d).
    Compares every recording in the test set against the reference DB.
    """
    # IMPORTANT: keep speaker iteration deterministic and consistent with
    # utils.get_labels_and_data(), which uses sorted speaker keys.
    speakers = sorted(test_data.keys())

    # Calculate total number of test files across all speakers
    num_test_files = sum(len(test_data[spk]) for spk in speakers)
    dist_matrix = np.zeros((num_test_files, len(db_keys)))

    row_idx = 0
    for spk in speakers:
        for word in sorted(test_data[spk].keys()):
            test_spectrogram = test_data[spk][word]
            # Compare current file to every entry in the reference DB
            for col_idx, db_word in enumerate(db_keys):
                reference_spectrogram = db_data[db_word]
                dist_matrix[row_idx, col_idx] = compute_dtw_distance(reference_spectrogram, test_spectrogram)
            row_idx += 1

    return dist_matrix