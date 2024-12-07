import numpy as np

class CoordinateShuffler:
    def __init__(self, rng):
        """
        Initialize with a random number generator.
        
        Parameters:
            rng (np.random.Generator): A NumPy random number generator.
        """
        self.rng = rng

    def shuffle_coordinates(self, coordinates):
        """
        Shuffle the coordinates deterministically with a fresh seed each time.

        Parameters:
            coordinates (list): A list of (x, y) tuples.

        Returns:
            list: Shuffled list of (x, y) tuples.
        """
        # Generate a deterministic fresh seed
        fresh_seed = self.rng.integers(0, 2**32)
        fresh_rng = np.random.default_rng(fresh_seed)
        # Shuffle coordinates with the fresh RNG
        fresh_rng.shuffle(coordinates)
        return coordinates

def generate_unique_random_coordinates(m, n, num_points, shuffler):
    """
    Generate unique random (x, y) coordinates for a 2D array of size m x n.

    Parameters:
        m (int): Number of rows.
        n (int): Number of columns.
        num_points (int): Number of random points to generate.
        shuffler (CoordinateShuffler): Instance of CoordinateShuffler.

    Returns:
        List of tuples representing unique random (x, y) coordinates.
    """
    total_points = m * n
    if num_points > total_points:
        raise ValueError("num_points exceeds the total number of unique points in the array.")
    
    # Generate the full Cartesian product of coordinates
    all_coordinates = [(i, j) for i in range(m) for j in range(n)]
    
    # Shuffle the coordinates
    shuffled_coordinates = shuffler.shuffle_coordinates(all_coordinates.copy())
    
    # Select the first num_points coordinates
    return shuffled_coordinates[:num_points]

# Example usage
m, n = 10, 15  # Size of the array
num_points = 5  # Number of random points
seed = 42  # Seed for reproducibility

# Initialize a single RNG and shuffler
rng = np.random.default_rng(seed)
shuffler = CoordinateShuffler(rng)

# Generate coordinates twice using the same shuffler
random_coordinates_1 = generate_unique_random_coordinates(m, n, num_points, shuffler)
random_coordinates_2 = generate_unique_random_coordinates(m, n, num_points, shuffler)

print("Unique Random Coordinates (1st call):", random_coordinates_1)
print("Unique Random Coordinates (2nd call):", random_coordinates_2)
