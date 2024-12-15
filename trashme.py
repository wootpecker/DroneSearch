import numpy as np
import create_dataloader
import utils
import model_builder

DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge","EncoderDecoder"]
MODEL_TYPES=["VGG24","CNN","VGGVariation","UnetEncoderDecoder","SimpleUNet"] #model_types of model_builder -> Simple CNN, VGGVariation(2 Conv Blocks), VGG24(more complex 3 Conv Blocks)








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








import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dataset class
class SourceDataset(Dataset):
    def __init__(self, data, locations, transform=None):
        """
        data: List of input images [N, 1, 24, 24]
        locations: List of source locations [(h, w)] for each image
        """
        self.data = data
        self.locations = locations
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        mask = np.zeros((24, 24))
        h, w = self.locations[idx]
        mask[h, w] = 1  # Set the source pixel to 1
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# Example data
N = 1000  # Number of samples
data = [np.random.uniform(0, 1, (1, 24, 24)) for _ in range(N)]
locations = [(np.random.randint(0, 24), np.random.randint(0, 24)) for _ in range(N)]


dataloader_type=DATASET_TYPES[6]
# Dataset and DataLoader
dataset = SourceDataset(data, locations)
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
dataloader,test_dataloader,valid_dataloader,classes = create_dataloader.create_dataloader(dataloader_type)

# Model: A simple UNet-like structure
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Output probabilities
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss, and optimizer
model = SimpleUNet()
model = model_builder.choose_model(MODEL_TYPES[3],output_shape=1)
model = model.to('cuda')  # Move to GPU if available
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to('cuda'), masks.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)  # Add channel dimension to masks
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

utils.save_model(model=model,target_dir=dataloader_type,model_type="SimpleUNet")
#utils.plot_loss_curves(mode_results)
print("Training complete.")



import matplotlib.pyplot as plt
dataloader=test_dataloader
model.eval()
with torch.no_grad():
    for images, masks in dataloader:
        images = images.to('cuda')
        predictions = model(images).cpu().numpy()
        plt.subplot(1, 3, 1); plt.imshow(images[0, 0].cpu(), cmap='gray'); plt.title('Input')
        plt.subplot(1, 3, 2); plt.imshow(masks[0].squeeze(0), cmap='gray'); plt.title('Target')
        plt.subplot(1, 3, 3); plt.imshow(predictions[0, 0], cmap='gray'); plt.title('Prediction')
        plt.show()
        break