"""

This script performs the following major steps:
1. Import dependencies
2. Define StarSamplerGraphDataset for loading graph data from an HDF5 file
3. Define StellarGNN model architecture
4. Implement the train() function to train, validate, and save the model,
   plot loss curves, and evaluate on a test set
5. Provide an entry point to start training when the script is executed
"""

import h5py
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Custom Dataset Class for Loading Graph Data from an HDF5 file
# ----------------------------
class StarSamplerGraphDataset(Dataset):
    """
    Dataset that loads node features and graph labels from an HDF5 file,
    builds k-nearest-neighbor graphs, and returns PyG Data objects.
    """
    def __init__(self, h5_path, k=20):
        super().__init__()
        self.h5_path = h5_path  # Path to the HDF5 file
        self.k = k  # Number of neighbors for each node to form edges
        # Load features, labels, and ptr array from HDF5 file
        with h5py.File(self.h5_path, 'r') as f:
            self.features = f['features'][:]  # Node features
            self.labels = f['labels'][:]      # Graph-level labels
            self.ptr = f['ptr'][:]            # Index pointer for graph separation

    def len(self):
        """
        Return the number of graphs in the dataset,
        determined by the ptr array (graphs = len(ptr) - 1).
        """
        return len(self.ptr) - 1

    def get(self, idx):
        """
        Retrieve the graph at index idx as a PyG Data object.

        Constructs node features tensor, label tensor, and edge_index
        via k-NN on the node positions.
        """
        # Get node features for graph at idx
        start, end = self.ptr[idx], self.ptr[idx + 1]
        x_np = self.features[start:end]

        # Convert features to tensor
        x = torch.tensor(x_np, dtype=torch.float32)

        # Convert labels to tensor, add extra dimension to match shape [1, 5]
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        # Build k-nearest neighbor graph (edges)
        nbrs = NearestNeighbors(n_neighbors=min(self.k, len(x_np))).fit(x_np)
        _, indices = nbrs.kneighbors(x_np)

        # Construct edge_index list
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                edge_index.append([i, j])

        # Convert edge list to tensor and transpose to [2, num_edges] format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Return as PyG Data object
        return Data(x=x, edge_index=edge_index, y=y)

# ----------------------------
# GNN Model Definition: simple 2-layer GCN followed by fully connected layer
# ----------------------------
class StellarGNN(nn.Module):
    """
    Graph neural network consisting of two GCNConv layers with ReLU activations
    followed by a global mean pool and a linear layer to produce 5 outputs.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=5):
        super(StellarGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)   # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.fc = nn.Linear(hidden_dim, output_dim)   # Final linear layer

    def forward(self, data):
        """
        Forward pass: 
        - Apply two GCN layers with ReLU,
        - Perform global mean pooling on node embeddings,
        - Apply final linear layer to obtain graph-level output.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ----------------------------
# Training function
# ----------------------------
def train():
    """
    Train the StellarGNN on data/training.h5 with an 80/20 split for validation.
    Implements early stopping with patience=10, saves best model to best_model.pt,
    plots training and validation loss curves, and evaluates on data/test.h5 if present.
    """
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Load full dataset
    full_dataset = StarSamplerGraphDataset("data/training.h5")

    # Split into train and validation sets (80/20 split)
    total_len = len(full_dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    print(f"Dataset split: {train_len} training, {val_len} validation", flush=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Initialize model, optimizer, and loss function
    model = StellarGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    # Variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    patience = 10  # Stop if no improvement after 10 epochs

    # For plotting loss curves
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, 301):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
                pred = model(batch)
                loss = loss_fn(pred, batch.y.to(device))
                val_loss += loss.item()

        # Compute average losses
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    # After training, load the best model and save it
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "best_model.pt")

    print("Training complete. Best validation loss:", best_val_loss, flush=True)

    # Plot training and validation loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig("loss_curve.png")

    # Test set evaluation
    if os.path.exists("data/test.h5"):
        test_dataset = StarSamplerGraphDataset("data/test.h5")
        test_loader = DataLoader(test_dataset, batch_size=1)

        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
                pred = model(batch)
                loss = loss_fn(pred, batch.y.to(device))
                test_loss += loss.item()

                # Save predictions and labels
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

        # Stack and save predictions and ground truths
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        np.savetxt("data/predictions.csv", all_preds, delimiter=",",
                   header="Pred1,Pred2,Pred3,Pred4,Pred5", comments='')
        np.savetxt("data/ground_truths.csv", all_labels, delimiter=",",
                   header="GT1,GT2,GT3,GT4,GT5", comments='')

# ----------------------------
# Entry point for the script
# ----------------------------
if __name__ == "__main__":
    train()
