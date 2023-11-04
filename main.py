import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn import metrics
import tkinter as tk
from tkinter import ttk, filedialog

# Load MNIST dataset
mnist = fetch_openml("mnist_784")
images, labels = mnist.data, mnist.target
images = images / 255.0  # Normalize pixel values to the range [0, 1]

def kmeans_clustering(images, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(images)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    return cluster_centers, cluster_labels

def display_clusters(cluster_centers, k, cluster_labels):
    fig, ax = plt.subplots(1, k, figsize=(8, 2))
    plt.suptitle(f"K-Means Clustering (k={k})")

    for j in range(k):
        ax[j].imshow(cluster_centers[j].reshape(28, 28), cmap='gray')
        ax[j].set_title(f'Cluster {j}')
        ax[j].axis('off')

    plt.show()

def display_cluster_contents(cluster_labels, k, labels):
    for i in range(k):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_size = len(cluster_indices)

        print(f"Cluster {i}: Size={cluster_size}")
        
        if cluster_size > 0:
            cluster_images = images[cluster_indices]
            cluster_labels = labels[cluster_indices]

            fig, ax = plt.subplots(1, cluster_size, figsize=(12, 1))
            for j in range(cluster_size):
                ax[j].imshow(cluster_images[j].reshape(28, 28), cmap='gray')
                ax[j].set_title(f'Label: {cluster_labels[j]}')
                ax[j].axis('off')

            plt.show()

def evaluate_clustering(cluster_labels, labels):
    silhouette = metrics.silhouette_score(images, cluster_labels)
    adjusted_rand = metrics.adjusted_rand_score(labels, cluster_labels)
    completeness = metrics.completeness_score(labels, cluster_labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {adjusted_rand:.4f}")
    print(f"Completeness Score: {completeness:.4f}")

def browse_button():
    file_path = filedialog.askopenfilename()
    if file_path:
        k = int(combo_k.get())
        cluster_centers, cluster_labels = kmeans_clustering(images, k)
        display_clusters(cluster_centers, k, cluster_labels)
        display_cluster_contents(cluster_labels, k, labels)
        evaluate_clustering(cluster_labels, labels)

# Create a Tkinter window
root = tk.Tk()
root.title("MNIST Image Clustering")

frame = ttk.Frame(root, padding=10)
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

combo_k = ttk.Combobox(frame, values=[5, 10, 20], state='readonly')
combo_k.set(5)
combo_k.grid(column=0, row=0, padx=10, pady=10)

browse_button = ttk.Button(frame, text="Browse Image", command=browse_button)
browse_button.grid(column=1, row=0, padx=10, pady=10)

root.mainloop()
