# Import necessary libraries
import numpy as np

# Function to calculate the Gray Level Co-occurrence Matrix (GLCM)
def calculate_glcm(image, distances, angles, levels):
    height, width = image.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.int32)
    
    for i, distance in enumerate(distances):
        for j, angle in enumerate(angles):
            dx = int(np.round(np.cos(angle) * distance))
            dy = int(np.round(np.sin(angle) * distance))
            
            for y in range(height):
                for x in range(width):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        current_pixel = image[y, x]
                        neighbor_pixel = image[y + dy, x + dx]
                        glcm[current_pixel, neighbor_pixel, i, j] += 1

    return glcm

# Function to extract features from GLCM
def extract_features(glcm):
    contrast = np.sum(np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])[None, :]) * glcm, axis=(0, 1))
    dissimilarity = np.sum(np.abs(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])[None, :]) * glcm, axis=(0, 1))
    homogeneity = np.sum(glcm / (1.0 + np.abs(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])[None, :])), axis=(0, 1))
    energy = np.sum(glcm ** 2, axis=(0, 1))
    correlation = np.sum((np.arange(glcm.shape[0])[:, None] - np.mean(np.arange(glcm.shape[0]))) * (np.arange(glcm.shape[1])[None, :] - np.mean(np.arange(glcm.shape[1]))) * glcm, axis=(0, 1)) / (np.std(np.arange(glcm.shape[0])) * np.std(np.arange(glcm.shape[1])))
    
    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    return features.flatten()

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to implement K-Nearest Neighbors (KNN)
def knn_classify(train_features, train_labels, test_feature, k):
    distances = [euclidean_distance(test_feature, train_feature) for train_feature in train_features]
    k_indices = np.argsort(distances)[:k]
    k_labels = [train_labels[i] for i in k_indices]
    return max(set(k_labels), key=k_labels.count)

# Example usage
if __name__ == "__main__":
    # Example grayscale images
    train_images = [np.random.randint(0, 256, (4, 4), dtype=np.uint8) for _ in range(5)]
    train_labels = [0, 1, 0, 1, 0]
    test_image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)

    # Parameters for GLCM
    distances = [1]
    angles = [0]
    levels = 256

    # Extract features from training images
    train_features = []
    for image in train_images:
        glcm = calculate_glcm(image, distances, angles, levels)
        features = extract_features(glcm)
        train_features.append(features)
    
    # Extract features from test image
    test_glcm = calculate_glcm(test_image, distances, angles, levels)
    test_features = extract_features(test_glcm)

    # Classify the test image using KNN
    k = 3
    predicted_label = knn_classify(train_features, train_labels, test_features, k)
    print(f"Predicted Label: {predicted_label}")
