import numpy as np

# Example ground truth and predicted masks
ground_truth = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 1]])

predicted = np.array([[1, 0, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])

# Calculate IOU
intersection = np.logical_and(ground_truth, predicted)  # Areas where prediction and ground truth overlap
union = np.logical_or(ground_truth, predicted)          # Areas where there is either prediction or ground truth

iou_score = np.sum(intersection) / np.sum(union)        # Sum the intersection and union areas and calculate IOU

print(f"IOU Score: {iou_score:.2f}")

# Pixel Accuracy
correct_pixels = np.sum(ground_truth == predicted)
total_pixels = ground_truth.size

pixel_accuracy = correct_pixels / total_pixels

print(f"Pixel Accuracy: {pixel_accuracy:.2f}")

