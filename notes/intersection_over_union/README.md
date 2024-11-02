# Evaluating Segmentation Accuracy in Image Processing

When working on image segmentation tasks in machine learning, it's crucial to assess how well your model's predictions match the ground truth (the correct segmentation). Here are two key metrics used to evaluate segmentation accuracy, explained step-by-step for beginners.

---

## **1. Intersection Over Union (IOU)**

**What is IOU?**  
Intersection Over Union (IOU) is one of the most commonly used metrics to evaluate how well the predicted segmentation overlaps with the actual (ground truth) segmentation.

### **Formula - IOU**
\[
\text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]
- **Area of Overlap**: The region where both the predicted and actual segments overlap.
- **Area of Union**: The total area covered by both the predicted and actual segments combined.

### **How IOU is Calculated**
- IOU score ranges from **0** to **1**:
  - **1** means perfect segmentation (the predicted and actual segments completely overlap).
  - **0** means no overlap at all (the prediction and ground truth are completely different).

### **Python Example for IOU Calculation**

```python
import numpy as np

# Example masks: Ground truth and Predicted (4x4 matrix)
ground_truth = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 1]])

predicted = np.array([[1, 0, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])

# IOU Calculation
intersection = np.logical_and(ground_truth, predicted)  # Overlap region
union = np.logical_or(ground_truth, predicted)          # Union of both regions

iou_score = np.sum(intersection) / np.sum(union)        # IOU formula

print(f"IOU Score: {iou_score:.2f}")
```

### **Output**
```
IOU Score: 0.67
```

This **IOU score of 0.67** indicates that 67% of the predicted area overlaps with the ground truth.

---

## **2. Pixel Accuracy**

**What is Pixel Accuracy?**  
Pixel Accuracy is a simpler metric that measures the percentage of correctly predicted pixels in the entire image. It compares each pixel of the predicted mask with the ground truth and calculates how many of them are identical.

### **Formula:**
\[
\text{Pixel Accuracy} = \frac{\text{Correctly Predicted Pixels}}{\text{Total Pixels}}
\]

### **Python Example for Pixel Accuracy Calculation:**

```python
# Pixel Accuracy Calculation
correct_pixels = np.sum(ground_truth == predicted)  # Pixels where prediction matches ground truth
total_pixels = ground_truth.size                    # Total number of pixels

pixel_accuracy = correct_pixels / total_pixels      # Pixel accuracy formula

print(f"Pixel Accuracy: {pixel_accuracy:.2f}")
```

### **Output:**
```
Pixel Accuracy: 0.88
```

This **Pixel Accuracy of 88%** means that 88% of the total pixels in the image were predicted correctly.

---

## **3. Comparison of IOU vs Pixel Accuracy**

| **Metric**      | **Purpose**                                       | **Usefulness**                   |
|-----------------|---------------------------------------------------|----------------------------------|
| **IOU**         | Measures overlap between predicted and actual regions. | More useful for object-based evaluation. |
| **Pixel Accuracy** | Measures overall correctness of pixel predictions. | Simpler but may not reflect object-level accuracy well. |

**Key Points for Beginners:**
- **IOU** is generally preferred in image segmentation tasks because it focuses on object overlap, which is a more direct measure of segmentation quality.
- **Pixel Accuracy** can be misleading in cases where the background is large compared to the objects. Even if the objects are segmented poorly, you might still get a high pixel accuracy if the background is correctly predicted.
- **For segmentation tasks**, always consider **IOU** as it gives a more accurate measure of how well the objects are segmented.
- **Pixel accuracy** is easier to calculate but less insightful when objects are small compared to the image size.

---

## **Integrating IOU and Pixel Accuracy in a Segmentation Pipeline**

### **1. Data Preparation**
Before evaluating the model, you need:
- **Ground truth masks**: The correct segmentation for each image.
- **Predicted masks**: What your model predicts for each image.

Both should be represented as binary (or multi-class) masks, where:
- **1** represents the object (or a specific class for multi-class segmentation).
- **0** represents the background (or other classes).

---

### **2. Training the Model**
Your model will output predictions in the form of masks for each image in the dataset. If you’re working on binary segmentation, each pixel will be either **0** or **1**. For multi-class segmentation, each pixel will have the class number predicted by the model.

The output of the model for an image might look like this:
```python
predicted_mask = model.predict(image)
```

---

#### **3. Evaluate Model Performance Using IOU and Pixel Accuracy**

After getting the predicted mask, the evaluation metrics (IOU and Pixel Accuracy) can be calculated and monitored.

#### **IOU Calculation for Binary Segmentation:**

In binary segmentation, IOU can be calculated for each image after the model prediction. Let’s assume you have a ground truth mask and a predicted mask for one image:

```python
import numpy as np

def calculate_iou(ground_truth, predicted):
    # Calculate the intersection and union
    intersection = np.logical_and(ground_truth, predicted)
    union = np.logical_or(ground_truth, predicted)
    
    # IOU formula
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
```

#### **Pixel Accuracy Calculation:**

Pixel accuracy can also be computed similarly:

```python
def calculate_pixel_accuracy(ground_truth, predicted):
    correct_pixels = np.sum(ground_truth == predicted)
    total_pixels = ground_truth.size
    return correct_pixels / total_pixels
```

---

#### **4. Loop Through the Dataset**

You need to loop through all the images in your dataset and calculate IOU and pixel accuracy for each one. Then, you can compute the average IOU and pixel accuracy across the dataset for performance evaluation.

```python
# Loop through each image and ground truth in your dataset
iou_scores = []
pixel_accuracies = []

for image, ground_truth_mask in dataset:  # Assuming dataset has image and corresponding ground truth mask
    # Predict mask using your model
    predicted_mask = model.predict(image)
    
    # Calculate IOU and Pixel Accuracy for each image
    iou = calculate_iou(ground_truth_mask, predicted_mask)
    pixel_accuracy = calculate_pixel_accuracy(ground_truth_mask, predicted_mask)
    
    # Append results to lists
    iou_scores.append(iou)
    pixel_accuracies.append(pixel_accuracy)

# Calculate mean IOU and Pixel Accuracy
mean_iou = np.mean(iou_scores)
mean_pixel_accuracy = np.mean(pixel_accuracies)

print(f"Mean IOU: {mean_iou:.2f}")
print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.2f}")
```

---

#### **5. Use Multi-class Segmentation Metrics (Optional)**

If you are working on **multi-class segmentation**, you'll need to calculate the IOU for each class separately and take the mean over all classes (known as **Mean IOU**).

Here’s how you can modify the IOU calculation for **multi-class segmentation**:

```python
def calculate_mean_iou(ground_truth, predicted, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        # Create binary masks for each class
        ground_truth_class = (ground_truth == cls)
        predicted_class = (predicted == cls)
        
        # Calculate IOU for the current class
        intersection = np.logical_and(ground_truth_class, predicted_class)
        union = np.logical_or(ground_truth_class, predicted_class)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        
        iou_scores.append(iou)
    
    # Return mean IOU
    return np.mean(iou_scores)

# Example for multi-class segmentation with 3 classes
mean_iou = calculate_mean_iou(ground_truth_mask, predicted_mask, num_classes=3)
print(f"Mean IOU: {mean_iou:.2f}")
```

---

### **3. Integrating with Machine Learning Frameworks**

In popular machine learning frameworks like **Keras** or **PyTorch**, you can integrate these metrics as part of the evaluation process:

**PyTorch Example**:

For **PyTorch**, you can define a similar function and calculate IOU as part of the validation loop:

```python
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.int()
    labels = labels.int()
    
    intersection = (outputs & labels).float().sum((1, 2))  # Intersection over batch
    union = (outputs | labels).float().sum((1, 2))         # Union over batch
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)       # IOU calculation
    return iou.mean()                                      # Mean IOU for the batch
```

---

### **4. Visualize Results**

To better understand how your model is performing, you can visualize the predictions along with their respective ground truth:

```python
import matplotlib.pyplot as plt

# Function to visualize prediction vs ground truth
def plot_predictions(image, ground_truth, predicted):
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    
    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(ground_truth, cmap='gray')
    
    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted, cmap='gray')
    
    plt.show()

# Call this function for an image, ground truth mask, and predicted mask
plot_predictions(image, ground_truth_mask, predicted_mask)
```

---

### **5. Monitor Metrics During Training**

Finally, you should monitor IOU and pixel accuracy throughout the training process. This can be done using:

- **PyTorch**: Use libraries like `Matplotlib` or TensorBoard to log and visualize the metrics.
- **IOU** is the go-to metric for evaluating segmentation performance, especially in object detection.
- **Pixel Accuracy** provides a simpler but less detailed view of segmentation accuracy.
- Integrating these metrics into your training and evaluation pipeline will allow you to monitor model performance effectively and make adjustments as necessary.
