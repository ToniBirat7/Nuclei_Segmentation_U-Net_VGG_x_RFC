# Automated Nuclei Detection in Microscopy Images
## A Comparative Analysis of Traditional and Deep Learning Approaches
*Presenter: Birat Gautam*

---

## Slide 1: Introduction
**Title: Automated Nuclei Detection in Microscopy Images: A Comparative Analysis**

**Visual**: Simple title slide with a microscopy image containing nuclei in the background (semi-transparent), university logo, presenter name, and date.

**Script**:
"Good morning everyone. Today I'll be presenting our research on automated nuclei detection in microscopy images. This study compares traditional image processing techniques with deep learning approaches, specifically focusing on U-Net architecture. The ability to automatically identify cell nuclei is crucial for accelerating biomedical research and drug discovery processes."

---

## Slide 2: The Challenge of Nuclei Segmentation
**Title: Why Automated Nuclei Segmentation Matters**

**Visual**: 
- Split screen showing: 
  - Left: Manual annotation process (researcher at microscope)
  - Right: Results of automated detection with time comparison

**Script**:
"Nuclei segmentation is a cornerstone of modern biomedical research. Manual analysis of microscopy images is time-consuming, requiring hours of focused attention from skilled professionals. This process introduces unavoidable subjectivity and becomes increasingly impractical as imaging technologies advance. Automated approaches not only accelerate research timelines but also enable more consistent processing of large datasets, which is critical for drug discovery where rapid assessment of compound effects can significantly accelerate screening processes. In cancer research, precise nuclei segmentation provides crucial insights into disease progression and treatment responses. We can solve this problem with the use of AI Image Segmentation"  

---

## Slide 3: Dataset Overview
**Title: 2018 Data Science Bowl Dataset**

**Visual**: 
- Example images from dataset showing diversity
- Folder structure diagram showing imageId, images, and masks organization

**Script**:
"To solve this problem I've chosen the 2018 Data Science Bowl dataset, which contains diverse microscopy images with accompanying nuclei masks. Each image is assigned a unique identifier that serves as the basis for the folder structure. The dataset presents several challenges including varying imaging conditions, different cell types with distinct nuclear morphologies, presence of overlapping nuclei, and diverse background textures. We employed stratified sampling to ensure representative distribution across training and test sets, with 90% of data (670 images) allocated for training and 10% (70 images) for testing."

---

## Slide 4: Exploratory Data Analysis
**Title: Exploratory Data Analysis Findings**

**Visual**: 
- Bar charts showing image dimension distribution
- Histogram of nuclei per image
- Visualization of contrast/brightness variation

**Script**:
"Our exploratory analysis revealed substantial diversity in image characteristics. We observed significant variation in image dimensions, ranging from 256×256 to 1024×1024 pixels. Nuclear density showed a mean of 71 nuclei per image with a standard deviation of 9. Images exhibited considerable variation in contrast and brightness, with normalized contrast values ranging from 0.4 to 0.9. Approximately 27% of images contained significant nuclear overlap, presenting particular challenges for boundary delineation. These findings informed our preprocessing strategy and architectural decisions."

---

## Slide 5: Data Preprocessing Pipeline
**Title: Mask Combining and Image Preprocessing**

**Visual**: 
- Flowchart showing preprocessing steps
- Before/after examples of mask combining
- Gaussian blur effect demonstration

**Script**:
"A critical preprocessing challenge was consolidating multiple individual nuclear masks into unified segmentation masks. The dataset structure presented individual mask files for each nucleus, requiring careful preprocessing to create unified training masks. We implemented a pixel-wise maximum operation to combine these individual masks. For image preprocessing, we applied Gaussian filtering with a sigma value of 1.5, carefully calibrated to reduce noise while preserving essential edge information. Additionally, we implemented normalization to standardize pixel intensity distributions. These preprocessing steps were crucial for both our traditional and deep learning approaches."

---

## Slide 6 and 7: Model Selection Rationale
**Title: Approach Selection: Traditional vs. Deep Learning**

**Visual**: 
- Comparison table of approaches with characteristics
- Small images representing each approach

**Script**:
"Based on our exploratory analysis and literature review, we implemented two distinct approaches to nuclei segmentation. Our traditional approach combines VGG-based feature extraction with Random Forest classification, leveraging the strengths of both deep convolutional feature extraction and classical machine learning flexibility. For the deep learning approach, we selected the U-Net architecture due to its proven effectiveness in biomedical image segmentation. The diversity in nuclear morphology, image quality, and the presence of overlapping nuclei informed our decision to implement both approaches for comparative analysis."

**Visual**: 
- Architecture diagram showing VGG feature extraction flow
- Sample feature maps from VGG (colorized)
- Random Forest decision visualization

**Script**:
"Our traditional approach leveraged the feature extraction capabilities of convolutional neural networks combined with classical machine learning classification. We utilized the VGG architecture as a feature extractor, specifically the first two convolutional layers which generated 64 feature maps at 256×256 resolution. These feature maps were then fed into a Random Forest Classifier configured with 100 trees and a maximum depth of 20. This architecture, comprising 38,720 non-trainable parameters, served as an effective feature extractor while the Random Forest provided flexibility in decision boundary formation. We implemented automatic CPU fallback capability to manage memory constraints while maintaining processing throughput."

---

## Slide 8: Model Performance Metrics
**Title: Performance Comparison: Random Forest vs U-Net**

**Visual**: 
- Side-by-side bar charts comparing key metrics:
  - Accuracy
  - IoU
  - Precision
  - Recall
  - F1 Score

**Script**:
"The performance comparison between our two approaches reveals compelling differences. The VGG-based Random Forest approach achieved an accuracy of 0.954, precision of 0.916, recall of 0.894, F1 score of 0.905, and IoU of 0.826 on training data. However, the U-Net implementation demonstrated superior performance with an accuracy of 0.971, IoU of 0.883, precision of 0.92, and recall of 0.90. Most notably, when facing challenging conditions like overlapping nuclei or low contrast, the traditional approach showed a 15-20% performance reduction, while U-Net demonstrated remarkable robustness with only 5-8% performance reduction under similar conditions."

**Final Note**

"Our approach included VGG x RFC in we used VGG to extract features and train RFC but it did not perform well with IoU of 0.82 and Accuracy of 0.95 but it struggled with the overlapping Nuclei cases. U-Net had better performance with IoU of 0.88 and Accuracy of 0.971, therefore we will stick to the implementation of U-Net for rest of our presentation"

---

## Slide 9: Why U-Net Architecture Performed Well?
**Title: U-Net Architecture Explanation**

**Visual**: 
- Detailed U-Net architecture diagram showing:
  - Contracting path
  - Expanding path
  - Skip connections
  - Layer dimensions

**Script**:
"Our U-Net implementation features a symmetrical encoder-decoder structure with skip connections, specifically designed for the challenges of nuclei segmentation. The contracting path consists of five sequential blocks, each featuring dual convolutional layers followed by dropout for regularization. Starting with 16 feature maps, subsequent blocks progressively increase feature complexity to 32, 64, 128, and 256 maps while reducing spatial dimensions through max-pooling. The expanding path mirrors this structure with transposed convolutions for upsampling. Skip connections preserve fine-grained spatial information crucial for precise boundary delineation. This architecture contains approximately 1.94 million trainable parameters and employs a composite loss function combining Binary Cross-entropy with the Dice coefficient."

---

## Slide 10: Visualizing U-Net Layers
**Title: Input Actual Predicted**

**Visual**: 
- Detailed U-Net architecture diagram showing:
  - Contracting path
  - Expanding path
  - Skip connections
  - Layer dimensions

**Script**:
"We've an input nuclei image that goes first through the encoding path. As we can see in the image that 3rd layer with 32 filters is extracting the features from the input image. But in the decoding layer 15 with 32 filter it has regained it's spatial information (Shape, Size) which is needed for Segmentation problem (other CNN model do not regain their shape) and segmentation has started as we can see in the image. This way our model segments the given input image"

---

## Slide 11: Model Explanation - Why Not a Black Box?
**Title: Explaining Model Decisions: Beyond the Black Box**

**Visual**: 
- Conceptual illustration of model interpretability
- Diagram showing explanation techniques applied to CNN

**Script**:

"Now, comes an interesting part. But how can we trust our model that it's prediction is correct? How can we rely on this model?"

"Deep learning models including U-Net are often regarded as 'black boxes', making their interpretability crucial for ensuring trust and reliability in medical applications. To address this concern, I implemented two complementary explanation techniques: Gradient-weighted Class Activation Mapping (Grad-CAM) and Layer-wise Relevance Propagation (LRP). These techniques provide insights into which image regions influenced segmentation decisions, allowing us to verify that our model's predictions are based on biologically relevant features rather than artifacts or spurious correlations."

---

## Slide 12: Grad-CAM Analysis
**Title: Gradient-weighted Class Activation Mapping Results**

**Visual**: 
- Original test images
- Corresponding Grad-CAM heatmaps
- Combined overlay showing attention regions

**Script**:
"Grad-CAM analysis revealed our model's attention mechanisms across different cell densities and imaging conditions. As shown in these attention heatmaps, the model consistently focused on nuclear boundaries with high attention intensity (red regions) while demonstrating lower attention (blue regions) to background tissues. This pattern was consistent across diverse nuclear morphologies, confirming the model's ability to distinguish nuclear structures regardless of size, shape, or imaging conditions. The 13th convolutional filter, as visualized here, proved particularly influential in boundary detection."

---

## Slide 13: Layer-wise Relevance Propagation
**Title: LRP: Fine-grained Decision Explanation**

**Visual**: 
- Test images with LRP overlay
- Zoomed sections showing pixel-level attributions
- Comparison with ground truth boundaries

**Script**:
"While Grad-CAM provides broader activation regions, Layer-wise Relevance Propagation offers fine-grained pixel-wise explanations of segmentation decisions. Our LRP analysis revealed that the U-Net effectively leveraged hierarchical features, with deeper layers contributing to boundary refinement. The relevance map demonstrated a mean relevance of 0.45 across nuclear regions with 31,988 positively contributing regions identified. The visualization confirms that the model's decisions relied primarily on genuine nuclear features rather than artifacts or background noise, with a confidence score of 0.902. This validates that our U-Net model's segmentation decisions are biologically relevant."

---

---

## Slide 14: Conclusion and Future Work
**Title: Conclusion and Future Directions**

**Visual**: 
- Summary points with icons
- Future research directions
- Acknowledgments and contact information

**Script**:
"In conclusion, our study demonstrates the superior performance of deep learning approaches, particularly modified U-Net architectures, for automated nuclei detection in microscopy images. The comprehensive evaluation across diverse metrics and challenging conditions establishes U-Net as the preferred methodology for this critical task in biomedical image analysis. Future work could explore further architectural modifications to address specific challenges in nuclei segmentation, integration with downstream analysis pipelines, and extension to 3D microscopy datasets. I want to thank my supervisors and colleagues for their valuable guidance throughout this research. Are there any questions about our methodology or findings?"

---

Thank You