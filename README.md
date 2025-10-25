# building-extraction

## Task Summary

This task focuses on the **automatic extraction of building footprints** from **satellite imagery**, a fundamental component of urban Geographic Information Systems (GIS). The objective is to accurately detect and outline buildings despite variations in **shape, color, size**, and **environmental conditions** across different regions.

A key challenge lies in achieving **strong generalization** from limited training data. The provided dataset covers the **Tokyo area**, while testing extends to other regions in Japan. Models must therefore maintain high accuracy and adaptability under diverse geographic and visual conditions.

- **Competition and Data**
	- [Kaggle challenge and COCO-format dataset](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
		- Includes training data from the Tokyo area and evaluation on other regions

- **Hugging Face Resources**
    - [Profile](https://huggingface.co/tomascanivari)
    - Datasets and Models:
        - [Converted Dataset](https://huggingface.co/datasets/tomascanivari/buildings-extraction-coco-hf)
        - [Final Model](https://huggingface.co/tomascanivari/mask2former-swin-large-coco-instance-finetuned-buildings)


### Objective
Develop a **reliable, cost-effective, and scalable** approach for **nationwide building footprint extraction**, demonstrating strong **transferability** and **real-world applicability**.

### Challenge Evaluation Metrics

The evaluation is based on the **object-wise F1-score**, which measures the accuracy of extracted building footprints.

Each extracted building (predicted polygon) is compared with the ground-truth (GT) building polygon using the **Intersection over Union (IoU)** metric — the ratio of the overlapping area to the combined area of both polygons.

- A prediction is considered **correct (True Positive, TP)** if its **IoU ≥ 0.5** with a GT building.  
- Predictions with **IoU < 0.5** or without a corresponding GT are counted as **False Negatives (FN)**.  
- Any **extra or incorrect** extracted buildings are treated as **False Positives (FP)**.  

$\text{Precision} = \frac{TP}{TP + FP}$

$\text{Recall} = \frac{TP}{TP + FN}$

$\text{F1-score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

The final ranking is determined by the **average F1-score** over the test set (unweighted). The highest-scoring model achieves the best overall performance.

## Methodology

I approached this task in three steps:

1. **Dataset Preparation**: described in 'nb_00_dataset_preparation.ipynb'.
2. **Model Training**: described in 'nb_01_mask2former_train.ipynb'.
3. **Model Inference and Evaluation**: described in 'nb_02_mask2former_inference_evaluation.ipynb'

### Dataset Preparation

From the original **COCO-format** annotations and images, a new dataset ([Converted Dataset](https://huggingface.co/datasets/tomascanivari/buildings-extraction-coco-hf)) was created containing both **images** and **annotation mask images**.

Each mask encodes:
- **Semantic segmentation** — stored in the **Red channel**  
  - `0`: background  
  - `1`: building  
- **Instance segmentation** — stored in the **Green channel**  
  - `0`: background  
  - `1...N`: individual building instances  

These masks were generated from the COCO-format polygon annotations.  
The resulting dataset is **ready-to-use** for most **instance segmentation tasks**, as it provides both:
- COCO-format annotations  
- Image–Mask pairs  

Together, these two formats ensure compatibility with a wide range of segmentation frameworks and pipelines.

### Model Training

Clearly explained in the 'nb_01_mask2former_train.ipynb' file:

1. Load the HF dataset
2. Create a PyTorch dataset compatible with the 'mask2former' format and data augmentation
3. Load the processor and model for the 'facebook/mask2former-swin-large-coco-instance' pre-trained model
4. Finetune for 13 epochs using the hyper-parameters described on the file.

The best model was from epoch 12 as seen in the following train and validation loss by epoch: