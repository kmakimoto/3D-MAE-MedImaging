# Code and data to compare embeddings extracted from pretrained TANGETINE model using 1. SimpleITK BSpline and 2. MONAI Bilinear for CT volume resampling.

Two sets of embeddings have been extracted for a subset of COPDGene individuals (copdgene_29_sid_per_gold.csv). 

*Both methods used the same model parameters:
model: vit_large_patch16_yo
num_classes=1
drop_path_rate=0
global_pool=True

*Both methods use the same checkpoint: d594de5ec0f59c55a1c20ec9f6ad3bef  

*Both methods implement the same preprocessing, but using different libraries (SimpleITK vs. MONAI): Resample to (256, 256, 256), [-1200, 800] HU Clip, min-max HU normalization.

*Both methods extract the embeddings using the models_vit.py, specificially the model.forward_features(x) function, which concatentats the raw (cls+pooled).

## Embedding Extractions Methods
1) Using mains_predict/extract_encoder_embeddings.py
    - Settings: embedding_type=concat, global_pool=True, cudnn.benchmark = False
    - Embeddings saved as: test_embeddings_20260909.csv

2) Using embeddings_Test/inference_embeddings.py
    - In-house script with slighlty modified pre-processing, using the models_vit.py from TANGERINE and model checkpoint.
    - Embeddings savedd as: csv_b_from_npz.csv

## Sanity check
Modified the inference_embeddings.py -> inference_embeddings_revised.py to use the same settings and libraries as extract_encoder_embeddings.py to ensure the same embeddings are generated.
    - Changes: bfloat16 autocast -> float 32, Removed RAS reorientation, MONAI bilinear -> SimpleITK BSpline
    - These changes results in the embeddings being identical to the ones extracted using extract_encoder_embeddings.py

## Similarity Analysis 
Ran analyses using the two sets of embeddings to evaluate downstream task performance and similarity.
1) Downstram task: binary COPD classification and LAA950 (emphysema) regression
2) Linear reconstruction of embeddings
3) Cosine similarity, Top-k Jaccard, Linear CKA

- Results for this analysis are saved in: embeddings_test/embedding_comparison_results/
    - comparions_results.json
    - pairwise_similarity_correlation.png