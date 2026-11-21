# SingaRoar Pipeline Notebooks

Run these notebooks in order to reproduce the full data processing, text mining, clustering, and retrieval pipeline from raw Reddit posts to dashboard-ready artifacts.

| Stage | Notebook | Summary |
|---|---|---|
| 0 | [Stage_0_Introduction.ipynb](Stage_0_Introduction.ipynb) | Project overview, dataset motivation, and pipeline roadmap |
| 1 | [Stage_1_Data_Collection_and_Data_Cleaning.ipynb](Stage_1_Data_Collection_and_Data_Cleaning.ipynb) | Raw Reddit cleaning (URLs, bots, duplicates, deleted posts, emojis) → `PostVault.csv`, `CommentVault.csv` |
| 2 | [Stage_2_POS_and_NER_Tagging.ipynb](Stage_2_POS_and_NER_Tagging.ipynb) | Part-of-Speech and Named Entity Recognition tagging |
| 3 | [Stage_3_Singlish_Normalisation.ipynb](Stage_3_Singlish_Normalisation.ipynb) | Lexical normalization of Singlish expressions |
| 4 | [Stage_4_Singlish_to_English_Conversion.ipynb](Stage_4_Singlish_to_English_Conversion.ipynb) | Dictionary-based Singlish-to-English conversion and `singlish_count` tracking |
| 5 | [Stage_5_Common_Normalisation.ipynb](Stage_5_Common_Normalisation.ipynb) | Slang expansion, spelling normalization, stopword removal, and lemmatization |
| 6 | [Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb](Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb) | TF-IDF, BM25, Sentence-BERT representations and inverted index construction |
| 8 | [Stage_8_Clustering_and_Visualization.ipynb](Stage_8_Clustering_and_Visualization.ipynb) | SVD dimensionality reduction, Silhouette-guided K-means clustering, t-SNE & word clouds |
| 9 | [Stage_9_Document_Search.ipynb](Stage_9_Document_Search.ipynb) | Document retrieval and heuristic ranking (title weight, recency, upvotes, comments) |
| Appendix | [Appendix_1_Sentiment_Labelling_Benchmark.ipynb](Appendix_1_Sentiment_Labelling_Benchmark.ipynb) | Benchmark evaluation of 6 sentiment/emotion labelling models |

> **Note:** Stage 7 (emotion inference) is implemented as a standalone script at [`../emotion_inference.py`](../emotion_inference.py), with summary plotting scripts in [`../sentiment_plots/`](../sentiment_plots/).
