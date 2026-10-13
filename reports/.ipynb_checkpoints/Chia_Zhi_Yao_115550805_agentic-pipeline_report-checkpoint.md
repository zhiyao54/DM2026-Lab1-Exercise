# Your report

1. Data
result_id = load_dataset_1: The dataset has 2,257 documents: soc.religion.christian (599), sci.med (594), comp.graphics (584) and alt.atheism (480). The classes are fairly balanced, which suggest that results are unlikely to be caused by one category being much bigger than the others.


result_id = describe_data_2: Overall, the mean is 1,992.96, the median is 1,235 and the maximum is 60,713. The mean is much higher than the median because a small number of very long posts pull the average up. the box plot that was generated also shows similar medians across categories, and they have long tails due to outliers.

2. Document-term matrix
result_id = build_dtm_3: The document term matrix has 2,257 rows  and 35,788 columnswith 365,886 non-zero entries and it is 99.55% sparse. 

result_id = variance_filter_4:  With a threshold of 0.01, the filter kept 5,354 terms and removed 30,434. Variance measures how much a term count changes between documents, so the filter removes terms that barely change, mostly very rare words. Some of the kept terms include "the", "of", "to" and "and" which are very generic and common. This is likely because variance on raw counts rewards frequent words.

3. Correlation
result_id = pearson_filter_5:  I compared each term against a one-vs-rest label for comp.graphics.. The following were listed as top terms: thanks, re, writes, keywords, who, looking, vga, video, people, that, card, hi. Only a few are actually relate to graphics like vga, video, card. Most of the rest are generic  words, this shows that a high linear correlation with the label does not guarantee a word is topical.

result_id = feat_corr_9: This is the correlation matrix of the 20 highest-variance terms, which are mostly stop words. The heatmap shows common words like the, of, to, and, is, in. which are strongly correlated with each other. This makes sense, because they all appear more in longer documents, so their counts rise and fall together. In contrast, "jpeg" and "edu" are only weakly correlated with the others, since they are quite specific and would likely depend on the topic.


4. Dimension reduction 
result_id = reduce_dimensions_6:  The first principal component explains 65.94% of the variance and the second explains 4.33%. In the scatter plot, almost all points sit in one dense area near the origin, where the four categories overlap. A few outliers stretch the axes far from this area. So PCA does not separate the categories in this setup.

result_id = patterns_7: For comp.graphics, I mined patterns with TF-IDF filtering, FP-Growth and a minimum support of 80 and found patterns like {code, lines, subject}, {email, organization} and {information, lines, subject}. words like "subject", "lines" and "organization" are likely email fields. So these patterns mostly reflect how the messages are formatted and not what the graphics discussions are about. 

5. Other 
result_id = cosine_sim_8: documents 0 and 1 have cosine similarity 0.3532. This is well below 1, so they share some terms but are not very similar.

result_id = dtm_heatmap_10: The heatmap of the first 20 documents and first 20 terms is almost all zeros. Only documents 4 and 5 have a count for the term "00".