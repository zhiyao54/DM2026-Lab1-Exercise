# Your report

1. Data
result_id = load_dataset_1: The dataset has 847 Reddit posts with three sentiment labels: neutral (423), negative (315) and positive (109). We can see from the number mix that these classes are quite imbalanced. Positive posts are only about 13% of the data, so any result for the positive class rests on a small number of posts.

result_id = describe_data_2: I measured post length in characters. Overall, the mean is about 147 characters and the median is 78, and the longest post has 2,737 characters. The mean is higher than the median in every class, so the lengths are right-skewed suggesting that most posts are short, and a few long ones are pulling up the average. The box plot outputed also shows the same thing with the boxes sitting very low and all the long posts appearing as outliers. 

2. Document-term matrix
result_id = build_dtm_3: The document-term matrix has 847 rows and 4,346 columns, with 17,193 non-zero entries out of 3,681,062 cells, so it is 99.53% sparse. This works out to only about 20 distinct terms per post.  This might be because the reddit posts are usually quite short so each one uses only a tiny part of the entire vocabulary.

result_id = variance_filter_4: With a threshold of 0.01, the filter kept 506 terms and removed 3,840, so only about 12% of the vocabulary survived. This is likely becuase words show up in only a few posts, so their counts barely vary across 847 documents and their variance falls below 0.01. The sample of kept terms is almost all common words: "the", "to", "and", "is", "you", "of", "in", "it", "that", "for", with "market" as the only clearly topic related word in that sample. 

3. Correlation
result_id = pearson_filter_5: I compared each term against a one-vs-rest label for the positive class . The top terms were: huge, tldr, main, nice, technology, potential, perfectly, god, points, major, changing, exciting, wont, crush, redditors, stimulus, ice, 3d, vix and dividends. This is a mix of positive-sounding words and finance or Reddit words. The bar chart shows that the correlation is weak with the highest about 0.155 and the 15th highest is about 0.127. This suggests that the top terms are weak predictors of sentiment.

4. Dimension reduction and patterns
result_id = reduce_dimensions_6: In the PCA scatter plot, the first component explains 27.0% of the variance and the second 5.0%, so the 2D view captures only about 32% and loses the rest. The plot shows almost all points in one dense area near the origin, with the three sentiment classes overlapping. From this we can see that  PCA does not separate negative, neutral and positive posts.

result_id = patterns_7: For the positive class, I mined patterns with TF-IDF filtering and FP-Growth, using a minimum support of 11, which is about 10% of the 109 positive posts. Only two patterns were found, both single words: "good" and "just" and there were no multi-word patterns. Given that posts are usually short, it is rare for the same group of words to appear together in at least 11 of 109 posts. And "good" and "just" are generic words, so they likely occur in all classes and don't describe positive posts specifically. 
