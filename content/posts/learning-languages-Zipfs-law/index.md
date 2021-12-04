---
title: "Learning Languages with Zipf's Law"
date: 2017-04-05
draft: false
---

Learning a new language can be a daunting task. However, the word frequencies in many languages follow [Zipf's law](https://en.wikipedia.org/wiki/Zipf's_law) in that the most frequent word occurs twice as often as the second most frequent, and three times as often as the third, and so on. This means that a relatively small number of words make up the majority of the spoken and written corpus. So you only need to learn 500 or so words to understand ~75% of the words in common speech.

Here I use subtitles in English, German, French, Spanish, and Russian to explore this. The subtitles are from [opensubtitles.org](https://www.opensubtitles.org), provided by [Hermit Dave](https://github.com/hermitdave/FrequencyWords) as ranked lists of words with their word count. Code for this analysis can be found in the [.Rmd file](https://github.com/daniel-wells/learning-languages/blob/master/word_frequency.Rmd). This project is an expansion of work by [Tomi Mester](https://hackernoon.com/learning-languages-very-quickly-with-the-help-of-some-very-basic-data-science-cdbf95288333).

By plotting the cumulative frequency for the top N words we can see that you would only have to learn the 500 most frequent words to understand ~75% of all words, 1,000 for ~80%, and 2,000 gets you to ~85% (depending on the language).

<img src="word_frequency_files/figure-markdown_github/cumulative_percentage-1.png" width="960" />

We can test if a discrete power law (Zipf) fits the data well (for English). The red line shows the fitted power law with *α*= 1.6

<img src="word_frequency_files/figure-markdown_github/power_law-1.png" width="960" />

And what text analysis would be complete without a word cloud, here of the top 1,000 most frequent words. <img src="word_frequency_files/figure-markdown_github/wordcloud-1.png" width="960" /><img src="word_frequency_files/figure-markdown_github/wordcloud-2.png" width="960" /><img src="word_frequency_files/figure-markdown_github/wordcloud-3.png" width="960" /><img src="word_frequency_files/figure-markdown_github/wordcloud-4.png" width="960" /><img src="word_frequency_files/figure-markdown_github/wordcloud-5.png" width="960" />

### Caveats

-   Of course, even if you know 75% of all words that doesn't mean you will be able to understand 75% of sentences. However you may be able to glean some information from the context and I think these kind of lists could be a good place to kick start the learning process.
-   I have not checked the pre-processing steps (converting subtitle files to frequency lists) or performed any quality control. I can also not speak any of these languages apart from English and a small amount of German so it's hard for me to judge their quality!
