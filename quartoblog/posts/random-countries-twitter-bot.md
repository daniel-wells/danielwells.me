---
title: "Random countries Twitter bot"
date: 2017-09-16
draft: false
---

Being able to recognise flags and locate countries on a map is the kind of general knowledge I that feel I ought to have.
This project was a kind of productive procrastination towards that goal.
There are [quizzes on Sporcle](https://www.sporcle.com/games/g/worldflags) I could have practiced with but somehow that seemed too brute force and not technically over complicated enough!
Besides, I wanted an excuse to learn about serverless technologies and Twitter bots.

So the idea was that I could make a Twitter bot which posts flags and maps of random countries along with population and capital city facts.
That way I could learn this knowledge while I'm procrasta-scrolling on Twitter.
Here's the result: [@randomcountries](https://www.twitter.com/randomcountries)

The bot is built on the python library [Tweepy](http://www.tweepy.org).
The data is from [Wikidata](https://www.wikidata.org) accessed via their SPARQL query service, and the images of flags and maps are from [Wikimedia Commons](https://commons.wikimedia.org) using the MediaWiki API.
AWS Lambda is used to host the bot which is scheduled to run every 1301 minutes resulting in 404 tweets per year (~2 per country).

To create the package for AWS lambda the Tweepy library dependency is installed in the project directory so it can be uploaded with the rest of the code:
```bash
pip install tweepy -t .
```

To fit in the 50MB AWS lambda limit the images are compressed with OptiPNG. The whole directory is then zipped before uploading.

You can see the code at [the Github repo](https://github.com/daniel-wells/countries-twitterbot).
