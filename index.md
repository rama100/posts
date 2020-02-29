---
title: "The Intuition Behind Ensemble Learning, Bagging and Random Forests"
author: "Prof. Rama Ramakrishnan"
date: "February 27, 2020"
output:
  html_notebook: default
  html_document:
    df_print: paged
---


```{r, include=FALSE}

library(ggplot2)
library(dplyr)

```

    
### Ensemble Learning

In this brief note, I want to provide some intuition behind the idea of **Ensemble Learning** and walk through the key ideas underlying the two ensembling techniques we have seen in class so far: **Bagging** and **Random Forests**. 

According to the dictionary, the word 'ensemble' means ...

>'a group of items viewed as a whole rather than individually'  

... and that certainly captures the spirit of ensemble learning.

The core idea behind ensemble learning is simple:

* train a number of **diverse** predictive models. The models can be anything - CART, regression, ... - more diverse the better.
* to make a prediction for a new data point, send the data point to each model and get back its prediction, and then simply **combine** the predictions:
    + for regression problems, calculate the average of all the predictions
    + for classification problems, pick the most frequently predicted class

This simple idea is extremely powerful and without it, you can't win Machine Learning competitions at sites like [Kaggle](www.kaggle.com). I'd go so far as to say that this may be the only *free lunch* in predictive analytics.

So how do we go about finding a **diverse** set of models?

Diversity can stem from many sources:

1. The *type* of model. This is probably the first thing you thought of when you thought of diversity. Is it a *tree*-based model like in CART, a *linear function* like in Linear Regression, a Generalized Additive Model (GAM) that uses those amazing *splines* we learned about, a Neural Network, ..., the possibilities are many. 'Ensembling' models of different types is a good way to increase diversity.
2. The *datasets* used to train the models. This is a bit counter-intuitive. We have *one* training dataset. How can we make it more 'diverse'? Turns out that there's a simple but clever way to do so.
3. The *predictors* used to train the models. This is also counter-intuitive. We have a set of predictors in the dataset. Shouldn't we just use them *all*? Where's the diversity here? Again, it turns out that there's a simple and very clever way to introduce diversity here. 

In the rest of this lecture, we will assume that we can only use one type of model -- CART trees -- and study how to use (2) and (3) to create ensembles of CART trees. Amazingly, even with this seemingly severe self-imposed restriction, we can build state-of-the-art predictive models if we are creative in (2) and (3). 

Before we look at specific ensembling algorithms, let's understand how and when ensembling helps. 


### How Ensembling Increases Predictive Ability

Imagine you have an ensemble of `r n = 1000; n` different classification models. Let's assume that they are mediocre models - in fact, on average, they predict the correct class only `r p = 0.51; 100*p`% of the time i.e., barely better than a coin toss. But they have one thing going for them: they are **independent** of each other. To predict the class of a new datapoint with this ensemble, you run it through the 1000 models and do a *majority vote* at the end: If `r ceiling(n/2)` or more of them predict class A, you predict A. Otherwise, you predict B (if the votes are *exactly* equal, we will go with A)

What's your ensemble's accuracy i.e., what % of the time will it predict the right answer? 

To answer this, we can use our old friends, the *binomial distribution* and its R companion, the `pbinom()` function. You will recall that, given $n$ 'trials' each of which **independently** results in success with probability $p$ and failure with probability $1-p$, the probability of at most $k$ successes is given by `pbinom(k,n,p)`. In our example, $p=0.51$, $n=1000$ and $k=500$ and we want the probability of 500 or more successes. That is `1-pbinom(499,1000,0.51)`, which is `r 100 * round(1-pbinom(499,1000,0.51),2)`%!

By simply taking the majority vote from 1000 very weak models, we went from `r 100*p`% to `r 100 * round(1-pbinom(499,1000,0.51),2)`%!

Does this mean that if the accuracy of our individual model is higher than `r 100*p`%, the ensemble can do even better? Let's find out.

```{r}
indiv.prob = seq(0.5,1.0,0.01)
ensemble.prob = sapply(indiv.prob, function(x) 1-pbinom(ceiling(n/2) - 1, n, x))
acc = data.frame(indiv.prob, ensemble.prob)
ggplot(data = acc) +
  geom_point(aes(x=indiv.prob, y=ensemble.prob)) +
  geom_line(aes(x=indiv.prob, y=indiv.prob), color="red") +
  xlab("Accuracy of Individual Model") +
  ylab("Accuracy of Ensemble")
```

We can see that the accuracy of the ensemble climbs *very rapidly* as the accuracy of the individual model increases above 50%. Let's zoom into the left side of the chart and take a closer look.


```{r}
indiv.prob = seq(0.5,0.55,0.005)
ensemble.prob = sapply(indiv.prob, function(x) 1-pbinom(ceiling(n/2) - 1, n, x))
acc = data.frame(indiv.prob, ensemble.prob)
ggplot(data = acc) +
  geom_point(aes(x=indiv.prob, y=ensemble.prob)) +
  geom_hline(yintercept = 1.0, color='red', linetype = 'dotted', size = 0.5) +
  geom_vline(xintercept = 0.55, color='red', linetype = 'dotted', size = 0.5) +
  xlab("Accuracy of Individual Model") +
  ylab("Accuracy of Ensemble")
```

By the time the accuracy of the individual model hits (a still unimpressive) 55%, the ensemble is basically making perfect predictions! That's the power of ensembles!!

Does this mean that if we can build an individual model with accuracy of atleast 55%, we can make perfect predictions by ensembling a 1000 of them?

Only if the 1000 models are **independent**. In the analysis above, we could use the binomial distribution (and `pbinom()`) because we assumed that the 1000 models were **independent**. When they are not, the accuracy of an ensemble won't be as good - it will be somewhere between the accuracy of the individual model and a fully independent ensemble. 


```{r}
indiv.prob = seq(0.5, 1.0, 0.005)
ensemble.prob = sapply(indiv.prob, function(x) 1-pbinom(ceiling(n/2) - 1, n, x))
acc = data.frame(indiv.prob, ensemble.prob)
ggplot(data = acc, aes(x=indiv.prob, y=ensemble.prob)) +
  geom_line(aes(x=indiv.prob, y=indiv.prob), color="red") +
  geom_ribbon(aes(ymin=indiv.prob, ymax=ensemble.prob), fill="grey80") +
  geom_line(color="blue", size=0.5) +
  annotate(geom="text", x=0.7, y=0.9, label="Ensembles live here!") +
  xlab("Accuracy of Individual Model") +
  ylab("Accuracy of Ensemble") 
```


If your ensemble of models is very independent, the ensemble's accuracy will be close to the *blue line*. At the other extreme, if they are practically clones of each other, the ensemble's accuracy will be close to the *red line*. 

**Ensembling techniques like Bagging and Random Forests are somewhere in the gray zone between the red and the blue lines and are the result of researchers' efforts to get closer to the blue line.**


### Bagging

Now that we understand how and when ensembling helps, let's learn perhaps the simplest way to leave the red line and journey to the blue line: **B**ootstrap **ag**gregation a.k.a. Bagging

The Bagging algorithm is very simple.

> **The Bagging Algorithm**
>
>Let's assume that the original training dataset has $n$ observations.
>
>Repeat $N$ times:
>
>1. From the training dataset, **randomly choose** $n$ observations
>2. Build a CART tree. 

When the algorithm finishes, we will have built $N$ trees that are ready for action. To make a prediction for a new datapoint, we simply get $N$ predictions, one from each of the $N$ trees we built. Then we average the predictions if this is a regression problem, or pick the most frequently predicted class if this is a classification problem.

At this point, you may be asking yourself: What does it mean to *randomly choose* $n$ observations in step (1) given that the entire training set has only $n$ observations in the first place?

Great question! The answer is a simple but very clever idea called the [**bootstrap**](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).

This is how step (1) works: Imagine that we have a giant bin with all $n$ observations that make up the original training set. We reach inside the bin and randomly pick an observation up. We make a copy of this observation and add it to the new training set we are in the process of creating. 

Now comes the important part: before we reach inside the bin again, **we put the first observation back in the bin**. We then reach inside and randomly pick an observation and add a copy of it to the training set. This is our second observation. We then put this observation back in the bin. And so on. We do this $n$ times and when we are done, our new training set will have $n$ observations. 

The technical name for "putting each observation back in the bin before picking the next one" is *sampling with replacement*. This creates diversity in two ways:

- We make it possible for some observations in the original dataset to **show up more than once** in this training dataset.
- We make it possible for some observations in the original dataset to **not show up at all** in this training dataset.
 
This is how the new training set we create in step (1) will differ from the original training dataset. 

Over $N$ iterations of the algorithm, $N$ *different* training datasets will be created in this manner and therefore $N$ *different* trees will be built. That's how we create an ensemble of trees from a single, original dataset.

Bagging is very simple to implement. If you can build a CART tree, it is just a few more lines of code.

How well does it work? Better than a single tree for sure but we don't make much progress towards 'the blue line'.

The reason for this modest progress is that the $N$ trees it builds aren't *diverse* enough. They tend to be quite similar because there's considerable *overlap* in the $N$ training datasets we created. In fact, for any two datasets, we can show that about `r round(100*0.63*0.63,0)`% of the observations will be the same. When the trees are similar, the predictions tend to be similar and we don't get the ensembling effect.

To make *further* progress towards 'the blue line', we need to *further* increase the diversity of the trees we generate. And that brings us to Random Forests.


### Random Forests

The  Random Forest algorithm is very similar to the Bagging Algorithm but with a change in Step (2).

>**The Random Forest Algorithm**
>
>Let's assume that the original training dataset has $n$ observations.
>
>Repeat $N$ times:
>
>1. From the training dataset, randomly choose $n$ observations
>2. Build a CART tree **but when considering which predictor to split on, don't choose from all predictors. Instead, choose from a random sample of the predictors. **

That's it.  

The all-important tweak in step (2) is where the magic happens.

Recall what happens when a CART tree is built. At every iteration, we consider each predictor in turn, and calculate the reduction in impurity if we split on a particular value of that predictor. Across all these predictor/split-point options, we pick the one that results in the most reduction in impurity.

Instead of doing this, in Random Forests, we first **randomly** choose a subset of predictors and pick the best split from *within* this subset of predictors. If we had `r k=1000; k` predictors, we may consider only `r floor(sqrt(k))` of them for any split. Note that this random choosing of predictors is done *each and every time* we have to decide on a split so the subset of predictors we consider for splitting keeps changing throughout.

At first glance, this sounds like a strange, even bad, idea. We have repeatedly learned that more data is better and it seems sensible to use as much 'raw material' as possible to build our models and not handcuff the process in any way. So why does this work?

(Aside: To quickly get insight into how algorithms work, a useful trick is to mentally run *extreme* cases through the algorithm)

Imagine a dataset where one predictor is *massively predictive* compared to the rest (extreme case). This predictor will likely be chosen as the first splitting variable in *every one* of the $N$ CART trees built by Bagging. And it will probably get picked for splitting again and again inside those $N$ trees. As a result, the $N$ trees will be very similar to each other and the ensembling effect will be muted.

Imagine, instead, that other predictors also were given a chance to be considered for splitting. The trees that get built will then be different from each other and we may get an ensembling benefit. 

What's the simplest way to ensure that every predictor gets a chance to be considered for splitting? *Randomly* choose a subset of predictors at every split. This levels the playing field.

Note that we are **not forcing** certain predictors to be used for splitting. We are only ensuring that all predictors have an equal chance to be **considered** for splitting. The actual split will be determined as before by which combination of predictor and split point reduces impurity the most.

If a very strong predictor gets randomly chosen, it will likely dominate the weaker predictors and be picked for splitting. But there's a good chance it won't get chosen at all and when that happens, the other predictors can shine.

All this said, we have to be careful that we don't handcuff the algoritm too much. If we randomly choose predictor subsets that are **too small**, predictive ability will start to suffer because the strong predictors won't get considered for splitting often enough to have an impact.

If you are beginning to feel that there's an underfitting-overfitting story developing here, you are on the right track!


To summarize:

* Randomly choosing a subset of predictors for consideration at every split can lead to trees that are different from each other
* If we choose **too few** predictors, it may hurt the ensembling effect since strong predictors don't get the chance they deserve and therefore the individual trees that get built have poor performance (in the chart above with the red/blue lines and the gray zone in between, it is as if you started out at a point on the red line and traveled *left* rather than *up*)
* If we choose **too many** predictors, it may hurt the ensembling effect since the trees will be too similar

The solution to this dilemma is easy: 

* Treat the "# of predictors to be chosen" as a *hyperparameter* (like `cp` or `minbucket` in CART). In the `RandomForest` package, this is the `mtry` parameter.
* Use a test set or cross-validation to try several different values for `mtry` and pick the best one.

If you don't have time, a good default value for `mtry` is $\sqrt{k}$ for classification problems and $\frac{k}{3}$ for regression problems, where $k$ is the total number of predictors.



































