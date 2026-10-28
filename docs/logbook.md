# Project logbook

This document provides a summary of changes made each week.

#### Week 1:

Initial Gaussian processes method implemented, using Scikit Learn implementations. UCB used for each function, with a function lengthscale set as according to the formula below. Each GP uses the RBF kernel. All functions have a set resolution - a number of evenly spaced points on each axis.
$$
lengthscale = 1/{N^D}
$$
Where N is the number of points, D is the dimension.

All functions are focused on exploration currently.

#### Week 2:

Principal Component Analysis (PCA) was included to help visualise high-dimensional functions, implemented via Scikit Learn. Additionally, a visualisation representing the distance of each point from the mean of all points, and the maximum distance of that point from any other point is included. These help give a sense of the structure of the search space, and where each new point is being placed. Clustering is also performed with Sklearn's DBSCAN implementation.

Functions 4 and 5 are switched to exploitation using PI (probability of improvement) due to the identification of potential strong maxima.

#### Week 3:

T-values are calculated for each dimension to determine their effect on the output; these are then used to inform length scales for each dimension, allowing dimensions with little effect on the output to be smoothed and contribute less to queries. ARD was not chosen to be used, as some functions such as functions 1 and 2 could be clearly seen to have degenerate function representations when ARD was used. This is likely due to a lack of data available for ARD, due to function sparsity.

Standard scaling is implemented for the GPs, and the Matern kernel is used for function 1 due to the spiky nature of it being poorly modelled by a RBF kernel.

#### Week 4:

Scree plot analysis has been implemented to determine the variance explained by each PCA component. This is generally well-explained for lower dimensional functions (>85% for function 3), however less than half the variance is explained for the highest dimensional functions, with function 8 achieving about a 30% variance explained by the greatest components here.

Dimension scales were further refined for each function.

#### Week 5:

This week has involved the use of kernel-based PCA to try and query points. Unfortunately, PCA appears to lose a significant amount of data during our reconstructions, so this week will serve as a likely dud week. This method is being attempted for all but functions 1, 2, and 5 - as 5 is already approaching a peak seemingly, and functions 1 and 2 do not need dimensionality reduction for interpretation.

#### Week 6:

Exploitation moved to for each function, after several weeks of exploration. Functions 7 and 8 are also being moved to a Matern kernel.

The previously trialled method of using PCA to choose points worked somewhat well; with all points chosen falling within the top 10 points of each function. No new maxima were found, however.

#### Week 7:

Power scaling implemented (informed by HEBO paper) to reduce heteroscedasticity in data. All functions have also now been moved to a Matern kernel, to allow more flexibility than RBF kernels. 

Additionally, an automated hyperparameter tuning method was implemented to choose ideal noise scales, Matern kernel nu values, and a global scaling of lengthscales (global smoothness of functions). 

Finally, random number generation for high-dimensional grids was replaced with Sobol sequence generation. This uses a method of random-number generation that ensures all points are evenly spaced - ensuring that generated random points are more evenly spaced across the range.

#### Week 8:

In the spirit of this week's modules being about large language models, Claude Sonnet 5 and Claude Opus 5 were used to suggest queries. The same prompt was supplied, using prompt engineering methods to guide the output and reported values - one-shot prompting, system prompting, and context prompting were all combined to improve the quality of outputs/analyses used, and limit hallucinations. 

Additionally, the final output decision was constrained to not use any Python scripts; as such, the models were forced to 'reason' using information gathered from analyses. 

#### Week 9:

The LLM prompts for last week successfully identified new maxima for functions 2, 4, 6, and 8. Functions 3 and 7 also found strong new maxima. The strength of these responses for so many functions is above a matter of luck, and as such it can be determined that the LLM executed an exploitative strategy well. 

Expected improvement is now being used for all functions; this is in response to its popularity in literature, as well as literature showing that EI will always swiftly converge to the system maxima given well-chosen system hyperparameters. This allows me to focus on tuning other hyperparameters, and stops the continuous tweaking of exploration vs exploitation by opaque (for hard-to-visualise higher dimensional functions) hyperparameters for the acquisition functions.

#### Week 10:

Due to the nature of the high-dimensional functions, at even 10 million points generated, this only yields a resolution of ~0.1 for each. As such, this week implements a basic trust-region inspired system for improving the effective resolution for these high-dimensional functions. The region of highest-performing points (based on the GP predicted values) is is used to constrain the maximum and minimum values for subsequent rounds of points. As such, we can effectively increase the granularity for guesses in higher-dimensional space.

This method, however, does have a trade-off in that it may shift attention from promising regions if the granularity of the initial Sobol-sequence guesses misses out on the peaks. It is unknown how this issue can be solved for any guess region, due to the nature of high-dimensional guesses for GPs - even augmenting with previously-best-known points will run into this issue with biasing towards existing well-known points (a.k.a unintentionally biasing the GP towards exploitation for strong maxima).

#### Week 11:

Kernel noise switched from noise variance being defined by alpha to noise being defined by a white kernel of the same variance.

Trust region implementation updated to contract the region of interest more reliably.

Hyperparameter tuning algorithm updated to use NLPD (negative log predictive density) instead of just RMSE between the mean and predicted values. This correctly includes confidence, such that confident incorrect and unconfident correct guesses are penalised correctly.

#### Week 12: 

No changes - exploitation focused

### Week 13:

No changes - exploitation focused



paper:

https://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf



#### PAPERS REFERENCED:

A key piece of work used is *'Pattern Recognition and Machine learning'* *- C. Bishop, 2006*. This book has provided technical backgrounds and justifications to many of the tools used, such as Principal Component Analysis (12.1), and the choice of Gaussian Processes as surrogates with RBF kernels (6.4). 

*'Taking the Human Out of the Loop: A Review of Bayesian Optimization' - B. Shahriari et al., 2016* is, whilst a research/review paper, a highly informative source of information and strategies used around bayesian optimisation. It also shows some interesting conclusions - EI is shown to have significantly better potential convergence than PI for some cases, though *'Convergence rates of efficient global optimization algorithms' - A. Bull (2011)* suggests that a poorly-tuned EI acquisition function can fail to converge at all. This paper is the main focus of the coming week's queries.

*'Practical Bayesian Optimization of Machine Learning Algorithms' - J. Snoek et al., 2012* justifies the choice of Gaussian process surrogates, and suggests all functions using the Matern 2.5 kernel. Whilst much of the paper focuses on an altered form of EI that is not applicable here, it again provides evidence that EI may be a more suitable acquisition function.

*'HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation' - A. Cowen-Rivers et al., 2020* discusses Huawei's approach to the NeurIPS 2020 black box optimisation challenge. The key takeaway implemented was the use of power scaling to reduce heteroscedasticity.

*'Team Optuna Developers’ Method for
Black-Box Optimization Challenge 2020 - Shibata et al., 2020'* extends *'Scalable Global Optimization via Local Bayesian Optimization' - D. Eriksson et al., 2019* and performs simple augmentations of the TuRBO algorithm for improved performance on bayesian optimisation problems. The growing/shrinking nature of the trust regions inspired my algorithm for improving the resolution of guesses, and the paper inspired my switch to Sobol sequences from random sampling for higher-dimensional guesses.