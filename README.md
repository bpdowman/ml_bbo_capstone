# Section 1: project overview

The BBO (black box optimisation) capstone project revolves around 8 different functions. These functions have 2 to 8 coordinate dimensions, and a wide range of landscapes. This project acts as a proxy for an actual BBO problem - these may be seen in the workplace/real life in difficult-to-evaluate problems, such as chemical experiments that take days or weeks to evaluate, or optimisation of complex machine learning systems with long training times.

The overall goal of this project is to introduce us to how a BBO problem can be methodically solved, and explore many different methods to do this. The processing time constraint and limited number of points seen simulates a true BBO problem, where we work with very little data present and long evaluation times for new points - forcing us to use every tool at our disposal to try and find the landscape's maxima.

For my career, this is well suited to how engineering problems may have to be solved in the real world - this is well-linked to my dissertation, where a material's response to different wavelengths of light is tied to its geometric properties. Evaluating these responses can be extremely costly, potentially taking hours or days for the most complex 3D anisotropic materials.

# Section 2: inputs and outputs

Each function is of a set dimension. Functions 1 and 2 are of 2 dimensions, function 3 is 3 dimensional, functions 4 and 5 are 4 dimensional, function 6 is 5 dimensional, function 7 is 6 dimensional, and function 8 is 8 dimensional. A set of input coordinates is provided, with an associated output value for each - there are 10 to 40 points given for each function.

The true functions are unknown, and are queried with a set of 6-d.p. coordinates of the format x1-x2-...-xn. After some days of processing, a scalar output is produced for each function - representing the true value of the function at that point. Only one set of queries can be processed at any given time.

# Section 3: challenge objectives

I am aiming to find the maxima of each function within a limited amount of queries - 13 in total, across 13 weeks. A brief outline of function behaviour is available - e.g. function 4 having a lot of variation and many local minima, though actual behaviour of the function landscape cannot be seen, and is difficult to visualise due to the high-dimensional nature of many of the functions. 

There are several key limitations - the multi-day response delay between queries, the limit of 13 queries in total, and the sparsity of the high-dimensional functions. These make it challenging to map the landscape, and a significant amount of rigor goes into making a well-informed decision each week given the limited amount of information.

# Section 4: technical approach

Each function is modelled using a Gaussian process surrogate, with some manual intervention each week to better improve models based on various analyses performed.

During week 1, there was a strong focus on exploration. A basic model for each function was built using Gaussian processes, and initial points were queried. Basic visualisations were performed for the lower-dimensional functions, and the models developed based on these were extended to the higher-dimensional functions.

For week 2, two analyses were introduced - PCA (principal component analysis), and DA (distance analysis). PCA is a dimensionality-reduction techique in which the data is transformed to a lower dimension by identifying uncorrelated direction vectors to remove redundant information. DA maps relative positions of each point relative to each other point, and is used to visualise relationships between points and maxima, and new queried points.

For week 3, t-values for each dimension were calculated to try and determine significance of each dimension. Previously, each dimension for each function was assumed to be equally important - this has proved to be an invalid assumption based on the t-values, and as such different dimensions are weighted differently. It remains to be seen if these updated values hold. Normalisation of output values was also introduced to allow for healthier surrogate models.

What makes my approach unique is that I'm unafraid to follow an exploration-heavy strategy. Whilst many others have begun exploiting functions quickly, I am focusing on mapping each function space better such that I can tune my algorithms to very quickly converge onto what I hope is a true maxima. Many forms of analysis are going into this process, and it's interesting sorting through the data to figure out what's working and what's not.

# Section 5: dynamic log
This section will cover musings and experiments done for each week

### Week 5:
This week has involved the use of kernel-based PCA to try and query points. Unfortunately, PCA appears to lose a significant amount of data during our reconstructions, so this week will serve as a likely dud week. It is an exciting approach however! This method is being attempted for all but functions 1, 2, and 5 - as 5 is already approaching a peak seemingly, and functions 1 and 2 do not need dimensionality reduction for interpretation.