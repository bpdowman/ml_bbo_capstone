# Dataset Datasheet

## Motivation

This dataset has been created to explore avenues to improve performance of a Bayesian black box optimiser, focused on using Gaussian processes to approximate low-dimensional functions. The dataset catalogues data for 8 different functions of varying dimensionality and characteristics. 

The data is a collection of initially provided data, provided by Imperial Executive Education, collated with data points obtained by successively querying the black box functions of interest. The creation of the dataset was further supported through training given by Imperial Executive Education.

## Composition

The dataset is made up of eight portions, describing the below functions. Function data is incomplete as of the time of writing, with four more weeks of querying planned to expand the dataset. 

Each entry in the dataset consists of a real float vector input, and a corresponding real float scalar output, creating a mapping of coordinates for each functions space. Input values range between 0 and 1, with up to 6 decimal places of precision for each value. Output values have no defined range or number of decimal places, with the scales of values depending on the function, and may be returned in scientific notation of the format `<number>e<power>`.

Information about each function is given below, with the number of entries including both the initial provided data and the queried data. 

| Function number | Input dimensions | Number of entries | Planned final number of entries |
| --------------- | ---------------- | ----------------- | ------------------------------- |
| 1               | 2                | 19                | 23                              |
| 2               | 2                | 19                | 23                              |
| 3               | 3                | 24                | 28                              |
| 4               | 4                | 39                | 43                              |
| 5               | 4                | 29                | 33                              |
| 6               | 5                | 29                | 33                              |
| 7               | 6                | 39                | 43                              |
| 8               | 8                | 49                | 53                              |

Data is provided as a collection of binary (`.npy`) files for the initially provided inputs and outputs, and a Python (`.py`) file containing queried inputs and outputs in the form of Python arrays. The latter may be imported into a project through `from data.bbo_data_dict import new_inputs, new_outputs`.

Due to the nature of the points as queried function data, the composition is naturally imbalanced across the function space; additionally, higher dimensional functions are sparse by nature.

Due to the nature of the data being sourced from black box queries, there are no individuals or entities associated with any data entries, and as such no privacy or population bias considerations. 

## Data collection process

Datapoints obtained beyond the initially provided dataset were obtained through querying of black box functions. The framework for obtaining these points was written in Python, primarily utilising  the publicly available Scikit Learn and Scipy libraries. Gaussian processes were fit to all previously available points for each function, and acquisition functions were used to decide the most suitable new point. 

Functions were queried through Imperial Executive Education's online function querying portal, run by Emeritus. Submitted points took the form of coordinate vectors, with the format `x1-x2...-xn`, where each value has 6 decimal places of precision. Responses were returned by email, taking the form of real scalars with a variable precision and no defined range.

Data collection was performed over the course of thirteen weeks (four as of yet planned), with one query performed each week. 

## Pre-processing

Power transforms are applied to each function, to combat the effect of heteroscedasticity. Raw data is available on request.

## Uses

The intended use of this dataset is for refinement of low-dimension automatic Bayesian optimisation methods. This dataset is not suitable for generalisation to high-dimensional Bayesian optimisers, or for as the only source of refinement for an optimiser.

## Distribution

As the data present is the property of Imperial Executive Education, this dataset is only to be shared with third parties with express permission from this party, and the creator of this dataset. Distribution will be done through online repositories. Distribution to other parties without express prior permission is not permitted to those given access to the dataset.

The dataset is available, once permission is given, under the `initial_data` and `data` folders of the repository `https://github.com/bpdowman/ml_bbo_capstone`

## Maintenance

Any changes or addition to the dataset will be clearly discussed if performed, with previous versions of the dataset saved and available on request. Long term storage will be done through the repository on Github, with communication channels for any requests for updates or reports of errors available on the same platform.