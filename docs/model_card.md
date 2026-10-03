## Model Overview

| **Model name**          | Capstone BBO Bayesian Optimisation Model |
| ----------------------- | ---------------------------------------- |
| **Version**             | V1.0                                     |
| **Developer(s)**        | Benjamin Dowman                          |
| **Contact information** | ben.dowman@gmail.com                     |
| **License**             | N/A                                      |

## Intended Use

| **Primary task**          | Bayesian optimisation of Imperial Executive Education's 'Professional Certificate in Machine Learning and Artificial Intelligence' capstone project functions |
| ------------------------- | ------------------------------------------------------------ |
| **Target users**          | Developers, course markers                                   |
| **Recommended use cases** | Bayesian optimisation query suggestion of low-dimension black box problems, with a limited number of queries |
| **Not recommended for**   | High-dimensional black box optimisation, discontinuous black box function optimisation |

## Training Data

| **Data sources**            | https://github.com/bpdowman/ml_bbo_capstone |
| --------------------------- | ------------------------------------------- |
| **Size of dataset**         | 19-49 items per function (247 items total)  |
| **Languages or modalities** | Python (Scikit Learn, Scipy)                |
| **Preprocessing steps**     | Power scaling                               |

## Evaluation Metrics

| **Metrics used**            | Best observed result                                         |
| --------------------------- | ------------------------------------------------------------ |
| **Performance results**     | Current best results are outlined in the below table         |
| **Fairness or bias checks** | Distances between points, PCA both used to try and determine quality of exploration/exploitation |

| Function number | Best result |
| --------------- | ----------- |
| 1               | 0.41        |
| 2               | 0.63        |
| 3               | -0.0012     |
| 4               | 0.07        |
| 5               | 8662.41     |
| 6               | -0.64       |
| 7               | 1.90        |
| 8               | 9.95        |

## Ethical Considerations

| **Potential biases or risks** | Sparsity of function space leading to ill informed queries.<br />Lack of documentation leading to poor reproducibility of results |
| ----------------------------- | ------------------------------------------------------------ |
| **Mitigation strategies**     | Lengthy exploration phase of functions helps find multiple regions of interest.<br />Ensure each week's decision is properly documented |
| **Privacy concerns**          | N/A                                                          |

## Model Life Cycle

| **Date of last update**           |                                             |
| --------------------------------- | ------------------------------------------- |
| **Version control or repository** | https://github.com/bpdowman/ml_bbo_capstone |
| **Monitoring plan**               |                                             |

