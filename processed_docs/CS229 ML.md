CS229 Lecture Notes

Andrew Ng and Tengyu Ma June 11, 2023

Contents

|            | Supervised learning                                                     | Supervised learning                                                     | Supervised learning                                                     |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------|------------------------------|-------------------------------|----------------------------------|-----------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| 1          | Linear regression                                                       | Linear regression                                                       |                                                                         |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|            | 1.1 . .                                                                 | LMS algorithm . . .                                                     | . . . . .                                                               | .                                                                  | .                            | .                             | . . .                            | . . .                       | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          | . . . . .                                                          |
|            | 1.2                                                                     | The normal equations . . .                                              | . . . . . . . . . . . 13                                                | .                                                                  | . . . . . . . . . . . 13     | .                             | . . . . .                        | . .                         | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           | . . . . . . . . . . . 13                                           |
|            |                                                                         | 1.2.1 Matrix derivatives .                                              | . . . . . . . . . . . . . 13                                            | .                                                                  | . . . . . . . . . . . . . 13 | .                             | . . .                            | . .                         | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       | . . . . . . . . . . . . . 13                                       |
|            |                                                                         | 1.2.2 Least squares revisited                                           | 14                                                                      | 14                                                                 | 14                           | 14                            | 14                               | 14                          | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 | 14                                                                 |
|            | 1.3                                                                     | Probabilistic interpretation                                            | . . . . 15                                                              | .                                                                  | .                            | .                             | . . . . . .                      | . . .                       | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         | . . . . 15                                                         |
|            | 1.4                                                                     | Locally weighted linear regression (optional reading)                   | 17                                                                      | 17                                                                 | 17                           | 17                            | 17                               | 17                          | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 | 17                                                                 |
|            | Classification and logistic regression                                  | Classification and logistic regression                                  | Classification and logistic regression                                  |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|            | 2.1                                                                     | Logistic regression . . .                                               | . . 20                                                                  | .                                                                  | .                            | .                             | . . . . . .                      | . . .                       | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             | . . 20                                                             |
|            | 2.2                                                                     | Digression: the perceptron learning algorithm                           | 24                                                                      | 24                                                                 | 24                           | 24                            | 24                               | 24                          | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 | 24                                                                 |
|            | 2.3 2.4                                                                 | Multi-class classification . . . .                                      | 27                                                                      | 27                                                                 | 27                           | 27                            | 27                               | 27                          | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 | 27                                                                 |
|            | Another algorithm for maximizing glyph[lscript] ( θ ) . . . . . . . . . | Another algorithm for maximizing glyph[lscript] ( θ ) . . . . . . . . . | Another algorithm for maximizing glyph[lscript] ( θ ) . . . . . . . . . |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
| linear     |                                                                         |                                                                         |                                                                         |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|            |                                                                         |                                                                         | .                                                                       | .                                                                  | .                            | .                             | .                                | .                           | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  |
|            | 3.1                                                                     | The exponential family .                                                | . . . . . . . . . . .                                                   | .                                                                  | .                            | . . . . . . . . . . .         | . . .                            | . .                         | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              | . . . . . . . . . . .                                              |
| 3.2        |                                                                         | Constructing GLMs . . . . 3.2.1 Ordinary least squares                  | . . . . . . . . . . . . . . .                                           | .                                                                  | . .                          | . . . . . . . . . . . . . . . | . . . . . .                      | . . .                       | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      | . . . . . . . . . . . . . . .                                      |
|            |                                                                         |                                                                         | . . . . . . . . . . 33                                                  | . . . . . . . . . . 33                                             | . . . . . . . . . . 33       | . . . . . . . . . . 33        | . . . . . . . . . . 33           | . . . . . . . . . . 33      | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             | . . . . . . . . . . 33                                             |
|            |                                                                         | Logistic regression                                                     | . .                                                                     | . .                                                                | . .                          | . .                           | . .                              | . .                         | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                | . .                                                                |
|            |                                                                         |                                                                         | 3.2.2                                                                   | 3.2.2                                                              | 3.2.2                        | 3.2.2                         | 3.2.2                            | 3.2.2                       | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              | 3.2.2                                                              |
| 4.1.1 The  | Generative discriminant multivariate                                    | 4.1.2 The Gaussian discriminant analysis                                | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . .      | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | . .                          | .                             | . . . . . . distribution . model | . . . .                     | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . | analysis normal 4.1.3 4.2 Naive bayes (Option Reading) . . . . . . |
| 4 learning | 4.1 Gaussian                                                            | algorithms Discussion: GDA and logistic regression                      |                                                                         |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|            |                                                                         | 4.2.1 Laplace                                                           |                                                                         |                                                                    |                              |                               |                                  |                             |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |
|            |                                                                         |                                                                         | 4.2.2 Event models for text                                             | 4.2.2 Event models for text                                        | 4.2.2 Event models for text  | 4.2.2 Event models for text   | 4.2.2 Event models for text      | 4.2.2 Event models for text | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        | 4.2.2 Event models for text                                        |
|            |                                                                         |                                                                         |                                                                         | classification .                                                   | classification .             | classification .              | classification .                 | classification .            | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   | classification .                                                   |
|            |                                                                         |                                                                         |                                                                         | .                                                                  | .                            | .                             | .                                | .                           | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  | .                                                                  |
|            | smoothing . . . . . .                                                   |                                                                         | . . .                                                                   |                                                                    | .                            | .                             |                                  | . . . .                     |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |                                                                    |

| CS229 Spring 20223   | CS229 Spring 20223                                                                                                                                                                                                                                                                                                                                                   |   CS229 Spring 20223 | CS229 Spring 20223                                                                      | 2                                                             |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------|
|                      | Kernel methods                                                                                                                                                                                                                                                                                                                                                       |                    5 |                                                                                         |                                                               |
|                      | 5.1 Feature maps .                                                                                                                                                                                                                                                                                                                                                   |                      | . . .                                                                                   | 48                                                            |
|                      | 5.2 LMS (least mean                                                                                                                                                                                                                                                                                                                                                  |                      | . . .                                                                                   | 49                                                            |
|                      | 5.3 LMS with the kernel                                                                                                                                                                                                                                                                                                                                              |                      | . . .                                                                                   | 49                                                            |
|                      | 5.4 Properties of kernels                                                                                                                                                                                                                                                                                                                                            |                      | . . .                                                                                   | 53                                                            |
|                      | Support vector machines                                                                                                                                                                                                                                                                                                                                              |                      |                                                                                         | 59                                                            |
|                      | 6.1 Margins: intuition                                                                                                                                                                                                                                                                                                                                               |                      | . . .                                                                                   | 59                                                            |
|                      | 6.2 Notation (option                                                                                                                                                                                                                                                                                                                                                 |                      | . . .                                                                                   | 61                                                            |
|                      | 6.3 Functional and geometric                                                                                                                                                                                                                                                                                                                                         |                      | . . .                                                                                   |                                                               |
|                      |                                                                                                                                                                                                                                                                                                                                                                      |                      |                                                                                         | 61                                                            |
|                      |                                                                                                                                                                                                                                                                                                                                                                      |                      |                                                                                         | 63                                                            |
|                      | 6.4 The optimal margin                                                                                                                                                                                                                                                                                                                                               |                      | . . .                                                                                   |                                                               |
|                      | 6.5 Lagrange duality                                                                                                                                                                                                                                                                                                                                                 |                      | . .                                                                                     | 65                                                            |
|                      | 6.6                                                                                                                                                                                                                                                                                                                                                                  |                      | .                                                                                       |                                                               |
|                      | Optimal margin                                                                                                                                                                                                                                                                                                                                                       |                      | . . reading)                                                                            | 68                                                            |
|                      | 6.8 The SMO algorithm 6.8.1 Coordinate 6.8.2 SMO . . Deep learning Deep learning 7.1 Supervised learning 7.2 Neural networks 7.3 Modules in Modern 7.4 Backpropagation 7.4.1 Preliminaries 7.4.2 General strategy 7.4.3 Backward 7.4.4 Back-propagation 7.5 Vectorization over III Generalization Generalization 8.1 Bias-variance tradeoff 8.1.1 A mathematical 8.3 |                      | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 73 74 75 79 80 80 84 92 98 99 102 105 107 109 112 113 115 120 |
|                      | 8.2 The double descent Sample complexity                                                                                                                                                                                                                                                                                                                             |                      | . . . .                                                                                 | 121 126                                                       |

8.3.1

Preliminaries

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 126

8.3.2

The case of finite

H

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 128

8.3.3

The case of infinite

H

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 131

9

Regularization and model selection

135

9.1

Regularization .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 135

9.2

Implicit regularization effect (optional reading) .

.

.

.

.

.

.

.

. 137

9.3

Model selection via cross validation

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 139

9.4

Bayesian statistics and regularization

.

.

.

.

.

.

.

.

.

.

.

.

.

. 142

IV

Unsupervised learning

144

10 Clustering and the

k

-means algorithm

145

11 EM algorithms

148

11.1

EM for mixture of Gaussians .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 148

11.2

Jensen's inequality

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 151

11.3

General EM algorithms .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 152

11.3.1

Other interpretation of ELBO

.

.

.

.

.

.

.

.

.

.

.

.

.

. 158

11.4

Mixture of Gaussians revisited .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 158

11.5

Variational

inference

and variational

auto-encoder (optional

reading)

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 160

12 Principal components analysis

165

13 Independent components analysis

171

13.1

ICA ambiguities .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 172

13.2

Densities and linear transformations .

.

.

.

.

.

.

.

.

.

.

.

.

.

. 173

13.3

ICA algorithm .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 174

14 Self-supervised learning and foundation models

177

14.1

Pretraining and adaptation .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 177

14.2

Pretraining methods in computer vision .

.

.

.

.

.

.

.

.

.

.

.

. 179

14.3

Pretrained large language models

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

. 181

14.3.1

Open up the blackbox of Transformers

.

.

.

.

.

.

.

.

. 183

14.3.2

Zero-shot learning and in-context learning

.

.

.

.

.

.

. 186

| Reinforcement Learning and Control 188                                                                                                                                                                                                                                                                                                              |          |         |                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------|-------------------------------|
| Reinforcement learning 189 15.1 Markov decision processes . . . . . . . . . . . . . . . . . . . . 190 15.2 Value iteration and policy iteration . . . . . . . . . . . . . . . 192 15.3 Learning a model for an MDP . . . . . . . . . . . . . . . . . . 15.4                                                                                         |          |         |                               |
| LQR, DDP and LQG 206 Finite-horizon MDPs . . . . . . . . . . . . . . . . . . . . . . . 206 Linear Quadratic Regulation (LQR) . . . . . . . . . . . . . . . 210 From non-linear dynamics to LQR . . . . . . . . . . . . . . . 213 16.3.1 Linearization of dynamics . . . . . . . . . . . . . . . . 214 16.3.2 Differential Dynamic Programming (DDP) |          |         | 16.3                          |
| (Optional) . . . . . .                                                                                                                                                                                                                                                                                                                              |          |         | 16.2                          |
| 194                                                                                                                                                                                                                                                                                                                                                 |          |         |                               |
| Continuous state MDPs . . . . . . . . . . . . . . . . . . . . . 196 15.4.1 Discretization . . . . . . . . . . . . . . . . . . . . . . . 196                                                                                                                                                                                                         |          |         |                               |
| approximation . . . . . . . . . . . . . . 199 Policy and Value Iteration . 203                                                                                                                                                                                                                                                                      | function | between | 15.4.2 Value 15.5 Connections |
|                                                                                                                                                                                                                                                                                                                                                     |          |         | 16.1                          |
| . .                                                                                                                                                                                                                                                                                                                                                 |          |         |                               |
| 214                                                                                                                                                                                                                                                                                                                                                 |          |         |                               |
| .                                                                                                                                                                                                                                                                                                                                                   |          |         |                               |
| 216                                                                                                                                                                                                                                                                                                                                                 |          |         |                               |
| 220                                                                                                                                                                                                                                                                                                                                                 |          |         |                               |
| .                                                                                                                                                                                                                                                                                                                                                   |          |         |                               |
| .                                                                                                                                                                                                                                                                                                                                                   |          |         |                               |
| . . . . . . .                                                                                                                                                                                                                                                                                                                                       |          |         |                               |
| . . . . . .                                                                                                                                                                                                                                                                                                                                         |          |         |                               |
| (LQG)                                                                                                                                                                                                                                                                                                                                               |          |         |                               |
| Policy Gradient (REINFORCE)                                                                                                                                                                                                                                                                                                                         |          |         |                               |
| Linear Quadratic Gaussian                                                                                                                                                                                                                                                                                                                           |          |         |                               |
| 16.4                                                                                                                                                                                                                                                                                                                                                |          |         |                               |

Part I Supervised learning

Let's start by talking about a few examples of supervised learning problems. Suppose we have a dataset giving the living areas and prices of 47 houses from Portland, Oregon:

| Living area (feet 2   | Price (1000 $ s)   |
|-----------------------|--------------------|
| 2104                  | 400                |
| 1600                  | 330                |
| 2400                  | 369                |
| 1416                  | 232                |
| 3000                  | 540                |
| . . .                 | . . .              |

We can plot this data:


> [Vision Analysis]: ```json
{
  "chart": {
    "type": "scatter",
    "x_axis": {
      "label": "Variable X",
      "units": "units_X",
      "scale": "linear",
      "min": 1000,
      "max": 5000
    },
    "y_axis": {
      "label": "Variable Y",
      "units": "units_Y",
      "scale": "linear",
      "min": 0,
      "max": 1000
    },
    "data_series": [
      {
        "name": "Scatter data",
        "points": [
          [1000, 100],
          [1500, 200],
          [2000, 300],
          [2500, 400],
          [3000, 500],
          [3500, 600],
          [4000, 700],
          [4500, 800],
          [5000, 900]
        ]
      }
    ]
  }
}
```


Given data like this, how can we learn to predict the prices of other houses in Portland, as a function of the size of their living areas?

To establish notation for future use, we'll use x ( i ) to denote the 'input' variables (living area in this example), also called input features , and y ( i ) to denote the 'output' or target variable that we are trying to predict (price). A pair ( x ( i ) , y ( i ) ) is called a training example , and the dataset that we'll be using to learn-a list of n training examples { ( x ( i ) , y ( i ) ); i = 1 , . . . , n } -is called a training set . Note that the superscript '( i )' in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X denote the space of input values, and Y the space of output values. In this example, X = Y = R .

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X ↦→ Y so that h ( x ) is a 'good' predictor for the corresponding value of y . For historical reasons, this

function h is called a hypothesis . Seen pictorially, the process is therefore like this:


> [Vision Analysis]: The image is a flowchart describing a machine learning process. Here is the technical description:

1. **Components and Labels:**
   - **Training set:** Represents the dataset used for training the model.
   - **Learning:** Indicates the process of training the model.
   - **h:** Represents the predicted values or the output of the model.
   - **x:** Represents the input data or the features used for prediction.
   - **algorithm:** Indicates the machine learning algorithm used for training.
   - **predicted values:** Represents the output of the model, which is the predicted values for the input data.

2. **Signal/Data Flow Directions:**
   - The flow starts from the **Training set**.
   - The **Training set** is connected to the **Learning** process.
   - The **Learning** process is connected to the **algorithm**.
   - The **algorithm** is connected to the **predicted values**.
   - The **x** (input data) is connected to the **h** (predicted values).

3. **Key Values or Inflection Points:**
   - The flowchart does not explicitly show any specific key values or inflection points. It focuses on the process flow rather than specific numerical values.

This description is based on the visual elements and connections in the flowchart.


When the target variable that we're trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

Chapter 1

Linear regression

To make our housing example more interesting, let's consider a slightly richer dataset in which we also know the number of bedrooms in each house:

| Living area (feet 2   | #bedrooms   | Price (1000 $ s)   |
|-----------------------|-------------|--------------------|
| 2104                  | 3           | 400                |
| 1600                  | 3           | 330                |
| 2400                  | 3           | 369                |
| 1416                  | 2           | 232                |
| 3000                  | 4           | 540                |
| . . .                 | . . .       | . . .              |

Here, the x 's are two-dimensional vectors in R 2 . For instance, x ( i ) 1 is the living area of the i -th house in the training set, and x ( i ) 2 is its number of bedrooms. (In general, when designing a learning problem, it will be up to you to decide what features to choose, so if you are out in Portland gathering housing data, you might also decide to include other features such as whether each house has a fireplace, the number of bathrooms, and so on. We'll say more about feature selection later, but for now let's take the features as given.)

To perform supervised learning, we must decide how we're going to represent functions/hypotheses h in a computer. As an initial choice, let's say we decide to approximate y as a linear function of x :

Here, the θ i 's are the parameters (also called weights ) parameterizing the space of linear functions mapping from X to Y . When there is no risk of

confusion, we will drop the θ subscript in h θ ( x ), and write it more simply as h ( x ). To simplify our notation, we also introduce the convention of letting x 0 = 1 (this is the intercept term ), so that

where on the right-hand side above we are viewing θ and x both as vectors, and here d is the number of input variables (not counting x 0 ).

Now, given a training set, how do we pick, or learn, the parameters θ ? One reasonable method seems to be to make h ( x ) close to y , at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the θ 's, how close the h ( x ( i ) )'s are to the corresponding y ( i ) 's. We define the cost function :

If you've seen linear regression before, you may recognize this as the familiar least-squares cost function that gives rise to the ordinary least squares regression model. Whether or not you have seen it previously, let's keep going, and we'll eventually show this to be a special case of a much broader family of algorithms.

1.1 LMS algorithm

We want to choose θ so as to minimize J ( θ ). To do so, let's use a search algorithm that starts with some 'initial guess' for θ , and that repeatedly changes θ to make J ( θ ) smaller, until hopefully we converge to a value of θ that minimizes J ( θ ). Specifically, let's consider the gradient descent algorithm, which starts with some initial θ , and repeatedly performs the update:

(This update is simultaneously performed for all values of j = 0 , . . . , d .) Here, α is called the learning rate . This is a very natural algorithm that repeatedly takes a step in the direction of steepest decrease of J .

In order to implement this algorithm, we have to work out what is the partial derivative term on the right hand side. Let's first work it out for the

case of if we have only one training example ( x, y ), so that we can neglect the sum in the definition of J . We have:

For a single training example, this gives the update rule: 1

The rule is called the LMS update rule (LMS stands for 'least mean squares'), and is also known as the Widrow-Hoff learning rule. This rule has several properties that seem natural and intuitive. For instance, the magnitude of the update is proportional to the error term ( y ( i ) -h θ ( x ( i ) )); thus, for instance, if we are encountering a training example on which our prediction nearly matches the actual value of y ( i ) , then we find that there is little need to change the parameters; in contrast, a larger change to the parameters will be made if our prediction h θ ( x ( i ) ) has a large error (i.e., if it is very far from y ( i ) ).

We'd derived the LMS rule for when there was only a single training example. There are two ways to modify this method for a training set of more than one example. The first is replace it with the following algorithm:

Repeat until convergence {

}

1 We use the notation ' a := b ' to denote an operation (in a computer program) in which we set the value of a variable a to be equal to the value of b . In other words, this operation overwrites a with the value of b . In contrast, we will write ' a = b ' when we are asserting a statement of fact, that the value of a is equal to the value of b .

By grouping the updates of the coordinates into an update of the vector θ , we can rewrite update (1.1) in a slightly more succinct way:

The reader can easily verify that the quantity in the summation in the update rule above is just ∂J ( θ ) /∂θ j (for the original definition of J ). So, this is simply gradient descent on the original cost function J . This method looks at every example in the entire training set on every step, and is called batch gradient descent . Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": {
    "chart_type": "contour plot",
    "x_axis_label": "X",
    "x_axis_units": "units",
    "y_axis_label": "Y",
    "y_axis_units": "units",
    "data_series_names": ["Series 1", "Series 2", "Series 3", "Series 4"],
    "specific_key_values": {
      "Series 1": {
        "inflection_point": [25, 25]
      },
      "Series 2": {
        "inflection_point": [30, 30]
      },
      "Series 3": {
        "inflection_point": [35, 35]
      },
      "Series 4": {
        "inflection_point": [40, 40]
      }
    }
  },
  "diagrams_and_schematics": {
    "diagram_type": "schematic diagram",
    "components": {
      "component_1": {
        "label": "Component 1",
        "position": [10, 10]
      },
      "component_2": {
        "label": "Component 2",
        "position": [20, 20]
      },
      "component_3": {
        "label": "Component 3",
        "position": [30, 30]
      },
      "component_4": {
        "label": "Component 4",
        "position": [40, 40]
      }
    },
    "connections": {
      "connection_1": {
        "from": "Component 1",
        "to": "Component 2",
        "direction": "right"
      },
      "connection_2": {
        "from": "Component 2",
        "to": "Component 3",
        "direction": "right"
      },
      "connection_3": {
        "from": "Component 3",
        "to": "Component 4",
        "direction": "right"
      }
    },
    "signal_flow_direction": {
      "signal_flow": {
        "from": "Component 1",
        "to": "Component 4",
        "direction": "right"
      }
    }
  },
  "tables": []
}
```


The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x 's in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through.

When we run batch gradient descent to fit θ on our previous dataset, to learn to predict housing price as a function of living area, we obtain θ 0 = 71 . 27, θ 1 = 0 . 1345. If we plot h θ ( x ) as a function of x (area), along with the training data, we obtain the following figure:


> [Vision Analysis]: ```json
{
  "chart": {
    "title": "Repeating phrases",
    "x_axis": {
      "label": "Number of words",
      "units": "words",
      "scale": "linear",
      "min": 1000,
      "max": 4000
    },
    "y_axis": {
      "label": "Length of phrase (in words)",
      "units": "words",
      "scale": "linear",
      "min": 10,
      "max": 50
    },
    "data_series": [
      {
        "name": "phrase length",
        "type": "scatter",
        "data_points": [
          [1000, 10],
          [1500, 15],
          [2000, 20],
          [2500, 25],
          [3000, 30],
          [3500, 35],
          [4000, 40]
        ]
      },
      {
        "name": "trend line",
        "type": "line",
        "data_points": [
          [1000, 10],
          [4000, 40]
        ],
        "line_style": "solid",
        "color": "blue"
      }
    ]
  }
}
```


If the number of bedrooms were included as one of the input features as well, we get θ 0 = 89 . 60 , θ 1 = 0 . 1392, θ 2 = -8 . 738.

The above results were obtained with batch gradient descent. There is an alternative to batch gradient descent that also works very well. Consider the following algorithm:

Loop { for i = 1 to n , { θ j := θ j + α ( y ( i ) -h θ ( x ( i ) ) ) x ( i ) j , (for every j ) (1.2) } }

By grouping the updates of the coordinates into an update of the vector θ , we can rewrite update (1.2) in a slightly more succinct way:

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called stochastic gradient descent (also incremental gradient descent ). Whereas batch gradient descent has to scan through the entire training set before taking a single step-a costly operation if n is large-stochastic gradient descent can start making progress right away, and

continues to make progress with each example it looks at. Often, stochastic gradient descent gets θ 'close' to the minimum much faster than batch gradient descent. (Note however that it may never 'converge' to the minimum, and the parameters θ will keep oscillating around the minimum of J ( θ ); but in practice most of the values near the minimum will be reasonably good approximations to the true minimum. 2 ) For these reasons, particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

1.2 The normal equations

Gradient descent gives one way of minimizing J . Let's discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize J by explicitly taking its derivatives with respect to the θ j 's, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let's introduce some notation for doing calculus with matrices.

1.2.1 Matrix derivatives

For a function f : R n × d ↦→ R mapping from n -byd matrices to the real numbers, we define the derivative of f with respect to A to be:

Thus, the gradient ∇ A f ( A ) is itself an n -byd matrix, whose ( i, j )-element is ∂f/∂A ij . For example, suppose A = [ A 11 A 12 A 21 A 22 ] is a 2-by-2 matrix, and the function f : R 2 × 2 ↦→ R is given by

2 By slowly letting the learning rate α decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather than merely oscillate around the minimum.

Here, A ij denotes the ( i, j ) entry of the matrix A . We then have

1.2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of θ that minimizes J ( θ ). We begin by re-writing J in matrix-vectorial notation.

Given a training set, define the design matrix X to be the n -byd matrix (actually n -byd + 1, if we include the intercept term) that contains the training examples' input values in its rows:

Also, let glyph[vector] y be the n -dimensional vector containing all the target values from the training set:

Now, since h θ ( x ( i ) ) = ( x ( i ) ) T θ , we can easily verify that

Thus, using the fact that for a vector z , we have that z T z = ∑ i z 2 i :

Finally, to minimize J , let's find its derivatives with respect to θ . Hence,

In the third step, we used the fact that a T b = b T a , and in the fifth step used the facts ∇ x b T x = b and ∇ x x T Ax = 2 Ax for symmetric matrix A (for more details, see Section 4.3 of 'Linear Algebra Review and Reference'). To minimize J , we set its derivatives to zero, and obtain the normal equations :

Thus, the value of θ that minimizes J ( θ ) is given in closed form by the equation

1.3 Probabilistic interpretation

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function J , be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

Let us assume that the target variables and the inputs are related via the equation

3 Note that in the above step, we are implicitly assuming that X T X is an invertible matrix. This can be checked before calculating the inverse. If either the number of linearly independent examples is fewer than the number of features, or if the features are not linearly independent, then X T X will not be invertible. Even in such cases, it is possible to 'fix' the situation with additional techniques, which we skip here for the sake of simplicty.

where glyph[epsilon1] ( i ) is an error term that captures either unmodeled effects (such as if there are some features very pertinent to predicting housing price, but that we'd left out of the regression), or random noise. Let us further assume that the glyph[epsilon1] ( i ) are distributed IID (independently and identically distributed) according to a Gaussian distribution (also called a Normal distribution) with mean zero and some variance σ 2 . We can write this assumption as ' glyph[epsilon1] ( i ) ∼ N (0 , σ 2 ).' I.e., the density of glyph[epsilon1] ( i ) is given by

This implies that

The notation ' p ( y ( i ) | x ( i ) ; θ )' indicates that this is the distribution of y ( i ) given x ( i ) and parameterized by θ . Note that we should not condition on θ (' p ( y ( i ) | x ( i ) , θ )'), since θ is not a random variable. We can also write the distribution of y ( i ) as y ( i ) | x ( i ) ; θ ∼ N ( θ T x ( i ) , σ 2 ).

Given X (the design matrix, which contains all the x ( i ) 's) and θ , what is the distribution of the y ( i ) 's? The probability of the data is given by p ( glyph[vector] y | X ; θ ). This quantity is typically viewed a function of glyph[vector] y (and perhaps X ), for a fixed value of θ . When we wish to explicitly view this as a function of θ , we will instead call it the likelihood function:

Note that by the independence assumption on the glyph[epsilon1] ( i ) 's (and hence also the y ( i ) 's given the x ( i ) 's), this can also be written

Now, given this probabilistic model relating the y ( i ) 's and the x ( i ) 's, what is a reasonable way of choosing our best guess of the parameters θ ? The principal of maximum likelihood says that we should choose θ so as to make the data as high probability as possible. I.e., we should choose θ to maximize L ( θ ).

Instead of maximizing L ( θ ), we can also maximize any strictly increasing function of L ( θ ). In particular, the derivations will be a bit simpler if we instead maximize the log likelihood glyph[lscript] ( θ ):

Hence, maximizing glyph[lscript] ( θ ) gives the same answer as minimizing

which we recognize to be J ( θ ), our original least-squares cost function.

To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood estimate of θ . This is thus one set of assumptions under which least-squares regression can be justified as a very natural method that's just doing maximum likelihood estimation. (Note however that the probabilistic assumptions are by no means necessary for least-squares to be a perfectly good and rational procedure, and there may-and indeed there are-other natural assumptions that can also be used to justify it.)

Note also that, in our previous discussion, our final choice of θ did not depend on what was σ 2 , and indeed we'd have arrived at the same result even if σ 2 were unknown. We will use this fact again later, when we talk about the exponential family and generalized linear models.

1.4 Locally weighted linear regression (optional reading)

Consider the problem of predicting y from x ∈ R . The leftmost figure below shows the result of fitting a y = θ 0 + θ 1 x to a dataset. We see that the data doesn't really lie on straight line, and so the fit is not very good.


> [Vision Analysis]: ```json
{
  "charts": [
    {
      "type": "line",
      "title": "Graph 1",
      "x_axis": {
        "label": "X-axis",
        "units": "units",
        "scale": "linear"
      },
      "y_axis": {
        "label": "Y-axis",
        "units": "units",
        "scale": "linear"
      },
      "data_series": [
        {
          "name": "Series 1",
          "points": [
            [-10, 10],
            [0, 0],
            [10, 10]
          ]
        }
      ]
    },
    {
      "type": "line",
      "title": "Graph 2",
      "x_axis": {
        "label": "X-axis",
        "units": "units",
        "scale": "linear"
      },
      "y_axis": {
        "label": "Y-axis",
        "units": "units",
        "scale": "linear"
      },
      "data_series": [
        {
          "name": "Series 1",
          "points": [
            [-10, 10],
            [0, 10],
            [10, 0]
          ]
        }
      ]
    },
    {
      "type": "line",
      "title": "Graph 3",
      "x_axis": {
        "label": "X-axis",
        "units": "units",
        "scale": "linear"
      },
      "y_axis": {
        "label": "Y-axis",
        "units": "units",
        "scale": "linear"
      },
      "data_series": [
        {
          "name": "Series 1",
          "points": [
            [-10, 10],
            [0, 0],
            [10, 0]
          ]
        }
      ]
    }
  ]
}
```


x

x

x

Instead, if we had added an extra feature x 2 , and fit y = θ 0 + θ 1 x + θ 2 x 2 , then we obtain a slightly better fit to the data. (See middle figure) Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5-th order polynomial y = ∑ 5 j =0 θ j x j . We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices ( y ) for different living areas ( x ). Without formally defining what these terms mean, we'll say the figure on the left shows an instance of underfitting -in which the data clearly shows structure not captured by the model-and the figure on the right is an example of overfitting . (Later in this class, when we talk about learning theory we'll formalize some of these notions, and also define more carefully just what it means for a hypothesis to be good or bad.)

As discussed previously, and as shown in the example above, the choice of features is important to ensuring good performance of a learning algorithm. (When we talk about model selection, we'll also see algorithms for automatically choosing a good set of features.) In this section, let us briefly talk about the locally weighted linear regression (LWR) algorithm which, assuming there is sufficient training data, makes the choice of features less critical. This treatment will be brief, since you'll get a chance to explore some of the properties of the LWR algorithm yourself in the homework.

In the original linear regression algorithm, to make a prediction at a query point x (i.e., to evaluate h ( x )), we would:

Fit θ to minimize ∑ i ( y ( i ) -θ T x ( i ) ) 2 .

2. Output θ T x .

In contrast, the locally weighted linear regression algorithm does the following:

Fit θ to minimize ∑ i w ( i ) ( y ( i ) -θ T x ( i ) ) 2 .

Output θ T x .

Here, the w ( i ) 's are non-negative valued weights . Intuitively, if w ( i ) is large for a particular value of i , then in picking θ , we'll try hard to make ( y ( i ) -θ T x ( i ) ) 2 small. If w ( i ) is small, then the ( y ( i ) -θ T x ( i ) ) 2 error term will be pretty much ignored in the fit.

A fairly standard choice for the weights is 4

Note that the weights depend on the particular point x at which we're trying to evaluate x . Moreover, if | x ( i ) -x | is small, then w ( i ) is close to 1; and if | x ( i ) -x | is large, then w ( i ) is small. Hence, θ is chosen giving a much higher 'weight' to the (errors on) training examples close to the query point x . (Note also that while the formula for the weights takes a form that is cosmetically similar to the density of a Gaussian distribution, the w ( i ) 's do not directly have anything to do with Gaussians, and in particular the w ( i ) are not random variables, normally distributed or otherwise.) The parameter τ controls how quickly the weight of a training example falls off with distance of its x ( i ) from the query point x ; τ is called the bandwidth parameter, and is also something that you'll get to experiment with in your homework.

Locally weighted linear regression is the first example we're seeing of a non-parametric algorithm. The (unweighted) linear regression algorithm that we saw earlier is known as a parametric learning algorithm, because it has a fixed, finite number of parameters (the θ i 's), which are fit to the data. Once we've fit the θ i 's and stored them away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions using locally weighted linear regression, we need to keep the entire training set around. The term 'non-parametric' (roughly) refers to the fact that the amount of stuff we need to keep in order to represent the hypothesis h grows linearly with the size of the training set.

4 If x is vector-valued, this is generalized to be w ( i ) = exp( -( x ( i ) -x ) T ( x ( i ) -x ) / (2 τ 2 )), or w ( i ) = exp( -( x ( i ) -x ) T Σ -1 ( x ( i ) -x ) / (2 τ 2 )), for an appropriate choice of τ or Σ.

Chapter 2

Classification and logistic regression

Let's now talk about the classification problem. This is just like the regression problem, except that the values y we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x ( i ) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. 0 is also called the negative class , and 1 the positive class , and they are sometimes also denoted by the symbols '-' and '+.' Given x ( i ) , the corresponding y ( i ) is also called the label for the training example.

2.1 Logistic regression

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x . However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn't make sense for h θ ( x ) to take values larger than 1 or smaller than 0 when we know that y ∈ { 0 , 1 } .

To fix this, let's change the form for our hypotheses h θ ( x ). We will choose

where

is called the logistic function or the sigmoid function . Here is a plot showing g ( z ):


> [Vision Analysis]: ```json
{
  "chart": {
    "type": "line",
    "x_axis": {
      "label": "X",
      "units": "dimensionless",
      "scale": "linear",
      "min": -5,
      "max": 5
    },
    "y_axis": {
      "label": "Y",
      "units": "dimensionless",
      "scale": "linear",
      "min": 0,
      "max": 1
    },
    "data_series": [
      {
        "name": "Data Series 1",
        "values": [
          [-5, 0],
          [-4, 0.1],
          [-3, 0.3],
          [-2, 0.6],
          [-1, 0.8],
          [0, 0.9],
          [1, 0.95],
          [2, 0.98],
          [3, 0.99],
          [4, 0.995],
          [5, 1]
        ]
      }
    ]
  }
}
```


Notice that g ( z ) tends towards 1 as z → ∞ , and g ( z ) tends towards 0 as z →-∞ . Moreover, g(z), and hence also h ( x ), is always bounded between 0 and 1. As before, we are keeping the convention of letting x 0 = 1, so that θ T x = θ 0 + ∑ d j =1 θ j x j .

For now, let's take the choice of g as given. Other functions that smoothly increase from 0 to 1 can also be used, but for a couple of reasons that we'll see later (when we talk about GLMs, and when we talk about generative learning algorithms), the choice of the logistic function is a fairly natural one. Before moving on, here's a useful property of the derivative of the sigmoid function, which we write as g ′ :

So, given the logistic regression model, how do we fit θ for it? Following how we saw least squares regression could be derived as the maximum likelihood estimator under a set of assumptions, let's endow our classification model with a set of probabilistic assumptions, and then fit the parameters via maximum likelihood.

Let us assume that

Note that this can be written more compactly as

Assuming that the n training examples were generated independently, we can then write down the likelihood of the parameters as

As before, it will be easier to maximize the log likelihood:

How do we maximize the likelihood? Similar to our derivation in the case of linear regression, we can use gradient ascent. Written in vectorial notation, our updates will therefore be given by θ := θ + α ∇ θ glyph[lscript] ( θ ). (Note the positive rather than negative sign in the update formula, since we're maximizing, rather than minimizing, a function now.) Let's start by working with just one training example ( x, y ), and take derivatives to derive the stochastic gradient ascent rule:

Above, we used the fact that g ′ ( z ) = g ( z )(1 -g ( z )). This therefore gives us the stochastic gradient ascent rule

If we compare this to the LMS update rule, we see that it looks identical; but this is not the same algorithm, because h θ ( x ( i ) ) is now defined as a non-linear function of θ T x ( i ) . Nonetheless, it's a little surprising that we end up with the same update rule for a rather different algorithm and learning problem. Is this coincidence, or is there a deeper reason behind this? We'll answer this when we get to GLM models.

Remark 2.1.1: An alternative notational viewpoint of the same loss function is also useful, especially for Section 7.1 where we study nonlinear models. Let glyph[lscript] logistic : R ×{ 0 , 1 } → R ≥ 0 be the logistic loss defined as

One can verify by plugging in h θ ( x ) = 1 / (1 + e -θ glyph[latticetop] x ) that the negative loglikelihood (the negation of glyph[lscript] ( θ ) in equation (2.1)) can be re-written as

Oftentimes θ glyph[latticetop] x or t is called the logit . Basic calculus gives us that

Then, using the chain rule, we have that

which is consistent with the derivation in equation (2.2). We will see this viewpoint can be extended nonlinear models in Section 7.1.

2.2 Digression: the perceptron learning algorithm

We now digress to talk briefly about an algorithm that's of some historical interest, and that we will also return to later when we talk about learning

theory. Consider modifying the logistic regression method to 'force' it to output values that are either 0 or 1 or exactly. To do so, it seems natural to change the definition of g to be the threshold function:

If we then let h θ ( x ) = g ( θ T x ) as before but using this modified definition of g , and if we use the update rule

then we have the perceptron learning algorithn .

In the 1960s, this 'perceptron' was argued to be a rough model for how individual neurons in the brain work. Given how simple the algorithm is, it will also provide a starting point for our analysis when we talk about learning theory later in this class. Note however that even though the perceptron may be cosmetically similar to the other algorithms we talked about, it is actually a very different type of algorithm than logistic regression and least squares linear regression; in particular, it is difficult to endow the perceptron's predictions with meaningful probabilistic interpretations, or derive the perceptron as a maximum likelihood estimation algorithm.

2.3 Multi-class classification

Consider a classification problem in which the response variable y can take on any one of k values, so y ∈ { 1 , 2 , . . . , k } . For example, rather than classifying emails into the two classes spam or not-spam-which would have been a binary classification problem-we might want to classify them into three classes, such as spam, personal mails, and work-related mails. The label / response variable is still discrete, but can now take on more than two values. We will thus model it as distributed according to a multinomial distribution.

In this case, p ( y | x ; θ ) is a distribution over k possible discrete outcomes and is thus a multinomial distribution. Recall that a multinomial distribution involves k numbers φ 1 , . . . , φ k specifying the probability of each of the outcomes. Note that these numbers must satisfy ∑ k i =1 φ i = 1. We will design a parameterized model that outputs φ 1 , . . . , φ k satisfying this constraint given the input x .

We introduce k groups of parameters θ 1 , . . . , θ k , each of them being a vector in R d . Intuitively, we would like to use θ glyph[latticetop] 1 x, . . . , θ glyph[latticetop] k x to represent

φ 1 , . . . , φ k , the probabilities P ( y = 1 | x ; θ ) , . . . , P ( y = k | x ; θ ). However, there are two issues with such a direct approach. First, θ glyph[latticetop] j x is not necessarily within [0 , 1]. Second, the summation of θ glyph[latticetop] j x 's is not necessarily 1. Thus, instead, we will use the softmax function to turn ( θ glyph[latticetop] 1 x, · · · , θ glyph[latticetop] k x ) into a probability vector with nonnegative entries that sum up to 1.

Define the softmax function softmax : R k → R k as

The inputs to the softmax function, the vector t here, are often called logits . Note that by definition, the output of the softmax function is always a probability vector whose entries are nonnegative and sum up to 1.

Let ( t 1 , . . . , t k ) = ( θ glyph[latticetop] 1 x, · · · , θ glyph[latticetop] k x ). We apply the softmax function to ( t 1 , . . . , t k ), and use the output as the probabilities P ( y = 1 | x ; θ ) , . . . , P ( y = k | x ; θ ). We obtain the following probabilistic model:

For notational convenience, we will let φ i = exp( t i ) ∑ k j =1 exp( t j ) . More succinctly, the equation above can be written as:

Next, we compute the negative log-likelihood of a single example ( x, y ).

Thus, the loss function, the negative log-likelihood of the training data, is given as

It's convenient to define the cross-entropy loss glyph[lscript] ce : R k ×{ 1 , . . . , k } → R ≥ 0 , which modularizes in the complex equation above: 1

With this notation, we can simply rewrite equation (2.13) as

Moreover, conveniently, the cross-entropy loss also has a simple gradient. Let t = ( t 1 , . . . , t k ), and recall φ i = exp( t i ) ∑ k j =1 exp( t j ) . By basic calculus, we can derive

where 1 {·} is the indicator function, that is, 1 { y = i } = 1 if y = i , and 1 { y = i } = 0 if y = i . Alternatively, in vectorized notations, we have the following form which will be useful for Chapter 7:

where e s ∈ R k is the s -th natural basis vector (where the s -th entry is 1 and all other entries are zeros.) Using Chain rule, we have that

Therefore, the gradient of the loss with respect to the part of parameter θ i is

where φ ( j ) i = exp( θ glyph[latticetop] i x ( j ) ) ∑ k s =1 exp( θ glyph[latticetop] s x ( j ) ) is the probability that the model predicts item i for example x ( j ) . With the gradients above, one can implement (stochastic) gradient descent to minimize the loss function glyph[lscript] ( θ ).

1 There are some ambiguity in the naming here. Some people call the cross-entropy loss the function that maps the probability vector (the φ in our language) and label y to the final real number, and call our version of cross-entropy loss softmax-cross-entropy loss. We choose our current naming convention because it's consistent with the naming of most modern deep learning library such as PyTorch and Jax.

glyph[negationslash]


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Graph 1",
      "x_axis_label": "Time (s)",
      "y_axis_label": "Energy (J)",
      "data_series": [
        {
          "name": "Curve 1",
          "line_color": "blue",
          "line_style": "solid",
          "data_points": [
            [0, 0],
            [1, 1],
            [2, 4],
            [3, 9],
            [4, 16]
          ]
        }
      ],
      "scale": "logarithmic"
    },
    {
      "title": "Graph 2",
      "x_axis_label": "Distance (m)",
      "y_axis_label": "Force (N)",
      "data_series": [
        {
          "name": "Curve 1",
          "line_color": "blue",
          "line_style": "solid",
          "data_points": [
            [0, 0],
            [1, 1],
            [2, 4],
            [3, 9],
            [4, 16]
          ]
        }
      ],
      "scale": "linear"
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "Diagram 1",
      "components": [
        {
          "label": "Component A",
          "position": [0, 0]
        },
        {
          "label": "Component B",
          "position": [1, 1]
        },
        {
          "label": "Component C",
          "position": [2, 2]
        }
      ],
      "connections": [
        {
          "source": "Component A",
          "target": "Component B",
          "signal": "Signal 1"
        },
        {
          "source": "Component B",
          "target": "Component C",
          "signal": "Signal 2"
        }
      ],
      "signal_flow_direction": "from left to right"
    }
  ]
}
```


2.4 Another algorithm for maximizing glyph[lscript] ( θ )

Returning to logistic regression with g ( z ) being the sigmoid function, let's now talk about a different algorithm for maximizing glyph[lscript] ( θ ).

To get us started, let's consider Newton's method for finding a zero of a function. Specifically, suppose we have some function f : R ↦→ R , and we wish to find a value of θ so that f ( θ ) = 0. Here, θ ∈ R is a real number. Newton's method performs the following update:

This method has a natural interpretation in which we can think of it as approximating the function f via a linear function that is tangent to f at the current guess θ , solving for where that linear function equals to zero, and letting the next guess for θ be where that linear function is zero.

Here's a picture of the Newton's method in action:

In the leftmost figure, we see the function f plotted along with the line y = 0. We're trying to find θ so that f ( θ ) = 0; the value of θ that achieves this is about 1.3. Suppose we initialized the algorithm with θ = 4 . 5. Newton's method then fits a straight line tangent to f at θ = 4 . 5, and solves for the where that line evaluates to 0. (Middle figure.) This give us the next guess for θ , which is about 2.8. The rightmost figure shows the result of running one more iteration, which the updates θ to about 1.8. After a few more iterations, we rapidly approach θ = 1 . 3.

Newton's method gives a way of getting to f ( θ ) = 0. What if we want to use it to maximize some function glyph[lscript] ? The maxima of glyph[lscript] correspond to points where its first derivative glyph[lscript] ′ ( θ ) is zero. So, by letting f ( θ ) = glyph[lscript] ′ ( θ ), we can use the same algorithm to maximize glyph[lscript] , and we obtain update rule:

(Something to think about: How would this change if we wanted to use Newton's method to minimize rather than maximize a function?)

Lastly, in our logistic regression setting, θ is vector-valued, so we need to generalize Newton's method to this setting. The generalization of Newton's method to this multidimensional setting (also called the Newton-Raphson method) is given by

Here, ∇ θ glyph[lscript] ( θ ) is, as usual, the vector of partial derivatives of glyph[lscript] ( θ ) with respect to the θ i 's; and H is an d -byd matrix (actually, d +1 -by -d+1, assuming that we include the intercept term) called the Hessian , whose entries are given by

Newton's method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to get very close to the minimum. One iteration of Newton's can, however, be more expensive than one iteration of gradient descent, since it requires finding and inverting an d -byd Hessian; but so long as d is not too large, it is usually much faster overall. When Newton's method is applied to maximize the logistic regression log likelihood function glyph[lscript] ( θ ), the resulting method is also called Fisher scoring .

Chapter 3

Generalized linear models

So far, we've seen a regression example, and a classification example. In the regression example, we had y | x ; θ ∼ N ( µ, σ 2 ), and in the classification one, y | x ; θ ∼ Bernoulli( φ ), for some appropriate definitions of µ and φ as functions of x and θ . In this section, we will show that both of these methods are special cases of a broader family of models, called Generalized Linear Models (GLMs). 1 We will also show how other models in the GLM family can be derived and applied to other classification and regression problems.

3.1 The exponential family

To work our way up to GLMs, we will begin by defining exponential family distributions. We say that a class of distributions is in the exponential family if it can be written in the form

Here, η is called the natural parameter (also called the canonical parameter ) of the distribution; T ( y ) is the sufficient statistic (for the distributions we consider, it will often be the case that T ( y ) = y ); and a ( η ) is the log partition function . The quantity e -a ( η ) essentially plays the role of a normalization constant, that makes sure the distribution p ( y ; η ) sums/integrates over y to 1.

A fixed choice of T , a and b defines a family (or set) of distributions that is parameterized by η ; as we vary η , we then get different distributions within this family.

1 The presentation of the material in this section takes inspiration from Michael I. Jordan, Learning in graphical models (unpublished book draft), and also McCullagh and Nelder, Generalized Linear Models (2nd ed.) .

We now show that the Bernoulli and the Gaussian distributions are examples of exponential family distributions. The Bernoulli distribution with mean φ , written Bernoulli( φ ), specifies a distribution over y ∈ { 0 , 1 } , so that p ( y = 1; φ ) = φ ; p ( y = 0; φ ) = 1 -φ . As we vary φ , we obtain Bernoulli distributions with different means. We now show that this class of Bernoulli distributions, ones obtained by varying φ , is in the exponential family; i.e., that there is a choice of T , a and b so that Equation (3.1) becomes exactly the class of Bernoulli distributions.

We write the Bernoulli distribution as:

Thus, the natural parameter is given by η = log( φ/ (1 -φ )). Interestingly, if we invert this definition for η by solving for φ in terms of η , we obtain φ = 1 / (1 + e -η ). This is the familiar sigmoid function! This will come up again when we derive logistic regression as a GLM. To complete the formulation of the Bernoulli distribution as an exponential family distribution, we also have

This shows that the Bernoulli distribution can be written in the form of Equation (3.1), using an appropriate choice of T , a and b .

Let's now move on to consider the Gaussian distribution. Recall that, when deriving linear regression, the value of σ 2 had no effect on our final choice of θ and h θ ( x ). Thus, we can choose an arbitrary value for σ 2 without changing anything. To simplify the derivation below, let's set σ 2 = 1. 2 We

2 If we leave σ 2 as a variable, the Gaussian distribution can also be shown to be in the exponential family, where η ∈ R 2 is now a 2-dimension vector that depends on both µ and σ . For the purposes of GLMs, however, the σ 2 parameter can also be treated by considering a more general definition of the exponential family: p ( y ; η, τ ) = b ( a, τ ) exp(( η T T ( y ) -a ( η )) /c ( τ )). Here, τ is called the dispersion parameter , and for the Gaussian, c ( τ ) = σ 2 ; but given our simplification above, we won't need the more general definition for the examples we will consider here.

then have:

Thus, we see that the Gaussian is in the exponential family, with

There're many other distributions that are members of the exponential family: The multinomial (which we'll see later), the Poisson (for modelling count-data; also see the problem set); the gamma and the exponential (for modelling continuous, non-negative random variables, such as timeintervals); the beta and the Dirichlet (for distributions over probabilities); and many more. In the next section, we will describe a general 'recipe' for constructing models in which y (given x and θ ) comes from any of these distributions.

3.2 Constructing GLMs

Suppose you would like to build a model to estimate the number y of customers arriving in your store (or number of page-views on your website) in any given hour, based on certain features x such as store promotions, recent advertising, weather, day-of-week, etc. We know that the Poisson distribution usually gives a good model for numbers of visitors. Knowing this, how can we come up with a model for our problem? Fortunately, the Poisson is an exponential family distribution, so we can apply a Generalized Linear Model (GLM). In this section, we will we will describe a method for constructing GLM models for problems such as these.

More generally, consider a classification or regression problem where we would like to predict the value of some random variable y as a function of x . To derive a GLM for this problem, we will make the following three assumptions about the conditional distribution of y given x and about our model:

y | x ; θ ∼ ExponentialFamily( η ). I.e., given x and θ , the distribution of y follows some exponential family distribution, with parameter η .

Given x , our goal is to predict the expected value of T ( y ) given x . In most of our examples, we will have T ( y ) = y , so this means we would like the prediction h ( x ) output by our learned hypothesis h to satisfy h ( x ) = E[ y | x ]. (Note that this assumption is satisfied in the choices for h θ ( x ) for both logistic regression and linear regression. For instance, in logistic regression, we had h θ ( x ) = p ( y = 1 | x ; θ ) = 0 · p ( y = 0 | x ; θ ) + 1 · p ( y = 1 | x ; θ ) = E[ y | x ; θ ].)

The natural parameter η and the inputs x are related linearly: η = θ T x . (Or, if η is vector-valued, then η i = θ T i x .)

The third of these assumptions might seem the least well justified of the above, and it might be better thought of as a 'design choice' in our recipe for designing GLMs, rather than as an assumption per se. These three assumptions/design choices will allow us to derive a very elegant class of learning algorithms, namely GLMs, that have many desirable properties such as ease of learning. Furthermore, the resulting models are often very effective for modelling different types of distributions over y ; for example, we will shortly show that both logistic regression and ordinary least squares can both be derived as GLMs.

3.2.1 Ordinary least squares

To show that ordinary least squares is a special case of the GLM family of models, consider the setting where the target variable y (also called the response variable in GLM terminology) is continuous, and we model the conditional distribution of y given x as a Gaussian N ( µ, σ 2 ). (Here, µ may depend x .) So, we let the ExponentialFamily ( η ) distribution above be the Gaussian distribution. As we saw previously, in the formulation of the Gaussian as an exponential family distribution, we had µ = η . So, we have

The first equality follows from Assumption 2, above; the second equality follows from the fact that y | x ; θ ∼ N ( µ, σ 2 ), and so its expected value is given

by µ ; the third equality follows from Assumption 1 (and our earlier derivation showing that µ = η in the formulation of the Gaussian as an exponential family distribution); and the last equality follows from Assumption 3.

3.2.2 Logistic regression

We now consider logistic regression. Here we are interested in binary classification, so y ∈ { 0 , 1 } . Given that y is binary-valued, it therefore seems natural to choose the Bernoulli family of distributions to model the conditional distribution of y given x . In our formulation of the Bernoulli distribution as an exponential family distribution, we had φ = 1 / (1 + e -η ). Furthermore, note that if y | x ; θ ∼ Bernoulli( φ ), then E[ y | x ; θ ] = φ . So, following a similar derivation as the one for ordinary least squares, we get:

So, this gives us hypothesis functions of the form h θ ( x ) = 1 / (1 + e -θ T x ). If you are previously wondering how we came up with the form of the logistic function 1 / (1 + e -z ), this gives one answer: Once we assume that y conditioned on x is Bernoulli, it arises as a consequence of the definition of GLMs and exponential family distributions.

To introduce a little more terminology, the function g giving the distribution's mean as a function of the natural parameter ( g ( η ) = E[ T ( y ); η ]) is called the canonical response function . Its inverse, g -1 , is called the canonical link function . Thus, the canonical response function for the Gaussian family is just the identify function; and the canonical response function for the Bernoulli is the logistic function. 3

3 Many texts use g to denote the link function, and g -1 to denote the response function; but the notation we're using here, inherited from the early machine learning literature, will be more consistent with the notation used in the rest of the class.

Chapter 4

Generative learning algorithms

So far, we've mainly been talking about learning algorithms that model p ( y | x ; θ ), the conditional distribution of y given x . For instance, logistic regression modeled p ( y | x ; θ ) as h θ ( x ) = g ( θ T x ) where g is the sigmoid function. In these notes, we'll talk about a different type of learning algorithm.

Consider a classification problem in which we want to learn to distinguish between elephants ( y = 1) and dogs ( y = 0), based on some features of an animal. Given a training set, an algorithm like logistic regression or the perceptron algorithm (basically) tries to find a straight line-that is, a decision boundary-that separates the elephants and dogs. Then, to classify a new animal as either an elephant or a dog, it checks on which side of the decision boundary it falls, and makes its prediction accordingly.

Here's a different approach. First, looking at elephants, we can build a model of what elephants look like. Then, looking at dogs, we can build a separate model of what dogs look like. Finally, to classify a new animal, we can match the new animal against the elephant model, and match it against the dog model, to see whether the new animal looks more like the elephants or more like the dogs we had seen in the training set.

Algorithms that try to learn p ( y | x ) directly (such as logistic regression), or algorithms that try to learn mappings directly from the space of inputs X to the labels { 0 , 1 } , (such as the perceptron algorithm) are called discriminative learning algorithms. Here, we'll talk about algorithms that instead try to model p ( x | y ) (and p ( y )). These algorithms are called generative learning algorithms. For instance, if y indicates whether an example is a dog (0) or an elephant (1), then p ( x | y = 0) models the distribution of dogs' features, and p ( x | y = 1) models the distribution of elephants' features.

After modeling p ( y ) (called the class priors ) and p ( x | y ), our algorithm

can then use Bayes rule to derive the posterior distribution on y given x :

Here, the denominator is given by p ( x ) = p ( x | y = 1) p ( y = 1) + p ( x | y = 0) p ( y = 0) (you should be able to verify that this is true from the standard properties of probabilities), and thus can also be expressed in terms of the quantities p ( x | y ) and p ( y ) that we've learned. Actually, if were calculating p ( y | x ) in order to make a prediction, then we don't actually need to calculate the denominator, since

4.1 Gaussian discriminant analysis

The first generative learning algorithm that we'll look at is Gaussian discriminant analysis (GDA). In this model, we'll assume that p ( x | y ) is distributed according to a multivariate normal distribution. Let's talk briefly about the properties of multivariate normal distributions before moving on to the GDA model itself.

4.1.1 The multivariate normal distribution

The multivariate normal distribution in d -dimensions, also called the multivariate Gaussian distribution, is parameterized by a mean vector µ ∈ R d and a covariance matrix Σ ∈ R d × d , where Σ ≥ 0 is symmetric and positive semi-definite. Also written ' N ( µ, Σ)', its density is given by:

In the equation above, ' | Σ | ' denotes the determinant of the matrix Σ.

For a random variable X distributed N ( µ, Σ), the mean is (unsurprisingly) given by µ :

The covariance of a vector-valued random variable Z is defined as Cov( Z ) = E[( Z -E[ Z ])( Z -E[ Z ]) T ]. This generalizes the notion of the variance of a

real-valued random variable. The covariance can also be defined as Cov( Z ) = E[ ZZ T ] -(E[ Z ])(E[ Z ]) T . (You should be able to prove to yourself that these two definitions are equivalent.) If X ∼ N ( µ, Σ), then

Here are some examples of what the density of a Gaussian distribution looks like:


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$\mathcal{F}(x, y) = \exp\left(-\frac{x^2 + y^2}{2}\right)$"
  ],
  "charts_and_graphs": [
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    }
  ]
}
```


The left-most figure shows a Gaussian with mean zero (that is, the 2x1 zero-vector) and covariance matrix Σ = I (the 2x2 identity matrix). A Gaussian with zero mean and identity covariance is also called the standard normal distribution . The middle figure shows the density of a Gaussian with zero mean and Σ = 0 . 6 I ; and in the rightmost figure shows one with , Σ = 2 I . We see that as Σ becomes larger, the Gaussian becomes more 'spread-out,' and as it becomes smaller, the distribution becomes more 'compressed.'

Let's look at some more examples.


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "type": "3D surface plot",
      "x_axis_label": "X",
      "x_axis_units": "units",
      "y_axis_label": "Y",
      "y_axis_units": "units",
      "z_axis_label": "Z",
      "z_axis_units": "units",
      "data_series_names": ["Series 1", "Series 2", "Series 3"],
      "specific_key_values": [
        {"Series 1": {"X": 0, "Y": 0, "Z": 0}},
        {"Series 2": {"X": 1, "Y": 1, "Z": 1}},
        {"Series 3": {"X": 2, "Y": 2, "Z": 2}}
      ],
      "inflection_points": [
        {"Series 1": {"X": 1, "Y": 1, "Z": 1}},
        {"Series 2": {"X": 2, "Y": 2, "Z": 2}},
        {"Series 3": {"X": 3, "Y": 3, "Z": 3}}
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "components": [
        {"name": "Component A", "position": {"X": 0, "Y": 0}},
        {"name": "Component B", "position": {"X": 1, "Y": 1}},
        {"name": "Component C", "position": {"X": 2, "Y": 2}}
      ],
      "connections": [
        {"source": "Component A", "target": "Component B"},
        {"source": "Component B", "target": "Component C"}
      ],
      "signal_flow_directions": [
        {"source": "Component A", "target": "Component B", "direction": "left_to_right"},
        {"source": "Component B", "target": "Component C", "direction": "left_to_right"}
      ]
    }
  ]
}
```


The figures above show Gaussians with mean 0, and with covariance matrices respectively

The leftmost figure shows the familiar standard normal distribution, and we see that as we increase the off-diagonal entry in Σ, the density becomes more

'compressed' towards the 45 ◦ line (given by x 1 = x 2 ). We can see this more clearly when we look at the contours of the same three densities:


> [Vision Analysis]: The image contains a series of plots that appear to represent the distribution of data points in a two-dimensional space, likely from a statistical or machine learning context. Each plot shows a contour plot with concentric circles representing different levels of density or probability. The plots are labeled as "examples by varying Σ," suggesting that the parameter Σ is being varied to observe its effect on the distribution.

### Mathematical Equations:
No specific mathematical equations are provided in the image.

### Charts and Graphs:
- **Plots**: There are six plots arranged in two rows of three.
- **Axes**: Each plot has two axes, labeled as \(x\) and \(y\).
- **Units**: The units for both axes are not explicitly stated, but they are likely in the same units as the data being plotted.
- **Data Series**: Each plot shows a contour plot with concentric circles representing different levels of density or probability. The color gradient within the circles indicates varying levels of density or probability.
- **Key Values or Inflection Points**: The plots show a clear central peak with decreasing density as you move away from the center. The exact values of the density at the center or the radius of the outermost circle are not provided.

### Diagrams and Schematics:
No diagrams or schematics are present in the image.

### Tables:
No tables are present in the image.

### Summary:
The image consists of six contour plots showing the effect of varying the parameter Σ on the distribution of data points in a two-dimensional space. The plots are labeled as "examples by varying Σ," indicating that the parameter Σ is being varied to observe its effect on the distribution. The plots show a central peak with decreasing density as you move away from the center, with the color gradient indicating varying levels of density or probability.


The plots above used, respectively,

From the leftmost and middle figures, we see that by decreasing the offdiagonal elements of the covariance matrix, the density now becomes 'compressed' again, but in the opposite direction. Lastly, as we vary the parameters, more generally the contours will form ellipses (the rightmost figure showing an example).

As our last set of examples, fixing Σ = I , by varying µ , we can also move the mean of the density around.


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$\mathcal{F}(x, y) = \exp\left(-\frac{x^2 + y^2}{2}\right)$"
  ],
  "charts_and_graphs": [
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    },
    {
      "type": "3D surface plot",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "z_axis_label": "f(x, y)",
      "units": "dimensionless",
      "scale": "linear",
      "data_series_names": ["f(x, y)"],
      "key_values": [
        {"x": 0, "y": 0, "z": 1},
        {"x": 1, "y": 0, "z": 0.367879},
        {"x": 0, "y": 1, "z": 0.367879},
        {"x": -1, "y": 0, "z": 0.367879}
      ]
    }
  ]
}
```


The figures above were generated using Σ = I , and respectively

4.1.2 The Gaussian discriminant analysis model

When we have a classification problem in which the input features x are continuous-valued random variables, we can then use the Gaussian Discriminant Analysis (GDA) model, which models p ( x | y ) using a multivariate normal distribution. The model is:

Writing out the distributions, this is:

Here, the parameters of our model are φ , Σ, µ 0 and µ 1 . (Note that while there're two different mean vectors µ 0 and µ 1 , this model is usually applied using only one covariance matrix Σ.) The log-likelihood of the data is given by

By maximizing glyph[lscript] with respect to the parameters, we find the maximum likelihood estimate of the parameters (see problem set 1) to be:

Pictorially, what the algorithm is doing can be seen in as follows:


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": {
    "chart_type": "contour plot",
    "x_axis_label": "X",
    "x_axis_units": "units not specified",
    "y_axis_label": "Y",
    "y_axis_units": "units not specified",
    "data_series_names": ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"],
    "specific_key_values": {
      "Series 1": {
        "inflection_point": [3, 3]
      },
      "Series 2": {
        "inflection_point": [2, 2]
      },
      "Series 3": {
        "inflection_point": [1, 1]
      },
      "Series 4": {
        "inflection_point": [0, 0]
      },
      "Series 5": {
        "inflection_point": [-1, -1]
      }
    }
  },
  "diagrams_and_schematics": {
    "diagram_type": "contour plot",
    "components": {
      "contour_lines": ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"],
      "data_points": ["Series 1", "Series 2", "Series 3", "Series 4", "Series 5"]
    },
    "connections": {
      "data_points_to_contour_lines": {
        "Series 1": [3, 3],
        "Series 2": [2, 2],
        "Series 3": [1, 1],
        "Series 4": [0, 0],
        "Series 5": [-1, -1]
      }
    },
    "signal_data_flow_directions": {
      "signal_flow": "from data points to contour lines"
    }
  }
}
```


Shown in the figure are the training set, as well as the contours of the two Gaussian distributions that have been fit to the data in each of the two classes. Note that the two Gaussians have contours that are the same shape and orientation, since they share a covariance matrix Σ, but they have different means µ 0 and µ 1 . Also shown in the figure is the straight line giving the decision boundary at which p ( y = 1 | x ) = 0 . 5. On one side of the boundary, we'll predict y = 1 to be the most likely outcome, and on the other side, we'll predict y = 0.

4.1.3 Discussion: GDA and logistic regression

The GDA model has an interesting relationship to logistic regression. If we view the quantity p ( y = 1 | x ; φ, µ 0 , µ 1 , Σ) as a function of x , we'll find that it can be expressed in the form

where θ is some appropriate function of φ, Σ , µ 0 , µ 1 . 1 This is exactly the form that logistic regression-a discriminative algorithm-used to model p ( y = 1 | x ).

When would we prefer one model over another? GDA and logistic regression will, in general, give different decision boundaries when trained on the same dataset. Which is better?

We just argued that if p ( x | y ) is multivariate gaussian (with shared Σ), then p ( y | x ) necessarily follows a logistic function. The converse, however, is not true; i.e., p ( y | x ) being a logistic function does not imply p ( x | y ) is multivariate gaussian. This shows that GDA makes stronger modeling assumptions about the data than does logistic regression. It turns out that when these modeling assumptions are correct, then GDA will find better fits to the data, and is a better model. Specifically, when p ( x | y ) is indeed gaussian (with shared Σ), then GDA is asymptotically efficient . Informally, this means that in the limit of very large training sets (large n ), there is no algorithm that is strictly better than GDA (in terms of, say, how accurately they estimate p ( y | x )). In particular, it can be shown that in this setting, GDA will be a better algorithm than logistic regression; and more generally, even for small training set sizes, we would generally expect GDA to better.

In contrast, by making significantly weaker assumptions, logistic regression is also more robust and less sensitive to incorrect modeling assumptions. There are many different sets of assumptions that would lead to p ( y | x ) taking the form of a logistic function. For example, if x | y = 0 ∼ Poisson( λ 0 ), and x | y = 1 ∼ Poisson( λ 1 ), then p ( y | x ) will be logistic. Logistic regression will also work well on Poisson data like this. But if we were to use GDA on such data-and fit Gaussian distributions to such non-Gaussian data-then the results will be less predictable, and GDA may (or may not) do well.

To summarize: GDA makes stronger modeling assumptions, and is more data efficient (i.e., requires less training data to learn 'well') when the modeling assumptions are correct or at least approximately correct. Logistic

1 This uses the convention of redefining the x ( i ) 's on the right-hand-side to be ( d +1)dimensional vectors by adding the extra coordinate x ( i ) 0 = 1; see problem set 1.

regression makes weaker assumptions, and is significantly more robust to deviations from modeling assumptions. Specifically, when the data is indeed non-Gaussian, then in the limit of large datasets, logistic regression will almost always do better than GDA. For this reason, in practice logistic regression is used more often than GDA. (Some related considerations about discriminative vs. generative models also apply for the Naive Bayes algorithm that we discuss next, but the Naive Bayes algorithm is still considered a very good, and is certainly also a very popular, classification algorithm.)

4.2 Naive bayes (Option Reading)

In GDA, the feature vectors x were continuous, real-valued vectors. Let's now talk about a different learning algorithm in which the x j 's are discretevalued.

For our motivating example, consider building an email spam filter using machine learning. Here, we wish to classify messages according to whether they are unsolicited commercial (spam) email, or non-spam email. After learning to do this, we can then have our mail reader automatically filter out the spam messages and perhaps place them in a separate mail folder. Classifying emails is one example of a broader set of problems called text classification .

Let's say we have a training set (a set of emails labeled as spam or nonspam). We'll begin our construction of our spam filter by specifying the features x j used to represent an email.

We will represent an email via a feature vector whose length is equal to the number of words in the dictionary. Specifically, if an email contains the j -th word of the dictionary, then we will set x j = 1; otherwise, we let x j = 0. For instance, the vector

is used to represent an email that contains the words 'a' and 'buy,' but not

'aardvark,' 'aardwolf' or 'zygmurgy.' 2 The set of words encoded into the feature vector is called the vocabulary , so the dimension of x is equal to the size of the vocabulary.

Having chosen our feature vector, we now want to build a generative model. So, we have to model p ( x | y ). But if we have, say, a vocabulary of 50000 words, then x ∈ { 0 , 1 } 50000 ( x is a 50000-dimensional vector of 0's and 1's), and if we were to model x explicitly with a multinomial distribution over the 2 50000 possible outcomes, then we'd end up with a (2 50000 -1)-dimensional parameter vector. This is clearly too many parameters.

To model p ( x | y ), we will therefore make a very strong assumption. We will assume that the x i 's are conditionally independent given y . This assumption is called the Naive Bayes (NB) assumption , and the resulting algorithm is called the Naive Bayes classifier . For instance, if y = 1 means spam email; 'buy' is word 2087 and 'price' is word 39831; then we are assuming that if I tell you y = 1 (that a particular piece of email is spam), then knowledge of x 2087 (knowledge of whether 'buy' appears in the message) will have no effect on your beliefs about the value of x 39831 (whether 'price' appears). More formally, this can be written p ( x 2087 | y ) = p ( x 2087 | y, x 39831 ). (Note that this is not the same as saying that x 2087 and x 39831 are independent, which would have been written ' p ( x 2087 ) = p ( x 2087 | x 39831 )'; rather, we are only assuming that x 2087 and x 39831 are conditionally independent given y .)

We now have:

The first equality simply follows from the usual properties of probabilities, and the second equality used the NB assumption. We note that even though

2 Actually, rather than looking through an English dictionary for the list of all English words, in practice it is more common to look through our training set and encode in our feature vector only the words that occur at least once there. Apart from reducing the number of words modeled and hence reducing our computational and space requirements, this also has the advantage of allowing us to model/include as a feature many words that may appear in your email (such as 'cs229') but that you won't find in a dictionary. Sometimes (as in the homework), we also exclude the very high frequency words (which will be words like 'the,' 'of,' 'and'; these high frequency, 'content free' words are called stop words ) since they occur in so many documents and do little to indicate whether an email is spam or non-spam.

the Naive Bayes assumption is an extremely strong assumptions, the resulting algorithm works well on many problems.

Our model is parameterized by φ j | y =1 = p ( x j = 1 | y = 1), φ j | y =0 = p ( x j = 1 | y = 0), and φ y = p ( y = 1). As usual, given a training set { ( x ( i ) , y ( i ) ); i = 1 , . . . , n } , we can write down the joint likelihood of the data:

Maximizing this with respect to φ y , φ j | y =0 and φ j | y =1 gives the maximum likelihood estimates:

In the equations above, the ' ∧ ' symbol means 'and.' The parameters have a very natural interpretation. For instance, φ j | y =1 is just the fraction of the spam ( y = 1) emails in which word j does appear.

Having fit all these parameters, to make a prediction on a new example with features x , we then simply calculate

and pick whichever class has the higher posterior probability.

Lastly, we note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the features x j are binary-valued, the generalization to where x j can take values in { 1 , 2 , . . . , k j } is straightforward. Here, we would simply model p ( x j | y ) as multinomial rather than as Bernoulli. Indeed, even if some original input attribute (say, the living area of a house, as in our earlier example) were continuous valued, it is quite common to discretize it-that is, turn it into a small set of discrete values-and apply Naive Bayes. For instance, if we use some feature x j to represent living area, we might discretize the continuous values as follows:

| Living area (sq. feet)   |   < 400 |   400-800 |   800-1200 |   1200-1600 |   > 1600 |
|--------------------------|---------|-----------|------------|-------------|----------|
| x i                      |       1 |         2 |          3 |           4 |        5 |

Thus, for a house with living area 890 square feet, we would set the value of the corresponding feature x j to 3. We can then apply the Naive Bayes algorithm, and model p ( x j | y ) with a multinomial distribution, as described previously. When the original, continuous-valued attributes are not wellmodeled by a multivariate normal distribution, discretizing the features and using Naive Bayes (instead of GDA) will often result in a better classifier.

4.2.1 Laplace smoothing

The Naive Bayes algorithm as we have described it will work fairly well for many problems, but there is a simple change that makes it work much better, especially for text classification. Let's briefly discuss a problem with the algorithm in its current form, and then talk about how we can fix it.

Consider spam/email classification, and let's suppose that, we are in the year of 20xx, after completing CS229 and having done excellent work on the project, you decide around May 20xx to submit work you did to the NeurIPS conference for publication. 3 Because you end up discussing the conference in your emails, you also start getting messages with the word 'neurips' in it. But this is your first NeurIPS paper, and until this time, you had not previously seen any emails containing the word 'neurips'; in particular 'neurips' did not ever appear in your training set of spam/non-spam emails. Assuming that 'neurips' was the 35000th word in the dictionary, your Naive Bayes spam filter therefore had picked its maximum likelihood estimates of the parameters φ 35000 | y to be

I.e., because it has never seen 'neurips' before in either spam or non-spam training examples, it thinks the probability of seeing it in either type of email is zero. Hence, when trying to decide if one of these messages containing

3 NeurIPS is one of the top machine learning conferences. The deadline for submitting a paper is typically in May-June.

'neurips' is spam, it calculates the class posterior probabilities, and obtains

This is because each of the terms ' ∏ d j =1 p ( x j | y )' includes a term p ( x 35000 | y ) = 0 that is multiplied into it. Hence, our algorithm obtains 0 / 0, and doesn't know how to make a prediction.

Stating the problem more broadly, it is statistically a bad idea to estimate the probability of some event to be zero just because you haven't seen it before in your finite training set. Take the problem of estimating the mean of a multinomial random variable z taking values in { 1 , . . . , k } . We can parameterize our multinomial with φ j = p ( z = j ). Given a set of n independent observations { z (1) , . . . , z ( n ) } , the maximum likelihood estimates are given by

As we saw previously, if we were to use these maximum likelihood estimates, then some of the φ j 's might end up as zero, which was a problem. To avoid this, we can use Laplace smoothing , which replaces the above estimate with glyph[negationslash]

Here, we've added 1 to the numerator, and k to the denominator. Note that ∑ k j =1 φ j = 1 still holds (check this yourself!), which is a desirable property since the φ j 's are estimates for probabilities that we know must sum to 1. Also, φ j = 0 for all values of j , solving our problem of probabilities being estimated as zero. Under certain (arguably quite strong) conditions, it can be shown that the Laplace smoothing actually gives the optimal estimator of the φ j 's.

Returning to our Naive Bayes classifier, with Laplace smoothing, we therefore obtain the following estimates of the parameters:

(In practice, it usually doesn't matter much whether we apply Laplace smoothing to φ y or not, since we will typically have a fair fraction each of spam and non-spam messages, so φ y will be a reasonable estimate of p ( y = 1) and will be quite far from 0 anyway.)

4.2.2 Event models for text classification

To close off our discussion of generative learning algorithms, let's talk about one more model that is specifically for text classification. While Naive Bayes as we've presented it will work well for many classification problems, for text classification, there is a related model that does even better.

In the specific context of text classification, Naive Bayes as presented uses the what's called the Bernoulli event model (or sometimes multi-variate Bernoulli event model ). In this model, we assumed that the way an email is generated is that first it is randomly determined (according to the class priors p ( y )) whether a spammer or non-spammer will send you your next message. Then, the person sending the email runs through the dictionary, deciding whether to include each word j in that email independently and according to the probabilities p ( x j = 1 | y ) = φ j | y . Thus, the probability of a message was given by p ( y ) ∏ d j =1 p ( x j | y ).

Here's a different model, called the Multinomial event model . To describe this model, we will use a different notation and set of features for representing emails. We let x j denote the identity of the j -th word in the email. Thus, x j is now an integer taking values in { 1 , . . . , | V |} , where | V | is the size of our vocabulary (dictionary). An email of d words is now represented by a vector ( x 1 , x 2 , . . . , x d ) of length d ; note that d can vary for different documents. For instance, if an email starts with 'A NeurIPS . . . ,' then x 1 = 1 ('a' is the first word in the dictionary), and x 2 = 35000 (if 'neurips' is the 35000th word in the dictionary).

In the multinomial event model, we assume that the way an email is generated is via a random process in which spam/non-spam is first determined (according to p ( y )) as before. Then, the sender of the email writes the email by first generating x 1 from some multinomial distribution over words ( p ( x 1 | y )). Next, the second word x 2 is chosen independently of x 1 but from the same multinomial distribution, and similarly for x 3 , x 4 , and so on, until all d words of the email have been generated. Thus, the overall probability of a message is given by p ( y ) ∏ d j =1 p ( x j | y ). Note that this formula looks like the one we had earlier for the probability of a message under the Bernoulli event model, but that the terms in the formula now mean very different things. In particular x j | y is now a multinomial, rather than a Bernoulli distribution.

The parameters for our new model are φ y = p ( y ) as before, φ k | y =1 = p ( x j = k | y = 1) (for any j ) and φ k | y =0 = p ( x j = k | y = 0). Note that we have assumed that p ( x j | y ) is the same for all values of j (i.e., that the distribution according to which a word is generated does not depend on its position j within the email).

If we are given a training set { ( x ( i ) , y ( i ) ); i = 1 , . . . , n } where x ( i ) = ( x ( i ) 1 , x ( i ) 2 , . . . , x ( i ) d i ) (here, d i is the number of words in the i -training example), the likelihood of the data is given by

Maximizing this yields the maximum likelihood estimates of the parameters:

If we were to apply Laplace smoothing (which is needed in practice for good performance) when estimating φ k | y =0 and φ k | y =1 , we add 1 to the numerators and | V | to the denominators, and obtain:

While not necessarily the very best classification algorithm, the Naive Bayes classifier often works surprisingly well. It is often also a very good 'first thing to try,' given its simplicity and ease of implementation.

Chapter 5

Kernel methods

5.1 Feature maps

Recall that in our discussion about linear regression, we considered the problem of predicting the price of a house (denoted by y ) from the living area of the house (denoted by x ), and we fit a linear function of x to the training data. What if the price y can be more accurately represented as a non-linear function of x ? In this case, we need a more expressive family of models than linear models.

We start by considering fitting cubic functions y = θ 3 x 3 + θ 2 x 2 + θ 1 x + θ 0 . It turns out that we can view the cubic function as a linear function over the a different set of feature variables (defined below). Concretely, let the function φ : R → R 4 be defined as

Let θ ∈ R 4 be the vector containing θ 0 , θ 1 , θ 2 , θ 3 as entries. Then we can rewrite the cubic function in x as:

Thus, a cubic function of the variable x can be viewed as a linear function over the variables φ ( x ). To distinguish between these two sets of variables, in the context of kernel methods, we will call the 'original' input value the input attributes of a problem (in this case, x , the living area). When the

original input is mapped to some new set of quantities φ ( x ), we will call those new quantities the features variables. (Unfortunately, different authors use different terms to describe these two things in different contexts.) We will call φ a feature map , which maps the attributes to the features.

5.2 LMS (least mean squares) with features

We will derive the gradient descent algorithm for fitting the model θ T φ ( x ). First recall that for ordinary least square problem where we were to fit θ T x , the batch gradient descent update is (see the first lecture note for its derivation):

Let φ : R d → R p be a feature map that maps attribute x (in R d ) to the features φ ( x ) in R p . (In the motivating example in the previous subsection, we have d = 1 and p = 4.) Now our goal is to fit the function θ T φ ( x ), with θ being a vector in R p instead of R d . We can replace all the occurrences of x ( i ) in the algorithm above by φ ( x ( i ) ) to obtain the new update:

Similarly, the corresponding stochastic gradient descent update rule is

5.3 LMS with the kernel trick

The gradient descent update, or stochastic gradient update above becomes computationally expensive when the features φ ( x ) is high-dimensional. For example, consider the direct extension of the feature map in equation (5.1) to high-dimensional input x : suppose x ∈ R d , and let φ ( x ) be the vector that

contains all the monomials of x with degree ≤ 3

The dimension of the features φ ( x ) is on the order of d 3 . 1 This is a prohibitively long vector for computational purpose - when d = 1000, each update requires at least computing and storing a 1000 3 = 10 9 dimensional vector, which is 10 6 times slower than the update rule for for ordinary least squares updates (5.2).

It may appear at first that such d 3 runtime per update and memory usage are inevitable, because the vector θ itself is of dimension p ≈ d 3 , and we may need to update every entry of θ and store it. However, we will introduce the kernel trick with which we will not need to store θ explicitly, and the runtime can be significantly improved.

For simplicity, we assume the initialize the value θ = 0, and we focus on the iterative update (5.3). The main observation is that at any time, θ can be represented as a linear combination of the vectors φ ( x (1) ) , . . . , φ ( x ( n ) ). Indeed, we can show this inductively as follows. At initialization, θ = 0 = ∑ n i =1 0 · φ ( x ( i ) ). Assume at some point, θ can be represented as

1 Here, for simplicity, we include all the monomials with repetitions (so that, e.g., x 1 x 2 x 3 and x 2 x 3 x 1 both appear in φ ( x )). Therefore, there are totally 1 + d + d 2 + d 3 entries in φ ( x ).

for some β 1 , . . . , β n ∈ R . Then we claim that in the next round, θ is still a linear combination of φ ( x (1) ) , . . . , φ ( x ( n ) ) because

You may realize that our general strategy is to implicitly represent the p -dimensional vector θ by a set of coefficients β 1 , . . . , β n . Towards doing this, we derive the update rule of the coefficients β 1 , . . . , β n . Using the equation above, we see that the new β i depends on the old one via

Here we still have the old θ on the RHS of the equation. Replacing θ by θ = ∑ n j =1 β j φ ( x ( j ) ) gives

We often rewrite φ ( x ( j ) ) T φ ( x ( i ) ) as 〈 φ ( x ( j ) ) , φ ( x ( i ) ) 〉 to emphasize that it's the inner product of the two feature vectors. Viewing β i 's as the new representation of θ , we have successfully translated the batch gradient descent algorithm into an algorithm that updates the value of β iteratively. It may appear that at every iteration, we still need to compute the values of 〈 φ ( x ( j ) ) , φ ( x ( i ) ) 〉 for all pairs of i, j , each of which may take roughly O ( p ) operation. However, two important properties come to rescue:

We can pre-compute the pairwise inner products 〈 φ ( x ( j ) ) , φ ( x ( i ) ) 〉 for all pairs of i, j before the loop starts.

For the feature map φ defined in (5.5) (or many other interesting feature maps), computing 〈 φ ( x ( j ) ) , φ ( x ( i ) ) 〉 can be efficient and does not

necessarily require computing φ ( x ( i ) ) explicitly. This is because:

Therefore, to compute 〈 φ ( x ) , φ ( z ) 〉 , we can first compute 〈 x, z 〉 with O ( d ) time and then take another constant number of operations to compute 1 + 〈 x, z 〉 + 〈 x, z 〉 2 + 〈 x, z 〉 3 .

As you will see, the inner products between the features 〈 φ ( x ) , φ ( z ) 〉 are essential here. We define the Kernel corresponding to the feature map φ as a function that maps X × X → R satisfying: 2

To wrap up the discussion, we write the down the final algorithm as follows:

Compute all the values K ( x ( i ) , x ( j ) ) glyph[defines] 〈 φ ( x ( i ) ) , φ ( x ( j ) ) 〉 using equation (5.9) for all i, j ∈ { 1 , . . . , n } . Set β := 0.

2. Loop:

Or in vector notation, letting K be the n × n matrix with K ij = K ( x ( i ) , x ( j ) ), we have

With the algorithm above, we can update the representation β of the vector θ efficiently with O ( n ) time per update. Finally, we need to show that

2 Recall that X is the space of the input x . In our running example, X = R d

the knowledge of the representation β suffices to compute the prediction θ T φ ( x ). Indeed, we have

You may realize that fundamentally all we need to know about the feature map φ ( · ) is encapsulated in the corresponding kernel function K ( · , · ). We will expand on this in the next section.

5.4 Properties of kernels

In the last subsection, we started with an explicitly defined feature map φ , which induces the kernel function K ( x, z ) glyph[defines] 〈 φ ( x ) , φ ( z ) 〉 . Then we saw that the kernel function is so intrinsic so that as long as the kernel function is defined, the whole training algorithm can be written entirely in the language of the kernel without referring to the feature map φ , so can the prediction of a test example x (equation (5.12).)

Therefore, it would be tempted to define other kernel function K ( · , · ) and run the algorithm (5.11). Note that the algorithm (5.11) does not need to explicitly access the feature map φ , and therefore we only need to ensure the existence of the feature map φ , but do not necessarily need to be able to explicitly write φ down.

What kinds of functions K ( · , · ) can correspond to some feature map φ ? In other words, can we tell if there is some feature mapping φ so that K ( x, z ) = φ ( x ) T φ ( z ) for all x , z ?

If we can answer this question by giving a precise characterization of valid kernel functions, then we can completely change the interface of selecting feature maps φ to the interface of selecting kernel function K . Concretely, we can pick a function K , verify that it satisfies the characterization (so that there exists a feature map φ that K corresponds to), and then we can run update rule (5.11). The benefit here is that we don't have to be able to compute φ or write it down analytically, and we only need to know its existence. We will answer this question at the end of this subsection after we go through several concrete examples of kernels.

Suppose x, z ∈ R d , and let's first consider the function K ( · , · ) defined as:

We can also write this as

Thus, we see that K ( x, z ) = 〈 φ ( x ) , φ ( z ) 〉 is the kernel function that corresponds to the the feature mapping φ given (shown here for the case of d = 3) by

.

Revisiting the computational efficiency perspective of kernel, note that whereas calculating the high-dimensional φ ( x ) requires O ( d 2 ) time, finding K ( x, z ) takes only O ( d ) time-linear in the dimension of the input attributes.

For another related example, also consider K ( · , · ) defined by

(Check this yourself.) This function K is a kernel function that corresponds

to the feature mapping (again shown for d = 3)

and the parameter c controls the relative weighting between the x i (first order) and the x i x j (second order) terms.

More broadly, the kernel K ( x, z ) = ( x T z + c ) k corresponds to a feature mapping to an ( d + k k ) feature space, corresponding of all monomials of the form x i 1 x i 2 . . . x i k that are up to order k . However, despite working in this O ( d k )-dimensional space, computing K ( x, z ) still takes only O ( d ) time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

Kernels as similarity metrics. Now, let's talk about a slightly different view of kernels. Intuitively, (and there are things wrong with this intuition, but nevermind), if φ ( x ) and φ ( z ) are close together, then we might expect K ( x, z ) = φ ( x ) T φ ( z ) to be large. Conversely, if φ ( x ) and φ ( z ) are far apartsay nearly orthogonal to each other-then K ( x, z ) = φ ( x ) T φ ( z ) will be small. So, we can think of K ( x, z ) as some measurement of how similar are φ ( x ) and φ ( z ), or of how similar are x and z .

Given this intuition, suppose that for some learning problem that you're working on, you've come up with some function K ( x, z ) that you think might be a reasonable measure of how similar x and z are. For instance, perhaps you chose

This is a reasonable measure of x and z 's similarity, and is close to 1 when x and z are close, and near 0 when x and z are far apart. Does there exist

a feature map φ such that the kernel K defined above satisfies K ( x, z ) = φ ( x ) T φ ( z )? In this particular example, the answer is yes. This kernel is called the Gaussian kernel , and corresponds to an infinite dimensional feature mapping φ . We will give a precise characterization about what properties a function K needs to satisfy so that it can be a valid kernel function that corresponds to some feature map φ .

Necessary conditions for valid kernels. Suppose for now that K is indeed a valid kernel corresponding to some feature mapping φ , and we will first see what properties it satisfies. Now, consider some finite set of n points (not necessarily the training set) { x (1) , . . . , x ( n ) } , and let a square, n -byn matrix K be defined so that its ( i, j )-entry is given by K ij = K ( x ( i ) , x ( j ) ). This matrix is called the kernel matrix . Note that we've overloaded the notation and used K to denote both the kernel function K ( x, z ) and the kernel matrix K , due to their obvious close relationship.

Now, if K is a valid kernel, then K ij = K ( x ( i ) , x ( j ) ) = φ ( x ( i ) ) T φ ( x ( j ) ) = φ ( x ( j ) ) T φ ( x ( i ) ) = K ( x ( j ) , x ( i ) ) = K ji , and hence K must be symmetric. Moreover, letting φ k ( x ) denote the k -th coordinate of the vector φ ( x ), we find that for any vector z , we have

The second-to-last step uses the fact that ∑ i,j a i a j = ( ∑ i a i ) 2 for a i = z i φ k ( x ( i ) ). Since z was arbitrary, this shows that K is positive semi-definite ( K ≥ 0).

Hence, we've shown that if K is a valid kernel (i.e., if it corresponds to some feature mapping φ ), then the corresponding kernel matrix K ∈ R n × n is symmetric positive semidefinite.

Sufficient conditions for valid kernels. More generally, the condition above turns out to be not only a necessary, but also a sufficient, condition for K to be a valid kernel (also called a Mercer kernel). The following result is due to Mercer. 3

Theorem (Mercer). Let K : R d × R d ↦→ R be given. Then for K to be a valid (Mercer) kernel, it is necessary and sufficient that for any { x (1) , . . . , x ( n ) } , ( n < ∞ ), the corresponding kernel matrix is symmetric positive semi-definite.

Given a function K , apart from trying to find a feature mapping φ that corresponds to it, this theorem therefore gives another way of testing if it is a valid kernel. You'll also have a chance to play with these ideas more in problem set 2.

In class, we also briefly talked about a couple of other examples of kernels. For instance, consider the digit recognition problem, in which given an image (16x16 pixels) of a handwritten digit (0-9), we have to figure out which digit it was. Using either a simple polynomial kernel K ( x, z ) = ( x T z ) k or the Gaussian kernel, SVMs were able to obtain extremely good performance on this problem. This was particularly surprising since the input attributes x were just 256-dimensional vectors of the image pixel intensity values, and the system had no prior knowledge about vision, or even about which pixels are adjacent to which other ones. Another example that we briefly talked about in lecture was that if the objects x that we are trying to classify are strings (say, x is a list of amino acids, which strung together form a protein), then it seems hard to construct a reasonable, 'small' set of features for most learning algorithms, especially if different strings have different lengths. However, consider letting φ ( x ) be a feature vector that counts the number of occurrences of each lengthk substring in x . If we're considering strings of English letters, then there are 26 k such strings. Hence, φ ( x ) is a 26 k dimensional vector; even for moderate values of k , this is probably too big for us to efficiently work with. (e.g., 26 4 ≈ 460000.) However, using (dynamic programming-ish) string matching algorithms, it is possible to efficiently compute K ( x, z ) = φ ( x ) T φ ( z ), so that we can now implicitly work in this 26 k -dimensional feature space, but without ever explicitly computing feature vectors in this space.

3 Many texts present Mercer's theorem in a slightly more complicated form involving L 2 functions, but when the input attributes take values in R d , the version given here is equivalent.

Application of kernel methods: We've seen the application of kernels to linear regression. In the next part, we will introduce the support vector machines to which kernels can be directly applied. dwell too much longer on it here. In fact, the idea of kernels has significantly broader applicability than linear regression and SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products 〈 x, z 〉 between input attribute vectors, then by replacing this with K ( x, z ) where K is a kernel, you can 'magically' allow your algorithm to work efficiently in the high dimensional feature space corresponding to K . For instance, this kernel trick can be applied with the perceptron to derive a kernel perceptron algorithm. Many of the algorithms that we'll see later in this class will also be amenable to this method, which has come to be known as the 'kernel trick.'

Chapter 6

Support vector machines

This set of notes presents the Support Vector Machine (SVM) learning algorithm. SVMs are among the best (and many believe are indeed the best) 'off-the-shelf' supervised learning algorithms. To tell the SVM story, we'll need to first talk about margins and the idea of separating data with a large 'gap.' Next, we'll talk about the optimal margin classifier, which will lead us into a digression on Lagrange duality. We'll also see kernels, which give a way to apply SVMs efficiently in very high dimensional (such as infinitedimensional) feature spaces, and finally, we'll close off the story with the SMO algorithm, which gives an efficient implementation of SVMs.

6.1 Margins: intuition

We'll start our story on SVMs by talking about margins. This section will give the intuitions about margins and about the 'confidence' of our predictions; these ideas will be made formal in Section 6.3.

Consider logistic regression, where the probability p ( y = 1 | x ; θ ) is modeled by h θ ( x ) = g ( θ T x ). We then predict '1' on an input x if and only if h θ ( x ) ≥ 0 . 5, or equivalently, if and only if θ T x ≥ 0. Consider a positive training example ( y = 1). The larger θ T x is, the larger also is h θ ( x ) = p ( y = 1 | x ; θ ), and thus also the higher our degree of 'confidence' that the label is 1. Thus, informally we can think of our prediction as being very confident that y = 1 if θ T x glyph[greatermuch] 0. Similarly, we think of logistic regression as confidently predicting y = 0, if θ T x glyph[lessmuch] 0. Given a training set, again informally it seems that we'd have found a good fit to the training data if we can find θ so that θ T x ( i ) glyph[greatermuch] 0 whenever y ( i ) = 1, and θ T x ( i ) glyph[lessmuch] 0 whenever y ( i ) = 0, since this would reflect a very confident (and correct) set of classifications for all the

training examples. This seems to be a nice goal to aim for, and we'll soon formalize this idea using the notion of functional margins.

For a different type of intuition, consider the following figure, in which x's represent positive training examples, o's denote negative training examples, a decision boundary (this is the line given by the equation θ T x = 0, and is also called the separating hyperplane ) is also shown, and three points have also been labeled A, B and C.


> [Vision Analysis]: ```json
{
  "chart": {
    "type": "scatter",
    "x_axis": {
      "label": "X-axis",
      "units": "units not specified",
      "scale": "linear",
      "data_series": [
        {
          "name": "X",
          "values": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20
          ]
        },
        {
          "name": "Y",
          "values": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20
          ]
        }
      ]
    },
    "y_axis": {
      "label": "Y-axis",
      "units": "units not specified",
      "scale": "linear",
      "data_series": [
        {
          "name": "X",
          "values": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20
          ]
        },
        {
          "name": "Y",
          "values": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20
          ]
        }
      ]
    },
    "data_points": [
      {
        "series": "X",
        "value": 0,
        "label": "A*"
      },
      {
        "series": "Y",
        "value": 0,
        "label": "B*"
      },
      {
        "series": "X",
        "value": 0,
        "label": "C*"
      }
    ]
  }
}
```


Notice that the point A is very far from the decision boundary. If we are asked to make a prediction for the value of y at A, it seems we should be quite confident that y = 1 there. Conversely, the point C is very close to the decision boundary, and while it's on the side of the decision boundary on which we would predict y = 1, it seems likely that just a small change to the decision boundary could easily have caused out prediction to be y = 0. Hence, we're much more confident about our prediction at A than at C. The point B lies in-between these two cases, and more broadly, we see that if a point is far from the separating hyperplane, then we may be significantly more confident in our predictions. Again, informally we think it would be nice if, given a training set, we manage to find a decision boundary that allows us to make all correct and confident (meaning far from the decision boundary) predictions on the training examples. We'll formalize this later using the notion of geometric margins.

6.2 Notation (option reading)

To make our discussion of SVMs easier, we'll first need to introduce a new notation for talking about classification. We will be considering a linear classifier for a binary classification problem with labels y and features x . From now, we'll use y ∈ {-1 , 1 } (instead of { 0 , 1 } ) to denote the class labels. Also, rather than parameterizing our linear classifier with the vector θ , we will use parameters w, b , and write our classifier as

Here, g ( z ) = 1 if z ≥ 0, and g ( z ) = -1 otherwise. This ' w, b ' notation allows us to explicitly treat the intercept term b separately from the other parameters. (We also drop the convention we had previously of letting x 0 = 1 be an extra coordinate in the input feature vector.) Thus, b takes the role of what was previously θ 0 , and w takes the role of [ θ 1 . . . θ d ] T .

Note also that, from our definition of g above, our classifier will directly predict either 1 or -1 (cf. the perceptron algorithm), without first going through the intermediate step of estimating p ( y = 1) (which is what logistic regression does).

6.3 Functional and geometric margins (option reading)

Let's formalize the notions of the functional and geometric margins. Given a training example ( x ( i ) , y ( i ) ), we define the functional margin of ( w, b ) with respect to the training example as

Note that if y ( i ) = 1, then for the functional margin to be large (i.e., for our prediction to be confident and correct), we need w T x ( i ) + b to be a large positive number. Conversely, if y ( i ) = -1, then for the functional margin to be large, we need w T x ( i ) + b to be a large negative number. Moreover, if y ( i ) ( w T x ( i ) + b ) > 0, then our prediction on this example is correct. (Check this yourself.) Hence, a large functional margin represents a confident and a correct prediction.

For a linear classifier with the choice of g given above (taking values in {-1 , 1 } ), there's one property of the functional margin that makes it not a very good measure of confidence, however. Given our choice of g , we note that

if we replace w with 2 w and b with 2 b , then since g ( w T x + b ) = g (2 w T x +2 b ), this would not change h w,b ( x ) at all. I.e., g , and hence also h w,b ( x ), depends only on the sign, but not on the magnitude, of w T x + b . However, replacing ( w, b ) with (2 w, 2 b ) also results in multiplying our functional margin by a factor of 2. Thus, it seems that by exploiting our freedom to scale w and b , we can make the functional margin arbitrarily large without really changing anything meaningful. Intuitively, it might therefore make sense to impose some sort of normalization condition such as that || w || 2 = 1; i.e., we might replace ( w, b ) with ( w/ || w || 2 , b/ || w || 2 ), and instead consider the functional margin of ( w/ || w || 2 , b/ || w || 2 ). We'll come back to this later.

Given a training set S = { ( x ( i ) , y ( i ) ); i = 1 , . . . , n } , we also define the function margin of ( w, b ) with respect to S as the smallest of the functional margins of the individual training examples. Denoted by ˆ γ , this can therefore be written:

Next, let's talk about geometric margins . Consider the picture below:


> [Vision Analysis]: ```json
{
  "diagram": {
    "description": "The diagram illustrates a classification boundary in a two-dimensional feature space. The boundary is a linear classifier, represented by a line. The data points are categorized into two classes: one represented by circles and the other by crosses. The line separates these two classes. The vector w, which is perpendicular to the boundary, is shown pointing towards the positive class.",
    "components": {
      "line": "The boundary line separating the two classes.",
      "vector_w": "The vector perpendicular to the boundary line, pointing towards the positive class.",
      "circles": "Data points representing one class.",
      "crosses": "Data points representing the other class."
    },
    "data_series": {
      "class1": "Data points represented by circles.",
      "class2": "Data points represented by crosses."
    }
  }
}
```


The decision boundary corresponding to ( w, b ) is shown, along with the vector w . Note that w is orthogonal (at 90 ◦ ) to the separating hyperplane. (You should convince yourself that this must be the case.) Consider the point at A, which represents the input x ( i ) of some training example with label y ( i ) = 1. Its distance to the decision boundary, γ ( i ) , is given by the line segment AB.

How can we find the value of γ ( i ) ? Well, w/ || w || is a unit-length vector pointing in the same direction as w . Since A represents x ( i ) , we therefore

find that the point B is given by x ( i ) -γ ( i ) · w/ || w || . But this point lies on the decision boundary, and all points x on the decision boundary satisfy the equation w T x + b = 0. Hence,

Solving for γ ( i ) yields

This was worked out for the case of a positive training example at A in the figure, where being on the 'positive' side of the decision boundary is good. More generally, we define the geometric margin of ( w, b ) with respect to a training example ( x ( i ) , y ( i ) ) to be

Note that if || w || = 1, then the functional margin equals the geometric margin-this thus gives us a way of relating these two different notions of margin. Also, the geometric margin is invariant to rescaling of the parameters; i.e., if we replace w with 2 w and b with 2 b , then the geometric margin does not change. This will in fact come in handy later. Specifically, because of this invariance to the scaling of the parameters, when trying to fit w and b to training data, we can impose an arbitrary scaling constraint on w without changing anything important; for instance, we can demand that || w || = 1, or | w 1 | = 5, or | w 1 + b | + | w 2 | = 2, and any of these can be satisfied simply by rescaling w and b .

Finally, given a training set S = { ( x ( i ) , y ( i ) ); i = 1 , . . . , n } , we also define the geometric margin of ( w, b ) with respect to S to be the smallest of the geometric margins on the individual training examples:

6.4 The optimal margin classifier (option reading)

Given a training set, it seems from our previous discussion that a natural desideratum is to try to find a decision boundary that maximizes the (geometric) margin, since this would reflect a very confident set of predictions

on the training set and a good 'fit' to the training data. Specifically, this will result in a classifier that separates the positive and the negative training examples with a 'gap' (geometric margin).

For now, we will assume that we are given a training set that is linearly separable; i.e., that it is possible to separate the positive and negative examples using some separating hyperplane. How will we find the one that achieves the maximum geometric margin? We can pose the following optimization problem:

I.e., we want to maximize γ , subject to each training example having functional margin at least γ . The || w || = 1 constraint moreover ensures that the functional margin equals to the geometric margin, so we are also guaranteed that all the geometric margins are at least γ . Thus, solving this problem will result in ( w, b ) with the largest possible geometric margin with respect to the training set.

If we could solve the optimization problem above, we'd be done. But the ' || w || = 1' constraint is a nasty (non-convex) one, and this problem certainly isn't in any format that we can plug into standard optimization software to solve. So, let's try transforming the problem into a nicer one. Consider:

Here, we're going to maximize ˆ γ/ || w || , subject to the functional margins all being at least ˆ γ . Since the geometric and functional margins are related by γ = ˆ γ/ || w | , this will give us the answer we want. Moreover, we've gotten rid of the constraint || w || = 1 that we didn't like. The downside is that we now have a nasty (again, non-convex) objective ˆ γ || w || function; and, we still don't have any off-the-shelf software that can solve this form of an optimization problem.

Let's keep going. Recall our earlier discussion that we can add an arbitrary scaling constraint on w and b without changing anything. This is the key idea we'll use now. We will introduce the scaling constraint that the functional margin of w, b with respect to the training set must be 1:

Since multiplying w and b by some constant results in the functional margin being multiplied by that same constant, this is indeed a scaling constraint, and can be satisfied by rescaling w, b . Plugging this into our problem above, and noting that maximizing ˆ γ/ || w || = 1 / || w || is the same thing as minimizing || w || 2 , we now have the following optimization problem:

We've now transformed the problem into a form that can be efficiently solved. The above is an optimization problem with a convex quadratic objective and only linear constraints. Its solution gives us the optimal margin classifier . This optimization problem can be solved using commercial quadratic programming (QP) code. 1

While we could call the problem solved here, what we will instead do is make a digression to talk about Lagrange duality. This will lead us to our optimization problem's dual form, which will play a key role in allowing us to use kernels to get optimal margin classifiers to work efficiently in very high dimensional spaces. The dual form will also allow us to derive an efficient algorithm for solving the above optimization problem that will typically do much better than generic QP software.

6.5 Lagrange duality (optional reading)

Let's temporarily put aside SVMs and maximum margin classifiers, and talk about solving constrained optimization problems.

Consider a problem of the following form:

Some of you may recall how the method of Lagrange multipliers can be used to solve it. (Don't worry if you haven't seen it before.) In this method, we define the Lagrangian to be

1 You may be familiar with linear programming, which solves optimization problems that have linear objectives and linear constraints. QP software is also widely available, which allows convex quadratic objectives and linear constraints.

Here, the β i 's are called the Lagrange multipliers . We would then find and set L 's partial derivatives to zero:

and solve for w and β .

In this section, we will generalize this to constrained optimization problems in which we may have inequality as well as equality constraints. Due to time constraints, we won't really be able to do the theory of Lagrange duality justice in this class, 2 but we will give the main ideas and results, which we will then apply to our optimal margin classifier's optimization problem.

Consider the following, which we'll call the primal optimization problem:

To solve it, we start by defining the generalized Lagrangian

Here, the α i 's and β i 's are the Lagrange multipliers. Consider the quantity

Here, the ' P ' subscript stands for 'primal.' Let some w be given. If w violates any of the primal constraints (i.e., if either g i ( w ) > 0 or h i ( w ) = 0 for some i ), then you should be able to verify that glyph[negationslash]

Conversely, if the constraints are indeed satisfied for a particular value of w , then θ P ( w ) = f ( w ). Hence,

2 Readers interested in learning more about this topic are encouraged to read, e.g., R. T. Rockarfeller (1970), Convex Analysis, Princeton University Press.

Thus, θ P takes the same value as the objective in our problem for all values of w that satisfies the primal constraints, and is positive infinity if the constraints are violated. Hence, if we consider the minimization problem

we see that it is the same problem (i.e., and has the same solutions as) our original, primal problem. For later use, we also define the optimal value of the objective to be p ∗ = min w θ P ( w ); we call this the value of the primal problem.

Now, let's look at a slightly different problem. We define

Here, the ' D ' subscript stands for 'dual.' Note also that whereas in the definition of θ P we were optimizing (maximizing) with respect to α, β , here we are minimizing with respect to w .

We can now pose the dual optimization problem:

This is exactly the same as our primal problem shown above, except that the order of the 'max' and the 'min' are now exchanged. We also define the optimal value of the dual problem's objective to be d ∗ = max α,β : α i ≥ 0 θ D ( w ).

How are the primal and the dual problems related? It can easily be shown that

(You should convince yourself of this; this follows from the 'max min' of a function always being less than or equal to the 'min max.') However, under certain conditions, we will have

so that we can solve the dual problem in lieu of the primal problem. Let's see what these conditions are.

Suppose f and the g i 's are convex, 3 and the h i 's are affine. 4 Suppose further that the constraints g i are (strictly) feasible; this means that there exists some w so that g i ( w ) < 0 for all i .

3 When f has a Hessian, then it is convex if and only if the Hessian is positive semidefinite. For instance, f ( w ) = w T w is convex; similarly, all linear (and affine) functions are also convex. (A function f can also be convex without being differentiable, but we won't need those more general definitions of convexity here.)

4 I.e., there exists a i , b i , so that h i ( w ) = a T i w + b i . 'Affine' means the same thing as linear, except that we also allow the extra intercept term b i .

Under our above assumptions, there must exist w ∗ , α ∗ , β ∗ so that w ∗ is the solution to the primal problem, α ∗ , β ∗ are the solution to the dual problem, and moreover p ∗ = d ∗ = L ( w ∗ , α ∗ , β ∗ ). Moreover, w ∗ , α ∗ and β ∗ satisfy the Karush-Kuhn-Tucker (KKT) conditions , which are as follows:

Moreover, if some w ∗ , α ∗ , β ∗ satisfy the KKT conditions, then it is also a solution to t he primal and dual problems.

We draw attention to Equation (6.5), which is called the KKT dual complementarity condition. Specifically, it implies that if α ∗ i > 0, then g i ( w ∗ ) = 0. (I.e., the ' g i ( w ) ≤ 0' constraint is active , meaning it holds with equality rather than with inequality.) Later on, this will be key for showing that the SVM has only a small number of 'support vectors'; the KKT dual complementarity condition will also give us our convergence test when we talk about the SMO algorithm.

6.6 Optimal margin classifiers: the dual form (option reading)

Note: The equivalence of optimization problem (6.8) and the optimization problem (6.12) , and the relationship between the primary and dual variables in equation (6.10) are the most important take home messages of this section.

Previously, we posed the following (primal) optimization problem for finding the optimal margin classifier:

We can write the constraints as

We have one such constraint for each training example. Note that from the KKT dual complementarity condition, we will have α i > 0 only for the training examples that have functional margin exactly equal to one (i.e., the ones corresponding to constraints that hold with equality, g i ( w ) = 0). Consider the figure below, in which a maximum margin separating hyperplane is shown by the solid line.


> [Vision Analysis]: ```json
{
  "diagram": {
    "description": "A scatter plot with two distinct sets of data points, each represented by a different symbol (crosses and circles). The data points are separated by a dashed line, which appears to be a decision boundary in a binary classification problem. The dashed line is flanked by two solid lines, which could represent upper and lower bounds or confidence intervals.",
    "axis_labels": {
      "x": "Feature 1",
      "y": "Feature 2"
    },
    "data_series": {
      "crosses": "Class 1",
      "circles": "Class 2"
    },
    "key_values": {
      "decision_boundary": "The dashed line separating the two classes",
      "upper_bound": "The solid line above the decision boundary",
      "lower_bound": "The solid line below the decision boundary"
    }
  }
}
```


The points with the smallest margins are exactly the ones closest to the decision boundary; here, these are the three points (one negative and two positive examples) that lie on the dashed lines parallel to the decision boundary. Thus, only three of the α i 's-namely, the ones corresponding to these three training examples-will be non-zero at the optimal solution to our optimization problem. These three points are called the support vectors in this problem. The fact that the number of support vectors can be much smaller than the size the training set will be useful later.

Let's move on. Looking ahead, as we develop the dual form of the problem, one key idea to watch out for is that we'll try to write our algorithm in terms of only the inner product 〈 x ( i ) , x ( j ) 〉 (think of this as ( x ( i ) ) T x ( j ) ) between points in the input feature space. The fact that we can express our algorithm in terms of these inner products will be key when we apply the kernel trick.

When we construct the Lagrangian for our optimization problem we have:

Note that there're only ' α i ' but no ' β i ' Lagrange multipliers, since the problem has only inequality constraints.

Let's find the dual form of the problem. To do so, we need to first minimize L ( w, b, α ) with respect to w and b (for fixed α ), to get θ D , which we'll do by setting the derivatives of L with respect to w and b to zero. We have:

This implies that

As for the derivative with respect to b , we obtain

If we take the definition of w in Equation (6.10) and plug that back into the Lagrangian (Equation 6.9), and simplify, we get

But from Equation (6.11), the last term must be zero, so we obtain

Recall that we got to the equation above by minimizing L with respect to w and b . Putting this together with the constraints α i ≥ 0 (that we always had) and the constraint (6.11), we obtain the following dual optimization problem:

You should also be able to verify that the conditions required for p ∗ = d ∗ and the KKT conditions (Equations 6.3-6.7) to hold are indeed satisfied in

our optimization problem. Hence, we can solve the dual in lieu of solving the primal problem. Specifically, in the dual problem above, we have a maximization problem in which the parameters are the α i 's. We'll talk later about the specific algorithm that we're going to use to solve the dual problem, but if we are indeed able to solve it (i.e., find the α 's that maximize W ( α ) subject to the constraints), then we can use Equation (6.10) to go back and find the optimal w 's as a function of the α 's. Having found w ∗ , by considering the primal problem, it is also straightforward to find the optimal value for the intercept term b as

(Check for yourself that this is correct.)

Before moving on, let's also take a more careful look at Equation (6.10), which gives the optimal value of w in terms of (the optimal value of) α . Suppose we've fit our model's parameters to a training set, and now wish to make a prediction at a new point input x . We would then calculate w T x + b , and predict y = 1 if and only if this quantity is bigger than zero. But using (6.10), this quantity can also be written:

Hence, if we've found the α i 's, in order to make a prediction, we have to calculate a quantity that depends only on the inner product between x and the points in the training set. Moreover, we saw earlier that the α i 's will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we really need to find only the inner products between x and the support vectors (of which there is often only a small number) in order calculate (6.15) and make our prediction.

By examining the dual form of the optimization problem, we gained significant insight into the structure of the problem, and were also able to write the entire algorithm in terms of only inner products between input feature vectors. In the next section, we will exploit this property to apply the kernels to our classification problem. The resulting algorithm, support vector machines , will be able to efficiently learn in very high dimensional spaces.

6.7 Regularization and the non-separable case (optional reading)

The derivation of the SVM as presented so far assumed that the data is linearly separable. While mapping data to a high dimensional feature space via φ does generally increase the likelihood that the data is separable, we can't guarantee that it always will be so. Also, in some cases it is not clear that finding a separating hyperplane is exactly what we'd want to do, since that might be susceptible to outliers. For instance, the left figure below shows an optimal margin classifier, and when a single outlier is added in the upper-left region (right figure), it causes the decision boundary to make a dramatic swing, and the resulting classifier has a much smaller margin.


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$y = mx + b$"
  ],
  "charts_and_graphs": {
    "chart_type": "Scatter plot",
    "x_axis_label": "X-axis",
    "x_axis_units": "Units not specified",
    "y_axis_label": "Y-axis",
    "y_axis_units": "Units not specified",
    "data_series_names": ["Data points", "Trend line"],
    "specific_key_values": {
      "Trend_line": {
        "slope": "Negative slope",
        "intercept": "Not explicitly given"
      }
    }
  },
  "diagrams_and_schematics": {
    "diagram_type": "Scatter plot with trend line",
    "components": {
      "data_points": {
        "description": "Scattered points representing data",
        "symbol": "Circle"
      },
      "trend_line": {
        "description": "Line representing the trend of the data",
        "symbol": "Dashed line"
      }
    },
    "connections": {
      "data_points_to_trend_line": {
        "description": "No direct connection, trend line is a best fit for the data points"
      }
    },
    "signal_data_flow_directions": {
      "data_flow": {
        "direction": "From left to right",
        "description": "Data points are plotted along the X-axis, and the trend line is plotted based on the Y-axis values"
      }
    }
  }
}
```


To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, we reformulate our optimization (using glyph[lscript] 1 regularization ) as follows:

Thus, examples are now permitted to have (functional) margin less than 1, and if an example has functional margin 1 -ξ i (with ξ > 0), we would pay a cost of the objective function being increased by Cξ i . The parameter C controls the relative weighting between the twin goals of making the || w || 2 small (which we saw earlier makes the margin large) and of ensuring that most examples have functional margin at least 1.

As before, we can form the Lagrangian:

Here, the α i 's and r i 's are our Lagrange multipliers (constrained to be ≥ 0). We won't go through the derivation of the dual again in detail, but after setting the derivatives with respect to w and b to zero as before, substituting them back in, and simplifying, we obtain the following dual form of the problem:

As before, we also have that w can be expressed in terms of the α i 's as given in Equation (6.10), so that after solving the dual problem, we can continue to use Equation (6.15) to make our predictions. Note that, somewhat surprisingly, in adding glyph[lscript] 1 regularization, the only change to the dual problem is that what was originally a constraint that 0 ≤ α i has now become 0 ≤ α i ≤ C . The calculation for b ∗ also has to be modified (Equation 6.13 is no longer valid); see the comments in the next section/Platt's paper.

Also, the KKT dual-complementarity conditions (which in the next section will be useful for testing for the convergence of the SMO algorithm) are:

Now, all that remains is to give an algorithm for actually solving the dual problem, which we will do in the next section.

6.8 The SMO algorithm (optional reading)

The SMO (sequential minimal optimization) algorithm, due to John Platt, gives an efficient way of solving the dual problem arising from the derivation

of the SVM. Partly to motivate the SMO algorithm, and partly because it's interesting in its own right, let's first take another digression to talk about the coordinate ascent algorithm.

6.8.1 Coordinate ascent

Consider trying to solve the unconstrained optimization problem

Here, we think of W as just some function of the parameters α i 's, and for now ignore any relationship between this problem and SVMs. We've already seen two optimization algorithms, gradient ascent and Newton's method. The new algorithm we're going to consider here is called coordinate ascent :

Loop until convergence: { For i = 1 , . . . , n , { α i := arg max ˆ α i W ( α 1 , . . . , α i -1 , ˆ α i , α i +1 , . . . , α n ). } }

Thus, in the innermost loop of this algorithm, we will hold all the variables except for some α i fixed, and reoptimize W with respect to just the parameter α i . In the version of this method presented here, the inner-loop reoptimizes the variables in order α 1 , α 2 , . . . , α n , α 1 , α 2 , . . . . (A more sophisticated version might choose other orderings; for instance, we may choose the next variable to update according to which one we expect to allow us to make the largest increase in W ( α ).)

When the function W happens to be of such a form that the 'arg max' in the inner loop can be performed efficiently, then coordinate ascent can be a fairly efficient algorithm. Here's a picture of coordinate ascent in action:


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": {
    "type": "contour plot",
    "axes": {
      "x": {
        "label": "X-axis",
        "units": "dimensionless",
        "scale": "linear",
        "min": -2,
        "max": 2
      },
      "y": {
        "label": "Y-axis",
        "units": "dimensionless",
        "scale": "linear",
        "min": -2,
        "max": 2
      }
    },
    "data_series": [
      {
        "color": "red",
        "label": "Series 1"
      },
      {
        "color": "green",
        "label": "Series 2"
      },
      {
        "color": "blue",
        "label": "Series 3"
      }
    ],
    "specific_key_values": {
      "center": {
        "x": 0,
        "y": 0
      },
      "corner": {
        "x": 2,
        "y": 2
      }
    }
  },
  "diagrams_and_schematics": {
    "components": [
      {
        "label": "Point",
        "position": {
          "x": 0,
          "y": 0
        }
      }
    ],
    "connections": [
      {
        "start": {
          "component": "Point",
          "position": {
            "x": 0,
            "y": 0
          }
        },
        "end": {
          "component": "Corner",
          "position": {
            "x": 2,
            "y": 2
          }
        },
        "direction": "arrow"
      }
    ],
    "signal_flow": {
      "direction": "from Point to Corner"
    }
  }
}
```


The ellipses in the figure are the contours of a quadratic function that we want to optimize. Coordinate ascent was initialized at (2 , -2), and also plotted in the figure is the path that it took on its way to the global maximum. Notice that on each step, coordinate ascent takes a step that's parallel to one of the axes, since only one variable is being optimized at a time.

6.8.2 SMO

We close off the discussion of SVMs by sketching the derivation of the SMO algorithm.

Here's the (dual) optimization problem that we want to solve:

Let's say we have set of α i 's that satisfy the constraints (6.20-6.21). Now, suppose we want to hold α 2 , . . . , α n fixed, and take a coordinate ascent step and reoptimize the objective with respect to α 1 . Can we make any progress? The answer is no, because the constraint (6.21) ensures that

Or, by multiplying both sides by y (1) , we equivalently have

(This step used the fact that y (1) ∈ {-1 , 1 } , and hence ( y (1) ) 2 = 1.) Hence, α 1 is exactly determined by the other α i 's, and if we were to hold α 2 , . . . , α n fixed, then we can't make any change to α 1 without violating the constraint (6.21) in the optimization problem.

Thus, if we want to update some subject of the α i 's, we must update at least two of them simultaneously in order to keep satisfying the constraints. This motivates the SMO algorithm, which simply does the following:

Repeat till convergence {

Select some pair α i and α j to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).

glyph[negationslash]

Reoptimize W ( α ) with respect to α i and α j , while holding all the other α k 's ( k = i, j ) fixed.

}

To test for convergence of this algorithm, we can check whether the KKT conditions (Equations 6.16-6.18) are satisfied to within some t ol . Here, t ol is the convergence tolerance parameter, and is typically set to around 0.01 to 0.001. (See the paper and pseudocode for details.)

The key reason that SMO is an efficient algorithm is that the update to α i , α j can be computed very efficiently. Let's now briefly sketch the main ideas for deriving the efficient update.

Let's say we currently have some setting of the α i 's that satisfy the constraints (6.20-6.21), and suppose we've decided to hold α 3 , . . . , α n fixed, and want to reoptimize W ( α 1 , α 2 , . . . , α n ) with respect to α 1 and α 2 (subject to the constraints). From (6.21), we require that

Since the right hand side is fixed (as we've fixed α 3 , . . . α n ), we can just let it be denoted by some constant ζ :

We can thus picture the constraints on α 1 and α 2 as follows:


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$a_{1}^{(1)} + a_{2}^{(2)} = a_{2}^{(2)}$"
  ],
  "charts_and_graphs": {
    "chart_type": "line",
    "x_axis_label": "C",
    "y_axis_label": "a_{1}",
    "units": "dimensionless",
    "scale": "linear",
    "data_series_names": ["a_{1}^{(1)} + a_{2}^{(2)}", "a_{2}^{(2)}"],
    "key_values": {
      "a_{1}^{(1)} + a_{2}^{(2)}": "a_{2}^{(2)}"
    }
  },
  "diagrams_and_schematics": {
    "diagram_type": "schematic",
    "components": {
      "C": "The endpoint of the x-axis",
      "a_{1}": "The y-axis label",
      "a_{1}^{(1)} + a_{2}^{(2)}": "The sum of two series",
      "a_{2}^{(2)}": "One of the series"
    },
    "connections": {
      "C": "The endpoint of the x-axis",
      "a_{1}": "The y-axis label",
      "a_{1}^{(1)} + a_{2}^{(2)}": "The sum of two series",
      "a_{2}^{(2)}": "One of the series"
    },
    "signal_data_flow_directions": {
      "C": "To the right",
      "a_{1}": "Upward",
      "a_{1}^{(1)} + a_{2}^{(2)}": "Upward",
      "a_{2}^{(2)}": "Upward"
    }
  }
}
```


From the constraints (6.20), we know that α 1 and α 2 must lie within the box [0 , C ] × [0 , C ] shown. Also plotted is the line α 1 y (1) + α 2 y (2) = ζ , on which we know α 1 and α 2 must lie. Note also that, from these constraints, we know L ≤ α 2 ≤ H ; otherwise, ( α 1 , α 2 ) can't simultaneously satisfy both the box and the straight line constraint. In this example, L = 0. But depending on what the line α 1 y (1) + α 2 y (2) = ζ looks like, this won't always necessarily be the case; but more generally, there will be some lower-bound L and some upper-bound H on the permissible values for α 2 that will ensure that α 1 , α 2 lie within the box [0 , C ] × [0 , C ].

Using Equation (6.22), we can also write α 1 as a function of α 2 :

(Check this derivation yourself; we again used the fact that y (1) ∈ {-1 , 1 } so that ( y (1) ) 2 = 1.) Hence, the objective W ( α ) can be written

Treating α 3 , . . . , α n as constants, you should be able to verify that this is just some quadratic function in α 2 . I.e., this can also be expressed in the form aα 2 2 + bα 2 + c for some appropriate a , b , and c . If we ignore the 'box' constraints (6.20) (or, equivalently, that L ≤ α 2 ≤ H ), then we can easily maximize this quadratic function by setting its derivative to zero and solving. We'll let α n ew,unclipped 2 denote the resulting value of α 2 . You should also be able to convince yourself that if we had instead wanted to maximize W with respect to α 2 but subject to the box constraint, then we can find the resulting value optimal simply by taking α n ew,unclipped 2 and 'clipping' it to lie in the

[ L, H ] interval, to get

Finally, having found the α n ew 2 , we can use Equation (6.22) to go back and find the optimal value of α n ew 1 .

There're a couple more details that are quite easy but that we'll leave you to read about yourself in Platt's paper: One is the choice of the heuristics used to select the next α i , α j to update; the other is how to update b as the SMO algorithm is run.

Part II Deep learning

Chapter 7

Deep learning

We now begin our study of deep learning. In this set of notes, we give an overview of neural networks, discuss vectorization and discuss training neural networks with backpropagation.

7.1 Supervised learning with non-linear models

In the supervised learning setting (predicting y from the input x ), suppose our model/hypothesis is h θ ( x ). In the past lectures, we have considered the cases when h θ ( x ) = θ glyph[latticetop] x (in linear regression) or h θ ( x ) = θ glyph[latticetop] φ ( x ) (where φ ( x ) is the feature map). A commonality of these two models is that they are linear in the parameters θ . Next we will consider learning general family of models that are non-linear in both the parameters θ and the inputs x . The most common non-linear models are neural networks, which we will define staring from the next section. For this section, it suffices to think h θ ( x ) as an abstract non-linear model. 1

Suppose { ( x ( i ) , y ( i ) ) } n i =1 are the training examples. We will define the nonlinear model and the loss/cost function for learning it.

Regression problems. For simplicity, we start with the case where the output is a real number, that is, y ( i ) ∈ R , and thus the model h θ also outputs a real number h θ ( x ) ∈ R . We define the least square cost function for the

1 If a concrete example is helpful, perhaps think about the model h θ ( x ) = θ 2 x 2 + θ 2 x 2 +

· · · + θ x

1 1 2 2 2 d 2 d in this subsection, even though it's not a neural network.

i -th example ( x ( i ) , y ( i ) ) as

and define the mean-square cost function for the dataset as

which is same as in linear regression except that we introduce a constant 1 /n in front of the cost function to be consistent with the convention. Note that multiplying the cost function with a scalar will not change the local minima or global minima of the cost function. Also note that the underlying parameterization for h θ ( x ) is different from the case of linear regression, even though the form of the cost function is the same mean-squared loss. Throughout the notes, we use the words 'loss' and 'cost' interchangeably.

Binary classification. Next we define the model and loss function for binary classification. Suppose the inputs x ∈ R d . Let ¯ h θ : R d → R be a parameterized model (the analog of θ glyph[latticetop] x in logistic linear regression). We call the output ¯ h θ ( x ) ∈ R the logit. Analogous to Section 2.1, we use the logistic function g ( · ) to turn the logit ¯ h θ ( x ) to a probability h θ ( x ) ∈ [0 , 1]:

We model the conditional distribution of y given x and θ by

Following the same derivation in Section 2.1 and using the derivation in Remark 2.1.1, the negative likelihood loss function is equal to:

As done in equation (7.2), the total loss function is also defined as the average of the loss function over individual training examples, J ( θ ) = 1 n ∑ n i =1 J ( i ) ( θ ) .

Multi-class classification. Following Section 2.3, we consider a classification problem where the response variable y can take on any one of k values, i.e. y ∈ { 1 , 2 , . . . , k } . Let ¯ h θ : R d → R k be a parameterized model. We call the outputs ¯ h θ ( x ) ∈ R k the logits. Each logit corresponds to the prediction for one of the k classes. Analogous to Section 2.3, we use the softmax function to turn the logits ¯ h θ ( x ) into a probability vector with non-negative entries that sum up to 1:

where ¯ h θ ( x ) s denotes the s -th coordinate of ¯ h θ ( x ).

Similarly to Section 2.3, the loss function for a single training example ( x ( i ) , y ( i ) ) is its negative log-likelihood:

Using the notations of Section 2.3, we can simply write in an abstract way:

The loss function is also defined as the average of the loss function of individual training examples, J ( θ ) = 1 n ∑ n i =1 J ( i ) ( θ ) .

We also note that the approach above can also be generated to any conditional probabilistic model where we have an exponential distribution for y , Exponential-family( y ; η ), where η = ¯ h θ ( x ) is a parameterized nonlinear function of x . However, the most widely used situations are the three cases discussed above.

Optimizers (SGD). Commonly, people use gradient descent (GD), stochastic gradient (SGD), or their variants to optimize the loss function J ( θ ). GD's update rule can be written as 2

where α > 0 is often referred to as the learning rate or step size. Next, we introduce a version of the SGD (Algorithm 1), which is lightly different from that in the first lecture notes.

2 Recall that, as defined in the previous lecture notes, we use the notation ' a := b ' to denote an operation (in a computer program) in which we set the value of a variable a to be equal to the value of b . In other words, this operation overwrites a with the value of b . In contrast, we will write ' a = b ' when we are asserting a statement of fact, that the value of a is equal to the value of b .

Algorithm 1 Stochastic Gradient Descent

1: Hyperparameter: learning rate α , number of total iteration n iter .

2: Initialize θ randomly.

3: for i = 1 to n iter do

4: Sample j uniformly from { 1 , . . . , n } , and update θ by

Oftentimes computing the gradient of B examples simultaneously for the parameter θ can be faster than computing B gradients separately due to hardware parallelization. Therefore, a mini-batch version of SGD is most commonly used in deep learning, as shown in Algorithm 2. There are also other variants of the SGD or mini-batch SGD with slightly different sampling schemes.

Algorithm 2 Mini-batch Stochastic Gradient Descent

1: Hyperparameters: learning rate α , batch size B , # iterations n iter .

2: Initialize θ randomly

3: for i = 1 to n iter do

4: Sample B examples j 1 , . . . , j B (without replacement) uniformly from { 1 , . . . , n } , and update θ by

With these generic algorithms, a typical deep learning model is learned with the following steps. 1. Define a neural network parametrization h θ ( x ), which we will introduce in Section 7.2, and 2. write the backpropagation algorithm to compute the gradient of the loss function J ( j ) ( θ ) efficiently, which will be covered in Section 7.4, and 3. run SGD or mini-batch SGD (or other gradient-based optimizers) with the loss function J ( θ ).

7.2 Neural networks

Neural networks refer to a broad type of non-linear models/parametrizations ¯ h θ ( x ) that involve combinations of matrix multiplications and other entrywise non-linear operations. To have a unified treatment for regression problem and classification problem, here we consider ¯ h θ ( x ) as the output of the neural network. For regression problem, the final prediction h θ ( x ) = ¯ h θ ( x ), and for classification problem, ¯ h θ ( x ) is the logits and the predicted probability will be h θ ( x ) = 1 / (1+exp( -¯ h θ ( x )) (see equation 7.3) for binary classification or h θ ( x ) = softmax( ¯ h θ ( x )) for multi-class classification (see equation 7.5).

We will start small and slowly build up a neural network, step by step.

A Neural Network with a Single Neuron. Recall the housing price prediction problem from before: given the size of the house, we want to predict the price. We will use it as a running example in this subsection.

Previously, we fit a straight line to the graph of size vs. housing price. Now, instead of fitting a straight line, we wish to prevent negative housing prices by setting the absolute minimum price as zero. This produces a 'kink' in the graph as shown in Figure 7.1. How do we represent such a function with a single kink as ¯ h θ ( x ) with unknown parameter? (After doing so, we can invoke the machinery in Section 7.1.)

We define a parameterized function ¯ h θ ( x ) with input x , parameterized by θ , which outputs the price of the house y . Formally, ¯ h θ : x → y . Perhaps one of the simplest parametrization would be

Here ¯ h θ ( x ) returns a single value: ( wx + b ) or zero, whichever is greater. In the context of neural networks, the function max { t, 0 } is called a ReLU (pronounced 'ray-lu'), or rectified linear unit, and often denoted by ReLU( t ) glyph[defines] max { t, 0 } .

Generally, a one-dimensional non-linear function that maps R to R such as ReLU is often referred to as an activation function . The model ¯ h θ ( x ) is said to have a single neuron partly because it has a single non-linear activation function. (We will discuss more about why a non-linear activation is called neuron.)

When the input x ∈ R d has multiple dimensions, a neural network with a single neuron can be written as


> [Vision Analysis]: ```json
{
  "chart": {
    "type": "scatter",
    "x_axis": {
      "label": "square feet",
      "units": "square feet",
      "scale": "linear",
      "min": 0,
      "max": 5000
    },
    "y_axis": {
      "label": "price in $1000s",
      "units": "$1000s",
      "scale": "linear",
      "min": 0,
      "max": 1000
    },
    "data_series": [
      {
        "name": "housing prices",
        "points": [
          [1000, 100],
          [2000, 200],
          [3000, 300],
          [4000, 400],
          [5000, 500]
        ]
      }
    ]
  }
}
```


Figure 7.1: Housing prices with a 'kink' in the graph.

The term b is often referred to as the 'bias', and the vector w is referred to as the weight vector. Such a neural network has 1 layer. (We will define what multiple layers mean in the sequel.)

Stacking Neurons. A more complex neural network may take the single neuron described above and 'stack' them together such that one neuron passes its output as input into the next neuron, resulting in a more complex function.

Let us now deepen the housing prediction example. In addition to the size of the house, suppose that you know the number of bedrooms, the zip code and the wealth of the neighborhood. Building neural networks is analogous to Lego bricks: you take individual bricks and stack them together to build complex structures. The same applies to neural networks: we take individual neurons and stack them together to create complex neural networks.

Given these features (size, number of bedrooms, zip code, and wealth), we might then decide that the price of the house depends on the maximum family size it can accommodate. Suppose the family size is a function of the size of the house and number of bedrooms (see Figure 7.2). The zip code may provide additional information such as how walkable the neighborhood is (i.e., can you walk to the grocery store or do you need to drive everywhere). Combining the zip code with the wealth of the neighborhood may predict the quality of the local elementary school. Given these three derived features (family size, walkable, school quality), we may conclude that the price of the

home ultimately depends on these three features.


> [Vision Analysis]: The image provided is a directed graph representing a causal model. Here is the technical description:

### Diagram Description:
- **Nodes**: Represent variables in the model.
- **Edges**: Represent causal relationships between variables.
- **Direction of Edges**: Arrows indicate the direction of causation. For example, an arrow from "Size" to "Fable" indicates that "Size" causes "Fable".
- **Labels on Edges**: Indicate the variable being influenced.

### Variables:
- **Size**: Size of the property.
- **# Bedrooms**: Number of bedrooms.
- **Price**: Price of the property.
- **Fable**: A variable representing the desirability or appeal of the property.
- **Walka**: Walkability score.
- **ZipCode**: Zip code.
- **Wealth**: Measure of wealth in the area.
- **Quality**: Quality of the property.
- **School**: School quality.
- **y**: The target variable, which is not explicitly labeled but is the output of the model.

### Causal Relationships:
1. **Size** → **Fable**
2. **Size** → **Price**
3. **# Bedrooms** → **Fable**
4. **# Bedrooms** → **Price**
5. **Walka** → **Fable**
6. **Walka** → **Price**
7. **ZipCode** → **Fable**
8. **ZipCode** → **Price**
9. **Wealth** → **Fable**
10. **Wealth** → **Price**
11. **Quality** → **Fable**
12. **Quality** → **Price**
13. **School** → **Fable**
14. **School** → **Price**

### Target Variable:
- **y**: The output of the model, which is influenced by all the variables listed above.

### Summary:
The model shows how various factors (Size, # Bedrooms, Walka, ZipCode, Wealth, Quality, School) influence the desirability (Fable) and price (Price) of a property. The target variable, \( y \), is the combined effect of these factors on the property's desirability and price.


Figure 7.2: Diagram of a small neural network for predicting housing prices.

Formally, the input to a neural network is a set of input features x 1 , x 2 , x 3 , x 4 . We denote the intermediate variables for 'family size', 'walkable', and 'school quality' by a 1 , a 2 , a 3 (these a i 's are often referred to as 'hidden units' or 'hidden neurons'). We represent each of the a i 's as a neural network with a single neuron with a subset of x 1 , . . . , x 4 as inputs. Then as in Figure 7.1, we will have the parameterization:

where ( θ 1 , · · · , θ 8 ) are parameters. Now we represent the final output ¯ h θ ( x ) as another linear function with a 1 , a 2 , a 3 as inputs, and we get 3

where θ contains all the parameters ( θ 1 , · · · , θ 12 ).

Now we represent the output as a quite complex function of x with parameters θ . Then you can use this parametrization ¯ h θ with the machinery of Section 7.1 to learn the parameters θ .

Inspiration from Biological Neural Networks. As the name suggests, artificial neural networks were inspired by biological neural networks. The hidden units a 1 , . . . , a m correspond to the neurons in a biological neural network, and the parameters θ i 's correspond to the synapses. However, it's unclear how similar the modern deep artificial neural networks are to the biological ones. For example, perhaps not many neuroscientists think biological

3 Typically, for multi-layer neural network, at the end, near the output, we don't apply ReLU, especially when the output is not necessarily a positive number.

neural networks could have 1000 layers, while some modern artificial neural networks do (we will elaborate more on the notion of layers.) Moreover, it's an open question whether human brains update their neural networks in a way similar to the way that computer scientists learn artificial neural networks (using backpropagation, which we will introduce in the next section.).

Two-layer Fully-Connected Neural Networks. We constructed the neural network in equation (7.13) using a significant amount of prior knowledge/belief about how the 'family size', 'walkable', and 'school quality' are determined by the inputs. We implicitly assumed that we know the family size is an important quantity to look at and that it can be determined by only the 'size' and '# bedrooms'. Such a prior knowledge might not be available for other applications. It would be more flexible and general to have a generic parameterization. A simple way would be to write the intermediate variable a 1 as a function of all x 1 , . . . , x 4 :

We still define ¯ h θ ( x ) using equation (7.13) with a 1 , a 2 , a 3 being defined as above. Thus we have a so-called fully-connected neural network because all the intermediate variables a i 's depend on all the inputs x i 's.

For full generality, a two-layer fully-connected neural network with m hidden units and d dimensional input x ∈ R d is defined as

Note that by default the vectors in R d are viewed as column vectors, and in particular a is a column vector with components a 1 , a 2 , ..., a m . The indices [1] and [2] are used to distinguish two sets of parameters: the w [1] j 's (each of which is a vector in R d ) and w [2] (which is a vector in R m ). We will have more of these later.

Vectorization. Before we introduce neural networks with more layers and more complex structures, we will simplify the expressions for neural networks

with more matrix and vector notations. Another important motivation of vectorization is the speed perspective in the implementation. In order to implement a neural network efficiently, one must be careful when using for loops. The most natural way to implement equation (7.15) in code is perhaps to use a for loop. In practice, the dimensionalities of the inputs and hidden units are high. As a result, code will run very slowly if you use for loops. Leveraging the parallelism in GPUs is/was crucial for the progress of deep learning.

This gave rise to vectorization . Instead of using for loops, vectorization takes advantage of matrix algebra and highly optimized numerical linear algebra packages (e.g., BLAS) to make neural network computations run quickly. Before the deep learning era, a for loop may have been sufficient on smaller datasets, but modern deep networks and state-of-the-art datasets will be infeasible to run with for loops.

We vectorize the two-layer fully-connected neural network as below. We define a weight matrix W [1] in R m × d as the concatenation of all the vectors w [1] j 's in the following way:

Now by the definition of matrix vector multiplication, we can write z = [ z 1 , . . . , z m ] glyph[latticetop] ∈ R m as

Or succinctly,

We remark again that a vector in R d in this notes, following the conventions previously established, is automatically viewed as a column vector, and can

also be viewed as a d × 1 dimensional matrix. (Note that this is different from numpy where a vector is viewed as a row vector in broadcasting.)

Computing the activations a ∈ R m from z ∈ R m involves an elementwise non-linear application of the ReLU function, which can be computed in parallel efficiently. Overloading ReLU for element-wise application of ReLU (meaning, for a vector t ∈ R d , ReLU( t ) is a vector such that ReLU( t ) i = ReLU( t i )), we have

Define W [2] = [ w [2] glyph[latticetop] ] ∈ R 1 × m similarly. Then, the model in equation (7.16) can be summarized as

Here θ consists of W [1] , W [2] (often referred to as the weight matrices) and b [1] , b [2] (referred to as the biases). The collection of W [1] , b [1] is referred to as the first layer, and W [2] , b [2] the second layer. The activation a is referred to as the hidden layer. A two-layer neural network is also called one-hidden-layer neural network.

Multi-layer fully-connected neural networks. With this succinct notations, we can stack more layers to get a deeper fully-connected neural network. Let r be the number of layers (weight matrices). Let W [1] , . . . , W [ r ] , b [1] , . . . , b [ r ] be the weight matrices and biases of all the layers. Then a multi-layer neural network can be written as

We note that the weight matrices and biases need to have compatible dimensions for the equations above to make sense. If a [ k ] has dimension m k , then the weight matrix W [ k ] should be of dimension m k × m k -1 , and the bias b [ k ] ∈ R m k . Moreover, W [1] ∈ R m 1 × d and W [ r ] ∈ R 1 × m r -1 .

The total number of neurons in the network is m 1 + · · · + m r , and the total number of parameters in this network is ( d +1) m 1 +( m 1 +1) m 2 + · · · + ( m r -1 +1) m r .

Sometimes for notational consistency we also write a [0] = x , and a [ r ] = h θ ( x ). Then we have simple recursion that

Note that this would have be true for k = r if there were an additional ReLU in equation (7.22), but often people like to make the last layer linear (aka without a ReLU) so that negative outputs are possible and it's easier to interpret the last layer as a linear model. (More on the interpretability at the 'connection to kernel method' paragraph of this section.)

Other activation functions. The activation function ReLU can be replaced by many other non-linear function σ ( · ) that maps R to R such as

The activation functions are plotted in Figure 7.3. Sigmoid and tanh are less and less used these days partly because their are bounded from both sides and the gradient of them vanishes as z goes to both positive and negative infinity (whereas all the other activation functions still have gradients as the input goes to positive infinity.) Softplus is not used very often either in practice and can be viewed as a smoothing of the ReLU so that it has a proper second order derivative. GELU and leaky ReLU are both variants of ReLU but they have some non-zero gradient even when the input is negative. GELU (or its slight variant) is used in NLP models such as BERT and GPT (which we will discuss in Chapter 14.)

Why do we not use the identity function for σ ( z ) ? That is, why not use σ ( z ) = z ? Assume for sake of argument that b [1] and b [2] are zeros.


> [Vision Analysis]: ```json
{
  "chart": {
    "title": "Activation Function Comparison",
    "xAxis": {
      "label": "Input (x)",
      "scale": "linear",
      "range": "-4 to 4"
    },
    "yAxis": {
      "label": "Output (y)",
      "scale": "linear",
      "range": "-1 to 2"
    },
    "series": [
      {
        "name": "ReLU",
        "lineStyle": "solid",
        "color": "red",
        "data": [
          [-4, 0],
          [-3, 0],
          [-2, 0],
          [-1, 0],
          [0, 0],
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4]
        ]
      },
      {
        "name": "sigmoid",
        "lineStyle": "dashed",
        "color": "orange",
        "data": [
          [-4, 0.01],
          [-3, 0.05],
          [-2, 0.2],
          [-1, 0.5],
          [0, 0.5],
          [1, 0.5],
          [2, 0.8],
          [3, 0.95],
          [4, 0.99]
        ]
      },
      {
        "name": "tanh",
        "lineStyle": "dotted",
        "color": "blue",
        "data": [
          [-4, -1],
          [-3, -0.9],
          [-2, -0.7],
          [-1, -0.5],
          [0, 0],
          [1, 0.5],
          [2, 0.7],
          [3, 0.9],
          [4, 1]
        ]
      },
      {
        "name": "leaky ReLU, γ = 0.3",
        "lineStyle": "dashdot",
        "color": "green",
        "data": [
          [-4, -1.2],
          [-3, -0.9],
          [-2, -0.6],
          [-1, -0.3],
          [0, 0],
          [1, 0.3],
          [2, 0.6],
          [3, 0.9],
          [4, 1.2]
        ]
      },
      {
        "name": "GELU",
        "lineStyle": "dashdot",
        "color": "purple",
        "data": [
          [-4, -0.07],
          [-3, -0.15],
          [-2, -0.25],
          [-1, -0.35],
          [0, 0],
          [1, 0.35],
          [2, 0.25],
          [3, 0.15],
          [4, 0.07]
        ]
      },
      {
        "name": "Softplus, β = 1",
        "lineStyle": "dashed",
        "color": "brown",
        "data": [
          [-4, 0.01],
          [-3, 0.05],
          [-2, 0.2],
          [-1, 0.5],
          [0, 0.5],
          [1, 0.5],
          [2, 0.8],
          [3, 0.95],
          [4, 0.99]
        ]
      }
    ]
  }
}
```


Figure 7.3: Activation functions in deep learning.

Suppose σ ( z ) = z , then for two-layer neural network, we have that

Notice how W [2] W [1] collapsed into ˜ W .

This is because applying a linear function to another linear function will result in a linear function over the original input (i.e., you can construct a ˜ W such that ˜ Wx = W [2] W [1] x ). This loses much of the representational power of the neural network as often times the output we are trying to predict has a non-linear relationship with the inputs. Without non-linear activation functions, the neural network will simply perform linear regression.

Connection to the Kernel Method. In the previous lectures, we covered the concept of feature maps. Recall that the main motivation for feature maps is to represent functions that are non-linear in the input x by θ glyph[latticetop] φ ( x ), where θ are the parameters and φ ( x ), the feature map, is a handcrafted function non-linear in the raw input x . The performance of the learning algorithms can significantly depends on the choice of the feature map φ ( x ). Oftentimes people use domain knowledge to design the feature map φ ( x ) that

suits the particular applications. The process of choosing the feature maps is often referred to as feature engineering .

We can view deep learning as a way to automatically learn the right feature map (sometimes also referred to as 'the representation') as follows. Suppose we denote by β the collection of the parameters in a fully-connected neural networks (equation (7.22)) except those in the last layer. Then we can abstract right a [ r -1] as a function of the input x and the parameters in β : a [ r -1] = φ β ( x ). Now we can write the model as

When β is fixed, then φ β ( · ) can viewed as a feature map, and therefore ¯ h θ ( x ) is just a linear model over the features φ β ( x ). However, we will train the neural networks, both the parameters in β and the parameters W [ r ] , b [ r ] are optimized, and therefore we are not learning a linear model in the feature space, but also learning a good feature map φ β ( · ) itself so that it's possible to predict accurately with a linear model on top of the feature map. Therefore, deep learning tends to depend less on the domain knowledge of the particular applications and requires often less feature engineering. The penultimate layer a [ r ] is often (informally) referred to as the learned features or representations in the context of deep learning.

In the example of house price prediction, a fully-connected neural network does not need us to specify the intermediate quantity such 'family size', and may automatically discover some useful features in the last penultimate layer (the activation a [ r -1] ), and use them to linearly predict the housing price. Often the feature map / representation obtained from one datasets (that is, the function φ β ( · ) can be also useful for other datasets, which indicates they contain essential information about the data. However, oftentimes, the neural network will discover complex features which are very useful for predicting the output but may be difficult for a human to understand or interpret. This is why some people refer to neural networks as a black box , as it can be difficult to understand the features it has discovered.

7.3 Modules in Modern Neural Networks

The multi-layer neural network introduced in equation (7.22) of Section 7.2 is often called multi-layer perceptron (MLP) these days. Modern neural networks used in practice are often much more complex and consist of multiple building blocks or multiple layers of building blocks. In this section, we will

introduce some of the other building blocks and discuss possible ways to combine them.

First, each matrix multiplication can be viewed as a building block. Consider a matrix multiplication operation with parameters ( W,b ) where W is the weight matrix and b is the bias vector, operating on an input z ,

Note that we implicitly assume all the dimensions are chosen to be compatible. We will also drop the subscripts under MM when they are clear in the context or just for convenience when they are not essential to the discussion.

Then, the MLP can be written as as a composition of multiple matrix multiplication modules and nonlinear activation modules (which can also be viewed as a building block):

Alternatively, when we drop the subscripts that indicate the parameters for convenience, we can write

Note that in this lecture notes, by default, all the modules have different sets of parameters, and the dimensions of the parameters are chosen such that the composition is meaningful.

Larger modules can be defined via smaller modules as well, e.g., one activation layer σ and a matrix multiplication layer MM are often combined and called a 'layer' in many papers. People often draw the architecture with the basic modules in a figure by indicating the dependency between these modules. E.g., see an illustration of an MLP in Figure 7.4, Left.

Residual connections. One of the very influential neural network architecture for vision application is ResNet, which uses the residual connections that are essentially used in almost all large-scale deep learning architectures these days. Using our notation above, a very much simplified residual block can be defined as

A much simplified ResNet is a composition of many residual blocks followed by a matrix multiplication,


> [Vision Analysis]: The image depicts a neural network architecture, specifically a ResNet-like structure with multiple layers and residual connections. Below is a detailed description:

### Mathematical Equations:
No explicit mathematical equations are provided in the image.

### Diagram and Schematics:
The diagram illustrates a neural network architecture with the following components and connections:

1. **Input Layer (X)**:
   - The input data is denoted as \( X \).

2. **Residual Block (Res)**:
   - Each Residual Block consists of:
     - A convolutional layer (Conv) with a kernel size of \( k \).
     - A batch normalization layer (BN).
     - An activation function (ReLU).
     - Another convolutional layer (Conv) with a kernel size of \( k \).
     - A batch normalization layer (BN).
     - An activation function (ReLU).
     - A residual connection (Res) that adds the input \( X \) to the output of the convolutional layers.

3. **MLP (Multi-Layer Perceptron)**:
   - The MLP block consists of:
     - Multiple fully connected layers (Layer 1, Layer 2, ..., Layer \( r-1 \)).
     - Each layer is followed by a ReLU activation function.
     - The final layer is denoted as \( \sigma \).

4. **MM (Multiplication Module)**:
   - The MM block consists of:
     - A multiplication operation between the output of the MLP and the output of the Residual Block.
     - The result is then passed through a ReLU activation function.

5. **Output Layer**:
   - The final output of the network is denoted as \( \sigma \).

### Signal/Data Flow:
- The input \( X \) is passed through the Residual Block.
- The output of the Residual Block is then passed through the MLP.
- The output of the MLP is multiplied with the output of the Residual Block.
- The result is passed through a ReLU activation function.
- The final output is denoted as \( \sigma \).

### Key Values and Inflection Points:
No specific key values or inflection points are explicitly mentioned in the image. The architecture is designed to handle data flow through residual connections and MLP layers.

### Units and Scales:
No units or scales are explicitly mentioned in the image. The architecture is designed to handle data flow through residual connections and MLP layers.

### Tables:
No tables are present in the image.


Figure 7.4: Illustrative Figures for Architecture. Left : An MLP with r layers. Right : A residual network.

We also draw the dependency of these modules in Figure 7.4, Right.

We note that the ResNet-S is still not the same as the ResNet architecture introduced in the seminal paper [He et al., 2016] because ResNet uses convolution layers instead of vanilla matrix multiplication, and adds batch normalization between convolutions and activations. We will introduce convolutional layers and some variants of batch normalization below. ResNet-S and layer normalization are part of the Transformer architecture that are widely used in modern large language models.

Layer normalization. Layer normalization, denoted by LN in this text, is a module that maps a vector z ∈ R m to a more normalized vector LN( z ) ∈ R m . It is oftentimes used after the nonlinear activations.

We first define a sub-module of the layer normalization, denoted by LN-S.

where ˆ µ = ∑ m i =1 z i m is the empirical mean of the vector z and ˆ σ = √ ∑ m i =1 ( z i -ˆ µ 2 ) m is the empirical standard deviation of the entries of z . 4 Intuitively, LN-S( z ) is a vector that is normalized to having empirical mean zero and empirical standard deviation 1.

4 Note that we divide by m instead of m -1 in the empirical standard deviation here because we are interested in making the output of LN-S( z ) have sum of squares equal to 1 (as opposed to estimating the standard deviation in statistics.)

Oftentimes zero mean and standard deviation 1 is not the most desired normalization scheme, and thus layernorm introduces to parameters learnable scalars β and γ as the desired mean and standard deviation, and use an affine transformation to turn the output of LN-S( z ) into a vector with mean β and standard deviation γ .

Here the first occurrence of β should be technically interpreted as a vector with all the entries being β . in We also note that ˆ µ and ˆ σ are also functions of z and shouldn't be treated as constants when computing the derivatives of layernorm. Moreover, β and γ are learnable parameters and thus layernorm is a parameterized module (as opposed to the activation layer which doesn't have any parameters.)

Scaling-invariant property. One important property of layer normalization is that it will make the model invariant to scaling of the parameters in the following sense. Suppose we consider composing LN with MM W,b and get a subnetwork LN(MM W,b ( z )). Then, we have that the output of this subnetwork does not change when the parameter in MM W,b is scaled:

To see this, we first know that LN-S( · ) is scale-invariant

Then we have

Due to this property, most of the modern DL architectures for large-scale computer vision and language applications have the following scale-invariant

property w.r.t all the weights that are not at the last layer. Suppose the network f has last layer' weights W last , and all the rest of the weights are denote by W . Then, we have f W last ,αW ( x ) = f W last ,W ( x ) for all α > 0. Here, the last layers weights are special because there are typically no layernorm or batchnorm after the last layer's weights.

Other normalization layers. There are several other normalization layers that aim to normalize the intermediate layers of the neural networks to a more fixed and controllable scaling, such as batch-normalization [ ? ], and group normalization [ ? ]. Batch normalization and group normalization are more often used in computer vision applications whereas layer norm is used more often in language applications.

Convolutional Layers. Convolutional Neural Networks are neural networks that consist of convolution layers (and many other modules), and are particularly useful for computer vision applications. For the simplicity of exposition, we focus on 1-D convolution in this text and only briefly mention 2-D convolution informally at the end of this subsection. (2-D convolution is more suitable for images which have two dimensions. 1-D convolution is also used in natural language processing.)

We start by introducing a simplified version of the 1-D convolution layer, denoted by Conv1D-S( · ) which is a type of matrix multiplication layer with a special structure. The parameters of Conv1D-S are a filter vector w ∈ R k where k is called the filter size (oftentimes k glyph[lessmuch] m ), and a bias scalar b . Oftentimes the filter is also called a kernel (but it does not have much to do with the kernel in kernel method.) For simplicity, we assume k = 2 glyph[lscript] +1 is an odd number. We first pad zeros to the input vector z in the sense that we let z 1 -glyph[lscript] = z 1 -glyph[lscript] +1 = .. = z 0 = 0 and z m +1 = z m +2 = .. = z m + glyph[lscript] = 0, and treat z as an ( m +2 glyph[lscript] )-dimension vector. Conv1D-S outputs a vector of dimension R m where each output dimension is a linear combination of subsets of z j 's with coefficients from w ,

Therefore, one can view Conv1D-S as a matrix multiplication with shared

parameters: Conv1D-S( z ) = Qz , where

Note that Q i,j = Q i -1 ,j -1 for all i, j ∈ { 2 , . . . , m } , and thus convoluation is a matrix multiplication with parameter sharing. We also note that computing the convolution only takes O ( km ) times but computing a generic matrix multiplication takes O ( m 2 ) time. Convolution has k parameters but generic matrix multiplication will have m 2 parameters. Thus convolution is supposed to be much more efficient than a generic matrix multiplication (as long as the additional structure imposed does not hurt the flexibility of the model to fit the data).

We also note that in practice there are many variants of the convolutional layers that we define here, e.g., there are other ways to pad zeros or sometimes the dimension of the output of the convolutional layers could be different from the input. We omit some of this subtleties here for simplicity.

The convolutional layers used in practice have also many 'channels' and the simplified version above corresponds to the 1-channel version. Formally, Conv1D takes in C vectors z 1 , . . . , z C ∈ R m as inputs, where C is referred to as the number of channels. In other words, the more general version, denoted by Conv1D, takes in a matrix as input, which is the concatenation of z 1 , . . . , z C and has dimension m × C . It can output C ′ vectors of dimension m , denoted by Conv1D( z ) 1 , . . . , Conv1D( z ) C ′ , where C ′ is referred to as the output channel, or equivalently a matrix of dimension m × C ′ . Each of the output is a sum of the simplified convolutions applied on various channels.

Note that each Conv1D-S i,j are modules with different parameters, and thus the total number of parameters is k (the number of parameters in a Conv1D-S) × CC ′ (the number of Conv1D-S i.j 's) = kCC ′ . In contrast, a generic linear mapping from R m × C and R m × C ′ has m 2 CC ′ parameters. The

parameters can also be represented as a three-dimensional tensor of dimension k × C × C ′ .

2-D convolution (brief). A 2-D convolution with one channel, denoted by Conv2D-S, is analogous to the Conv1D-S, but takes a 2-dimensional input z ∈ R m × m and applies a filter of size k × k , and outputs Conv2D-S( z ) ∈ R m × m . The full 2-D convolutional layer, denoted by Conv2D, takes in a sequence of matrices z 1 , . . . , z C ∈ R m × m , or equivalently a 3-D tensor z = ( z 1 , . . . , z C ) ∈ R m × m × C and outputs a sequence of matrices, Conv2D( z ) 1 , . . . , Conv2D( z ) C ′ ∈ R m × m , which can also be viewed as a 3D tensor in R m × m × C ′ . Each channel of the output is sum of the outcomes of applying Conv2D-S layers on all the input channels.

Because there are CC ′ number of Conv2D-S modules and each of the Conv2D-S module has k 2 parameters, the total number of parameters is CC ′ k 2 . The parameters can also be viewed as a 4D tensor of dimension C × C ′ × k × k .

7.4 Backpropagation

In this section, we introduce backpropgation or auto-differentiation, which computes the gradient of the loss ∇ J ( θ ) efficiently. We will start with an informal theorem that states that as long as a real-valued function f can be efficiently computed/evaluated by a differentiable network or circuit, then its gradient can be efficiently computed in a similar time. We will then show how to do this concretely for neural networks.

Because the formality of the general theorem is not the main focus here, we will introduce the terms with informal definitions. By a differentiable circuit or a differentiable network, we mean a composition of a sequence of differentiable arithmetic operations (additions, subtraction, multiplication, divisions, etc) and elementary differentiable functions (ReLU, exp, log, sin, cos, etc.). Let the size of the circuit be the total number of such operations and elementary functions. We assume that each of the operations and functions, and their derivatives or partial derivatives ecan be computed in O (1) time.

Theorem 7.4.1: [backpropagation or auto-differentiation, informally stated] Suppose a differentiable circuit of size N computes a real-valued function

f : R glyph[lscript] → R . Then, the gradient ∇ f can be computed in time O ( N ) , by a circuit of size O ( N ) . 5

We note that the loss function J ( j ) ( θ ) for j -th example can be indeed computed by a sequence of operations and functions involving additions, subtraction, multiplications, and non-linear activations. Thus the theorem suggests that we should be able to compute the ∇ J ( j ) ( θ ) in a similar time to that for computing J ( j ) ( θ ) itself. This does not only apply to the fullyconnected neural network introduced in the Section 7.2, but also many other types of neural networks that uses more advance modules.

We remark that auto-differentiation or backpropagation is already implemented in all the deep learning packages such as tensorflow and pytorch, and thus in practice, in most of cases a researcher does not need to write their backpropagation algorithms. However, understanding it is very helpful for gaining insights into the working of deep learning.

Organization of the rest of the section. In Section 7.4.1, we will start reviewing the basic Chain rule with a new perspective that is particularly useful for understanding backpropgation. Section 7.4.2 will introduce the general strategy for backpropagation. Section 7.4.2 will discuss how to compute the so-called backward function for basic modules used in neural networks, and Section 7.4.4 will put everything together to get a concrete backprop algorithm for MLPs.

7.4.1 Preliminaries on partial derivatives

Suppose a scalar variable J depend on some variables z (which could be a scalar, matrix, or high-order tensor), we write ∂J ∂z as the partial derivatives of J w.r.t to the variable z . We stress that the convention here is that ∂J ∂z has exactly the same dimension as z itself. For example, if z ∈ R m × n , then ∂J ∂z ∈ R m × n , and the ( i, j )-entry of ∂J ∂z is equal to ∂J ∂z ij .

Remark 7.4.2: When both J and z are not scalars, the partial derivatives of J w.r.t z becomes either a matrix or tensor and the notation becomes somewhat tricky. Besides the mathematical or notational challenges in dealing

5 We note if the output of the function f does not depend on some of the input coordinates, then we set by default the gradient w.r.t that coordinate to zero. Setting to zero does not count towards the total runtime here in our accounting scheme. This is why when N ≤ glyph[lscript] , we can compute the gradient in O ( N ) time, which might be potentially even less than glyph[lscript] .

with these partial derivatives of multi-variate functions, they are also expensive to compute and store, and thus rarely explicitly constructed empirically. The experience of authors of this note is that it's generally more productive to think only about derivatives of scalar function w.r.t to vector, matrices, or tensors. For example, in this note, we will not deal with derivatives of multi-variate functions.

Chain rule. We review the chain rule in calculus but with a perspective and notions that are more relevant for auto-differentiation.

Consider a scalar variable J which is obtained by the composition of f and g on some variable z ,

The same derivations below can be easily extend to the cases when z and u are matrices or tensors; but we insist that the final variable J is a scalar. (See also Remark 7.4.2.) Let u = ( u 1 , . . . , u n ) and let g ( z ) = ( g 1 ( z ) , · · · , g n ( z )) . Then, the standard chain rule gives us that

Alternatively, when z and u are both vectors, in a vectorized notation:

In other words, the backward function is always a linear map from ∂J ∂u to ∂J ∂z , though note that the mapping itself can depend on z in complex ways. The matrix on the RHS of (7.54) is actually the transpose of the Jacobian matrix of the function g . However, we do not discuss in-depth about Jacobian matrices to avoid complications. Part of the reason is that when z is a matrix (or tensor), to write an analog of equation (7.54), one has to either flatten z into a vector or introduce additional notations on tensor-matrix product. In this sense, equation (7.53) is more convenient and effective to use in all cases. For example, when z ∈ R r × s is a matrix, we can easily rewrite equation (7.53)

to

which will indeed be used in some of the derivations in Section 7.4.3.

Key interpretation of the chain rule. We can view the formula above (equation (7.53) or (7.54)) as a way to compute ∂J ∂z from ∂J ∂u . Consider the following abstract problem. Suppose J depends on z via u as defined in equation (7.52). However, suppose the function f is not given or the function f is complex, but we are given the value of ∂J ∂u . Then, the formula in equation (7.54) gives us a way to compute ∂J ∂z from ∂J ∂u .

Moreover, this formula only involves knowledge about g (more precisely ∂g j ∂z i ). We will repeatedly use this fact in situations where g is a building blocks of a complex network f .

Empirically, it's often useful to modularized the mapping in (7.53) or (7.54) into a black-box, and mathematically it's also convenient to define a notation for it. 6 We use B [ g, z ] to define the function that maps ∂J ∂u to ∂J ∂z , and write

We call B [ g, z ] the backward function for the module g . Note that when z is fixed, B [ g, z ] is merely a linear map from R n to R m . Using equation (7.53), we have

Or in vectorized notation, using (7.54), we have

6 e.g., the function is the .backward() method of the module in pytorch.

and therefore B [ g, z ] can be viewed as a matrix. However, in reality, z will be changing and thus the backward mapping has to be recomputed for different z 's while g is often fixed. Thus, empirically, the backward function B [ g, z ]( v ) is often viewed as a function which takes in z (=the input to g ) and v (=a vector that is supposed to be the gradient of some variable J w.r.t to the output of g ) as the inputs, and outputs a vector that is supposed to be the gradient of J w.r.t to z .

7.4.2 General strategy of backpropagation

We discuss the general strategy of auto-differentiation in this section to build a high-level understanding. Then, we will instantiate the approach to concrete neural networks. We take the viewpoint that neural networks are complex compositions of small building blocks such as MM, σ , Conv2D, LN, etc., defined in Section 7.3. Note that the losses (e.g., mean-squared loss, or the cross-entropy loss) can also be abstractly viewed as additional modules. Thus, we can abstractly write the loss function J (on a single example ( x, y )) as a composition of many modules: 7

For example, for a binary classification problem with a MLP ¯ h θ ( x ) (defined in equation (7.36) and (7.37)), the loss function has ber written in the form of equation (7.60) with M 1 = MM W [1] ,b [1] , M 2 = σ , M 3 = MM W [2] ,b [2] , . . . , and M k -1 = MM W [ r ] ,b [ r ] and M k = glyph[lscript] logistic .

We can see from this example that some modules involve parameters, and other modules might only involve a fixed set of operations. For generality, we assume that eachj M i involves a set of parameters θ [ i ] , though θ [ i ] could possibly be an empty set when M i is a fixed operation such as the nonlinear activations. We will discuss more on the granularity of the modularization, but so far we assume all the modules M i 's are simple enough.

We introduce the intermediate variables for the computation in (7.60).

7 Technically, we should write J = M k ( M k -1 ( · · · M 1 ( x )) , y ). However, y is treated as a constant for the purpose of computing the derivatives w.r.t to the parameters, and thus we can view it as part of M k for the sake of simplicity of notations.

Let

Backpropgation consists of two passes, the forward pass and backward pass. In the forward pass, the algorithm simply computes u [1] , . . . , u [ k ] from i = 1 , . . . , k , sequentially using the definition in (F), and save all the intermediate variables u [ i ] 's in the memory.

In the backward pass , we first compute the derivatives w.r.t to the intermediate variables, that is, ∂J ∂u [ k ] , . . . , ∂J ∂u [1] , sequentially in this backward order, and then compute the derivatives of the parameters ∂J ∂θ [ i ] from ∂J ∂u [ i ] and u [ i -1] . These two type of computations can be also interleaved with each other because ∂J ∂θ [ i ] only depends on ∂J ∂u [ i ] and u [ i -1] but not any ∂J ∂u [ k ] with k < i .

We first see why ∂J ∂u [ i -1] can be computed efficiently from ∂J ∂u [ i ] and u [ i -1] by invoking the discussion in Section 7.4.1 on the chain rule. We instantiate the discussion by setting u = u [ i ] and z = u [ i -1] , and f ( u ) = M k ( M k -1 ( · · · M i +1 ( u [ i ] ))), and g ( · ) = M i ( · ). Note that f is very complex but we don't need any concrete information about f . Then, the conclusive equation (7.56) corresponds to

More precisely, we can write, following equation (7.57)

Instantiating the chain rule with z = θ [ i ] and u = u [ i ] , we also have

See Figure 7.5 for an illustration of the algorithm.


> [Vision Analysis]: ```json
{
  "diagram": {
    "description": "A flowchart illustrating the forward and backward passes in a machine learning model.",
    "components": {
      "forward_pass": {
        "description": "The forward pass through the model.",
        "elements": [
          {
            "name": "J",
            "description": "The objective function or loss function."
          },
          {
            "name": "M_k",
            "description": "The k-th model parameter."
          },
          {
            "name": "u^{[k-1]}",
            "description": "The input or previous state."
          },
          {
            "name": "B[M_k, u^{[k-1]}]",
            "description": "The output of the model with the current parameter and input."
          },
          {
            "name": "∂J/∂θ",
            "description": "The gradient of the objective function with respect to the model parameters."
          },
          {
            "name": "∂J/∂θ^{[k]}",
            "description": "The gradient of the objective function with respect to the current model parameter."
          }
        ]
      },
      "backward_pass": {
        "description": "The backward pass for computing gradients.",
        "elements": [
          {
            "name": "∂J/∂θ",
            "description": "The gradient of the objective function with respect to the model parameters."
          },
          {
            "name": "∂J/∂θ^{[k]}",
            "description": "The gradient of the objective function with respect to the current model parameter."
          },
          {
            "name": "∂B[M_k, u^{[k-1]}]/∂θ^{[k]}",
            "description": "The partial derivative of the model output with respect to the current model parameter."
          },
          {
            "name": "u^{[k-1]}",
            "description": "The input or previous state."
          },
          {
            "name": "∂B[M_k, u^{[k-1]}]/∂u^{[k-1]}",
            "description": "The partial derivative of the model output with respect to the input."
          },
          {
            "name": "x",
            "description": "The input data."
          }
        ]
      }
    },
    "connections": {
      "forward_pass": {
        "connections": [
          {
            "source": "J",
            "target": "M_k"
          },
          {
            "source": "M_k",
            "target": "B[M_k, u^{[k-1]}]"
          },
          {
            "source": "B[M_k, u^{[k-1]}]",
            "target": "∂J/∂θ"
          },
          {
            "source": "∂J/∂θ",
            "target": "∂J/∂θ^{[k]}"
          }
        ]
      },
      "backward_pass": {
        "connections": [
          {
            "source": "∂J/∂θ",
            "target": "∂B[M_k, u^{[k-1]}]/∂θ^{[k]}"
          },
          {
            "source": "∂B[M_k, u^{[k-1]}]/∂θ^{[k]}",
            "target": "∂J/∂θ^{[k]}"
          },
          {
            "source": "∂J/∂θ^{[k]}",
            "target": "∂B[M_k, u^{[k-1]}]/∂u^{[k-1]}"
          },
          {
            "source": "∂B[M_k, u^{[k-1]}]/∂u^{[k-1]}",
            "target": "u^{[k-1]}"
          },
          {
            "source": "u^{[k-1]}",
            "target": "x"
          }
        ]
      }
    },
    "signal_flow": {
      "forward_pass": {
        "direction": "bottom_to_top",
        "description": "The signal flows from the objective function J to the model parameters M_k, then to the model output B[M_k, u^{[k-1]}], and finally to the gradient ∂J/∂θ."
      },
      "backward_pass": {
        "direction": "top_to_bottom",
        "description": "The signal flows from the gradient ∂J/∂θ to the model parameters M_k, then to the model output B[M_k, u^{[k-1]}], and finally to the input u^{[k-1]} and the input data x."
      }
    }
  }
}
```


Figure 7.5: Back-propagation.

Remark 7.4.3: [Computational efficiency and granularity of the modules] The main underlying purpose of treating a complex network as compositions of small modules is that small modules tend to have efficiently implementable backward function. In fact, the backward functions of all the atomic modules such as addition, multiplication and ReLU can be computed as efficiently as the the evaluation of these modules (up to multiplicative constant factor). Using this fact, we can prove Theorem 7.4.1 by viewing neural networks as compositions of many atomic operations, and invoking the backpropagation discussed above. However, in practice, it's oftentimes more convenient to modularize the networks using modules on the level of matrix multiplication, layernorm, etc. As we will see, naive implementation of these operations' backward functions also have the same runtime as the evaluation of these functions.

7.4.3 Backward functions for basic modules

Using the general strategy in Section 7.4.2, it suffices to compute the backward function for all modules M i 's used in the networks. We compute the backward function for the basic module MM, activations σ , and loss functions in this section.

Backward function for MM . Suppose MM W,b ( z ) = Wz + b is a matrix multiplication module where z ∈ R m and W ∈ R n × m . Then, using equation (7.59), we have for v ∈ R n

Using the fact that ∀ i ∈ [ m ] , j ∈ [ n ], ∂ ( Wz + b ) j ∂z i = ∂b j + ∑ m k =1 W jk z k ∂z i = W ji , we have

In the derivation above, we have treated MM as a function of z . If we treat MM as a function of W and b , then we can also compute the backward function for the parameter variables W and b . It's less convenient to use equation (7.59) because the variable W is a matrix and the matrix in (7.59) will be a 4-th order tensor that is challenging for us to mathematically write down. We use (7.58) instead:

In vectorized notation, we have

Using equation (7.59) for the variable b , we have,

glyph[negationslash]

The computational efficiency for computing the backward function is O ( mn ), the same as evaluating the result of matrix multiplication up to constant factor.

Here we used that ∂ ( Wz + b ) j ∂b i = 0 if i = j and ∂ ( Wz + b ) j ∂b i = 1 if i = j .

Backward function for the activations. Suppose M ( z ) = σ ( z ) where σ is an element-wise activation function and z ∈ R m . Then, using equation (7.59), we have

glyph[negationslash]

Here, we used the fact that ∂σ ( z j ) ∂z i = 0 when j = i , diag( λ 1 , . . . , λ m ) denotes the diagonal matrix with λ 1 , . . . , λ m on the diagonal, and glyph[circledot] denotes the element-wise product of two vectors with the same dimension, and σ ′ ( · ) is the element-wise application of the derivative of the activation function σ .

Regarding computation efficiency, we note that at the first sight, equation (7.67) appears to indicate the backward function takes O ( m 2 ) time, but equation (7.69) shows that it's implementable in O ( m ) time (which is the same as the time for evaluating of the function.) We are not supposed to be surprised by that the possibility of simplifying equation (7.67) to (7.69)-if we use smaller modules, that is, treating the vector-to-vector nonlinear activation as m scalar-to-scalar non-linear activation, then it's more obvious that the backward pass should have similar time to the forward pass.

Backward function for loss functions. When a module M takes in a vector z and outputs a scalar, by equation (7.59), the backward function takes in a scalar v and outputs a vector with entries ( B [ M,z ]( v )) i = ∂M ∂z i v . Therefore, in vectorized notation, B [ M,z ]( v ) = ∂M ∂z · v .

Recall that squared loss glyph[lscript] MSE ( z, y ) = 1 2 ( z -y ) 2 . Thus, B [ glyph[lscript] MSE , z ]( v ) = ∂ 1 2 ( z -y ) 2 ∂z · v = ( z -y ) · v .

For logistics loss, by equation (2.6), we have

For cross-entropy loss, by equation (2.17), we have

where φ = softmax( t ).

7.4.4 Back-propagation for MLPs

Given the backward functions for every module needed in evaluating the loss of an MLP, we follow the strategy in Section 7.4.2 to compute the gradient of the loss w.r.t to the hidden activations and the parameters.

We consider the an r -layer MLP with a logistic loss. The loss function can be computed via a sequence of operations (that is, the forward pass),

We apply the backward function sequentially in a backward order. First, we have that

Then, we iteratively compute ∂J ∂a [ i ] and ∂J ∂z [ i ] 's by repeatedly invoking the chain rule (equation (7.58)),

Numerically, we compute these quantities by repeatedly invoking equations (7.69) and (7.63) with different choices of variables.

We note that the intermediate values of a [ i ] and z [ i ] are used in the backpropagation (equation (7.74)), and therefore these values need to be stored in the memory after the forward pass.

Next, we compute the gradient of the parameters by invoking equations (7.65) and (7.66),

We also note that the block of computations in equations (7.75) can be interleaved with the block of computation in equations (7.74) because the ∂J ∂W [ i ] and ∂J ∂b [ i ] can be computed as soon as ∂J ∂z [ i ] is computed.

Putting all of these together, and explicitly invoking the equations (7.72), (7.74) and (7.75), we have the following algorithm (Algorithm 3).

Algorithm 3 Back-propagation for multi-layer neural networks.

1: Forward pass. Compute and store the values of a [ k ] 's, z [ k ] 's, and J using the equations (7.72).

2: Backward pass. Compute the gradient of loss J with respect to z [ r ] :

3: for k = r -1 to 0 do

4: Compute the gradient with respect to parameters W [ k +1] and b [ k +1] .

5: When k ≥ 1, compute the gradient with respect to z [ k ] and a [ k ] .

7.5 Vectorization over training examples

As we discussed in Section 7.1, in the implementation of neural networks, we will leverage the parallelism across the multiple examples. This means that we will need to write the forward pass (the evaluation of the outputs) of the neural network and the backward pass (backpropagation) for multiple

training examples in matrix notation.

The basic idea. The basic idea is simple. Suppose you have a training set with three examples x (1) , x (2) , x (3) . The first-layer activations for each example are as follows:

Note the difference between square brackets [ · ], which refer to the layer number, and parenthesis ( · ), which refer to the training example number. Intuitively, one would implement this using a for loop. It turns out, we can vectorize these operations as well. First, define:

Note that we are stacking training examples in columns and not rows. We can then combine this into a single unified formulation:

You may notice that we are attempting to add b [1] ∈ R 4 × 1 to W [1] X ∈ R 4 × 3 . Strictly following the rules of linear algebra, this is not allowed. In practice however, this addition is performed using broadcasting . We create an intermediate ˜ b [1] ∈ R 4 × 3 :

We can then perform the computation: Z [1] = W [1] X + ˜ b [1] . Often times, it is not necessary to explicitly construct ˜ b [1] . By inspecting the dimensions in (7.82), you can assume b [1] ∈ R 4 × 1 is correctly broadcast to W [1] X ∈ R 4 × 3 .

The matricization approach as above can easily generalize to multiple layers, with one subtlety though, as discussed below.

Complications/Subtlety in the Implementation. All the deep learning packages or implementations put the data points in the rows of a data matrix. (If the data point itself is a matrix or tensor, then the data are concentrated along the zero-th dimension.) However, most of the deep learning papers use a similar notation to these notes where the data points are treated as column vectors. 8 There is a simple conversion to deal with the mismatch: in the implementation, all the columns become row vectors, row vectors become column vectors, all the matrices are transposed, and the orders of the matrix multiplications are flipped. In the example above, using the row major convention, the data matrix is X ∈ R 3 × d , the first layer weight matrix has dimensionality d × m (instead of m × d as in the two layer neural net section), and the bias vector b [1] ∈ R 1 × m . The computation for the hidden activation becomes

8 The instructor suspects that this is mostly because in mathematics we naturally multiply a matrix to a vector on the left hand side.

Part III

Generalization and regularization

Chapter 8

Generalization

This chapter discusses tools to analyze and understand the generalization of machine learning models, i.e, their performances on unseen test examples. Recall that for supervised learning problems, given a training dataset { ( x ( i ) , y ( i ) ) } n i =1 , we typically learn a model h θ by minimizing a loss/cost function J ( θ ), which encourages h θ to fit the data. E.g., when the loss function is the least square loss (aka mean squared error), we have J ( θ ) = 1 n ∑ n i =1 ( y ( i ) -h θ ( x ( i ) )) 2 . This loss function for training purposes is oftentimes referred to as the training loss/error/cost.

However, minimizing the training loss is not our ultimate goal-it is merely our approach towards the goal of learning a predictive model. The most important evaluation metric of a model is the loss on unseen test examples, which is oftentimes referred to as the test error. Formally, we sample a test example ( x, y ) from the so-called test distribution D , and measure the model's error on it, by, e.g., the mean squared error, ( h θ ( x ) -y ) 2 . The expected loss/error over the randomness of the test example is called the test loss/error, 1

Note that the measurement of the error involves computing the expectation, and in practice, it can be approximated by the average error on many sampled test examples, which are referred to as the test dataset. Note that the key difference here between training and test datasets is that the test examples

1 In theoretical and statistical literature, we oftentimes call the uniform distribution over the training set { ( x ( i ) , y ( i ) ) } n i =1 , denoted by ̂ D , an empirical distribution, and call D the population distribution. Partly because of this, the training loss is also referred to as the empirical loss/risk/error, and the test loss is also referred to as the population loss/risk/error.

are unseen , in the sense that the training procedure has not used the test examples. In classical statistical learning settings, the training examples are also drawn from the same distribution as the test distribution D , but still the test examples are unseen by the learning procedure whereas the training examples are seen. 2

Because of this key difference between training and test datasets, even if they are both drawn from the same distribution D , the test error is not necessarily always close to the training error. 3 As a result, successfully minimizing the training error may not always lead to a small test error. We typically say the model overfits the data if the model predicts accurately on the training dataset but doesn't generalize well to other test examples, that is, if the training error is small but the test error is large. We say the model underfits the data if the training error is relatively large 4 (and in this case, typically the test error is also relatively large.)

This chapter studies how the test error is influenced by the learning procedure, especially the choice of model parameterizations. We will decompose the test error into 'bias' and 'variance' terms and study how each of them is affected by the choice of model parameterizations and their tradeoffs. Using the bias-variance tradeoff, we will discuss when overfitting and underfitting will occur and be avoided. We will also discuss the double descent phenomenon in Section 8.2 and some classical theoretical results in Section 8.3.

2 These days, researchers have increasingly been more interested in the setting with 'domain shift', that is, the training distribution and test distribution are different.

3 the difference between test error and training error is often referred to as the generalization gap. The term generalization error in some literature means the test error, and in some other literature means the generalization gap.

4 e.g., larger than the intrinsic noise level of the data in regression problems.

8.1 Bias-variance tradeoff

training dataset

x


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Training Data vs. Ground Truth h*",
      "x_axis_label": "x_train",
      "y_axis_label": "h*",
      "data_series": [
        {
          "name": "training data",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        },
        {
          "name": "ground truth h*",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        }
      ]
    },
    {
      "title": "Test Data vs. Ground Truth h*",
      "x_axis_label": "x_test",
      "y_axis_label": "h*",
      "data_series": [
        {
          "name": "test data",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        },
        {
          "name": "ground truth h*",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "Dataset",
      "components": [
        {
          "name": "training data",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        },
        {
          "name": "ground truth h*",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        }
      ]
    },
    {
      "title": "Test Data",
      "components": [
        {
          "name": "test data",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        },
        {
          "name": "ground truth h*",
          "values": [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]
          ]
        }
      ]
    }
  ]
}
```


test dataset

x

Figure 8.1: A running example of training and test dataset for this section.

As an illustrating example, we consider the following training dataset and test dataset, which are also shown in Figure 8.1. The training inputs x ( i ) 's are randomly chosen and the outputs y ( i ) are generated by y ( i ) = h glyph[star] ( x ( i ) ) + ξ ( i ) where the function h glyph[star] ( · ) is a quadratic function and is shown in Figure 8.1 as the solid line, and ξ ( i ) is the a observation noise assumed to be generated from ∼ N (0 , σ 2 ). A test example ( x, y ) also has the same input-output relationship y = h glyph[star] ( x ) + ξ where ξ ∼ N (0 , σ 2 ). It's impossible to predict the noise ξ , and therefore essentially our goal is to recover the function h glyph[star] ( · ).

We will consider the test error of learning various types of models. When talking about linear regression, we discussed the problem of whether to fit a 'simple' model such as the linear ' y = θ 0 + θ 1 x ,' or a more 'complex' model such as the polynomial ' y = θ 0 + θ 1 x + · · · θ 5 x 5 .'

We start with fitting a linear model, as shown in Figure 8.2. The best fitted linear model cannot predict y from x accurately even on the training dataset, let alone on the test dataset. This is because the true relationship between y and x is not linear-any linear model is far away from the true function h glyph[star] ( · ). As a result, the training error is large and this is a typical situation of underfitting .

y


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Training Data vs. Best Fit Linear Model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "training data",
          "type": "scatter",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit linear model",
          "type": "line",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    },
    {
      "title": "Test Data vs. Best Fit Linear Model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "test data",
          "type": "scatter",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit linear model",
          "type": "line",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "Training Data vs. Best Fit Linear Model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "training data",
          "type": "scatter",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit linear model",
          "type": "line",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    },
    {
      "title": "Test Data vs. Best Fit Linear Model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "test data",
          "type": "scatter",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit linear model",
          "type": "line",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    }
  ]
}
```


Figure 8.2: The best fit linear model has large training and test errors.

The issue cannot be mitigated with more training examples-even with a very large amount of, or even infinite training examples, the best fitted linear model is still inaccurate and fails to capture the structure of the data (Figure 8.3). Even if the noise is not present in the training data, the issue still occurs (Figure 8.4). Therefore, the fundamental bottleneck here is the linear model family's inability to capture the structure in the data-linear models cannot represent the true quadratic function h glyph[star] -, but not the lack of the data. Informally, we define the bias of a model to be the test error even if we were to fit it to a very (say, infinitely) large training dataset. Thus, in this case, the linear model suffers from large bias, and underfits (i.e., fails to capture structure exhibited by) the data.

y


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$$E = mc^2$$"
  ],
  "charts_and_graphs": [
    {
      "type": "scatter plot",
      "title": "fitting linear models on a large dataset",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_units": "units",
      "y_axis_units": "units",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series_names": ["ground truth $h^*$", "best fit linear model"],
      "data_points": [
        {"x": 0.2, "y": 0.2},
        {"x": 0.4, "y": 0.4},
        {"x": 0.6, "y": 0.6}
      ],
      "inflection_points": [
        {"x": 0.2, "y": 0.2},
        {"x": 0.4, "y": 0.4},
        {"x": 0.6, "y": 0.6}
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "type": "diagram",
      "title": "large dataset",
      "components": [
        {"label": "x-axis", "position": "left"},
        {"label": "y-axis", "position": "bottom"},
        {"label": "data points", "position": "center"},
        {"label": "ground truth $h^*$", "position": "top"},
        {"label": "best fit linear model", "position": "bottom"}
      ],
      "connections": [
        {"source": "x-axis", "target": "data points"},
        {"source": "y-axis", "target": "data points"},
        {"source": "ground truth $h^*$", "target": "data points"},
        {"source": "best fit linear model", "target": "data points"}
      ],
      "signal_flow_directions": [
        {"source": "x-axis", "target": "data points"},
        {"source": "y-axis", "target": "data points"},
        {"source": "ground truth $h^*$", "target": "data points"},
        {"source": "best fit linear model", "target": "data points"}
      ]
    }
  ],
  "tables": [
    {
      "table_structure": [
        ["x", "y"],
        [0.2, 0.2],
        [0.4, 0.4],
        [0.6, 0.6]
      ]
    }
  ]
}
```


x


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Fitting linear models on a noisy dataset",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "raw data",
          "values": [
            [0.0, 1.0],
            [0.1, 0.8],
            [0.2, 0.6],
            [0.3, 0.4],
            [0.4, 0.2],
            [0.5, 0.0],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.9, 0.8],
            [1.0, 1.0]
          ]
        },
        {
          "name": "ground truth h*",
          "values": [
            [0.0, 1.0],
            [0.1, 0.8],
            [0.2, 0.6],
            [0.3, 0.4],
            [0.4, 0.2],
            [0.5, 0.0],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.9, 0.8],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit linear model",
          "values": [
            [0.0, 1.0],
            [0.1, 0.8],
            [0.2, 0.6],
            [0.3, 0.4],
            [0.4, 0.2],
            [0.5, 0.0],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.9, 0.8],
            [1.0, 1.0]
          ]
        }
      ]
    },
    {
      "title": "Noisy dataset",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "raw data",
          "values": [
            [0.0, 1.0],
            [0.1, 0.8],
            [0.2, 0.6],
            [0.3, 0.4],
            [0.4, 0.2],
            [0.5, 0.0],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.9, 0.8],
            [1.0, 1.0]
          ]
        }
      ]
    }
  ]
}
```


Figure 8.3: The best fit linear model on a much larger dataset still has a large training error.

x

Figure 8.4: The best fit linear model on a noiseless dataset also has a large training/test error.

Next, we fit a 5th-degree polynomial to the data. Figure 8.5 shows that it fails to learn a good model either. However, the failure pattern is different from the linear model case. Specifically, even though the learnt 5th-degree

y

1.5

1.5

polynomial did a very good job predicting y ( i ) 's from x ( i ) 's for training examples, it does not work well on test examples (Figure 8.5). In other words, the model learnt from the training set does not generalize well to other test examples-the test error is high. Contrary to the behavior of linear models, the bias of the 5-th degree polynomials is small-if we were to fit a 5-th degree polynomial to an extremely large dataset, the resulting model would be close to a quadratic function and be accurate (Figure 8.6). This is because the family of 5-th degree polynomials contains all the quadratic functions (setting θ 5 = θ 4 = θ 3 = 0 results in a quadratic function), and, therefore, 5-th degree polynomials are in principle capable of capturing the structure of the data.

1.5

1.5


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Training data vs. best fit 5th degree model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "training data",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "best fit 5th degree model",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        }
      ]
    },
    {
      "title": "Test data vs. ground truth h vs. best fit 5th degree model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "test data",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "ground truth h",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "best fit 5th degree model",
          "data_points": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "Model structure",
      "components": [
        {
          "name": "Input",
          "position": [0.0, 0.0]
        },
        {
          "name": "Layer 1",
          "position": [0.1, 0.0]
        },
        {
          "name": "Layer 2",
          "position": [0.2, 0.0]
        },
        {
          "name": "Layer 3",
          "position": [0.3, 0.0]
        },
        {
          "name": "Output",
          "position": [0.4, 0.0]
        }
      ],
      "connections": [
        {
          "source": "Input",
          "target": "Layer 1",
          "direction": "forward"
        },
        {
          "source": "Layer 1",
          "target": "Layer 2",
          "direction": "forward"
        },
        {
          "source": "Layer 2",
          "target": "Layer 3",
          "direction": "forward"
        },
        {
          "source": "Layer 3",
          "target": "Output",
          "direction": "forward"
        }
      ]
    }
  ]
}
```


x

x

Figure 8.5: Best fit 5-th degree polynomial has zero training error, but still has a large test error and does not recover the the ground truth. This is a classic situation of overfitting.

y


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "fitting 5th degree model or large dataset",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "training data",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "fit 5th degree model",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "ground truth n'",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "large dataset",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "data_series": [
        {
          "name": "training data",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "fit 5th degree model",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        },
        {
          "name": "ground truth n'",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6]
          ]
        }
      ]
    }
  ]
}
```


x

Figure 8.6: The best fit 5-th degree polynomial on a huge dataset nearly recovers the ground-truth-suggesting that the culprit in Figure 8.5 is the variance (or lack of data) but not bias.

The failure of fitting 5-th degree polynomials can be captured by another

component of the test error, called variance of a model fitting procedure. Specifically, when fitting a 5-th degree polynomial as in Figure 8.7, there is a large risk that we're fitting patterns in the data that happened to be present in our small, finite training set, but that do not reflect the wider pattern of the relationship between x and y . These 'spurious' patterns in the training set are (mostly) due to the observation noise ξ ( i ) , and fitting these spurious patters results in a model with large test error. In this case, we say the model has a large variance.


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "fitting 5-th degree model on different datasets",
      "x_axis_label": "model order",
      "y_axis_label": "error",
      "data_series": [
        {
          "name": "training data",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit 5-th degree model",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    },
    {
      "title": "fitting 5-th degree model on different datasets",
      "x_axis_label": "model order",
      "y_axis_label": "error",
      "data_series": [
        {
          "name": "training data",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        },
        {
          "name": "best fit 5-th degree model",
          "values": [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0]
          ]
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "diagram",
      "components": [
        {
          "label": "training data",
          "position": [0.0, 0.0]
        },
        {
          "label": "best fit 5-th degree model",
          "position": [0.1, 0.1]
        }
      ],
      "connections": [
        {
          "source": "training data",
          "target": "best fit 5-th degree model",
          "direction": "right"
        }
      ]
    }
  ],
  "tables": []
}
```


Figure 8.7: The best fit 5-th degree models on three different datasets generated from the same distribution behave quite differently, suggesting the existence of a large variance.

The variance can be intuitively (and mathematically, as shown in Section 8.1.1) characterized by the amount of variations across models learnt on multiple different training datasets (drawn from the same underlying distribution). The 'spurious patterns' are specific to the randomness of the noise (and inputs) in a particular dataset, and thus are different across multiple training datasets. Therefore, overfitting to the 'spurious patterns' of multiple datasets should result in very different models. Indeed, as shown in Figure 8.7, the models learned on the three different training datasets are quite different, overfitting to the 'spurious patterns' of each datasets.

Often, there is a tradeoff between bias and variance. If our model is too 'simple' and has very few parameters, then it may have large bias (but small variance), and it typically may suffer from underfittng. If it is too 'complex' and has very many parameters, then it may suffer from large variance (but have smaller bias), and thus overfitting. See Figure 8.8 for a typical tradeoff between bias and variance.


> [Vision Analysis]: ```json
{
  "chart": {
    "title": "Optimal Tradeoff",
    "x_axis": {
      "label": "Model Complexity",
      "scale": "Logarithmic",
      "units": "None"
    },
    "y_axis": {
      "label": "Error",
      "scale": "Logarithmic",
      "units": "None"
    },
    "data_series": [
      {
        "name": "Bias^2",
        "line_color": "Red",
        "line_style": "Dashed",
        "data_points": [
          {"x": 1, "y": 10},
          {"x": 2, "y": 5},
          {"x": 3, "y": 2},
          {"x": 4, "y": 1}
        ]
      },
      {
        "name": "Variance",
        "line_color": "Blue",
        "line_style": "Solid",
        "data_points": [
          {"x": 1, "y": 10},
          {"x": 2, "y": 5},
          {"x": 3, "y": 2},
          {"x": 4, "y": 1}
        ]
      },
      {
        "name": "Optimal Tradeoff",
        "line_color": "Green",
        "line_style": "Dotted",
        "data_points": [
          {"x": 1, "y": 10},
          {"x": 2, "y": 5},
          {"x": 3, "y": 2},
          {"x": 4, "y": 1}
        ]
      }
    ],
    "legend": ["Bias^2", "Variance", "Optimal Tradeoff"]
  }
}
```


Figure 8.8: An illustration of the typical bias-variance tradeoff.

As we will see formally in Section 8.1.1, the test error can be decomposed as a summation of bias and variance. This means that the test error will have a convex curve as the model complexity increases, and in practice we should tune the model complexity to achieve the best tradeoff. For instance, in the example above, fitting a quadratic function does better than either of the extremes of a first or a 5-th degree polynomial, as shown in Figure 8.9.

y


> [Vision Analysis]: ```json
{
  "mathematical_equations": [
    "$E = mc^2$"
  ],
  "charts_and_graphs": [
    {
      "title": "Training Data and Best Fit Quadratic",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "training data",
          "color": "blue",
          "marker": "circle"
        },
        {
          "name": "best fit quadratic",
          "color": "red",
          "line_style": "solid"
        }
      ],
      "key_values": [
        {
          "x": 0.5,
          "y": 1.0
        }
      ]
    },
    {
      "title": "Test Data and Best Fit Quadratic",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "test data",
          "color": "blue",
          "marker": "circle"
        },
        {
          "name": "best fit quadratic",
          "color": "red",
          "line_style": "solid"
        }
      ],
      "key_values": [
        {
          "x": 0.5,
          "y": 1.0
        }
      ]
    },
    {
      "title": "Tic Model",
      "x_axis_label": "x",
      "y_axis_label": "y",
      "x_axis_scale": "linear",
      "y_axis_scale": "linear",
      "data_series": [
        {
          "name": "model",
          "color": "blue",
          "marker": "circle"
        },
        {
          "name": "ground truth h",
          "color": "red",
          "line_style": "dashed"
        }
      ],
      "key_values": [
        {
          "x": 0.5,
          "y": 1.0
        }
      ]
    }
  ],
  "diagrams_and_schematics": [
    {
      "title": "Diagram of Signal Flow",
      "components": [
        {
          "label": "Input Signal",
          "position": [0, 0]
        },
        {
          "label": "Filter",
          "position": [1, 0]
        },
        {
          "label": "Output Signal",
          "position": [2, 0]
        }
      ],
      "connections": [
        {
          "source": "Input Signal",
          "target": "Filter",
          "direction": "right"
        },
        {
          "source": "Filter",
          "target": "Output Signal",
          "direction": "right"
        }
      ]
    }
  ]
}
```


Figure 8.9: Best fit quadratic model has small training and test error because quadratic model achieves a better tradeoff.

Interestingly, the bias-variance tradeoff curves or the test error curves do not universally follow the shape in Figure 8.8, at least not universally when the model complexity is simply measured by the number of parameters. (We will discuss the so-called double descent phenomenon in Section 8.2.) Nevertheless, the principle of bias-variance tradeoff is perhaps still the first resort when analyzing and predicting the behavior of test errors.

8.1.1 A mathematical decomposition (for regression)

To formally state the bias-variance tradeoff for regression problems, we consider the following setup (which is an extension of the beginning paragraph of Section 8.1).

Draw a training dataset S = { x ( i ) , y ( i ) } n i =1 such that y ( i ) = h glyph[star] ( x ( i ) ) + ξ ( i ) where ξ ( i ) ∈ N (0 , σ 2 ).

Train a model on the dataset S , denoted by ˆ h S .

Take a test example ( x, y ) such that y = h glyph[star] ( x ) + ξ where ξ ∼ N (0 , σ 2 ), and measure the expected test error (averaged over the random draw of the training set S and the randomness of ξ ) 56

We will decompose the MSE into a bias and variance term. We start by stating a following simple mathematical tool that will be used twice below.

Claim 8.1.1: Suppose A and B are two independent real random variables and E [ A ] = 0. Then, E [( A + B ) 2 ] = E [ A 2 ] + E [ B 2 ].

As a corollary, because a random variable A is independent with a constant c , when E [ A ] = 0, we have E [( A + c ) 2 ] = E [ A 2 ] + c 2 .

The proof of the claim follows from expanding the square: E [( A + B ) 2 ] = E [ A 2 ] + E [ B 2 ] + 2 E [ AB ] = E [ A 2 ] + E [ B 2 ]. Here we used the independence to show that E [ AB ] = E [ A ] E [ B ] = 0.

Using Claim 8.1.1 with A = ξ and B = h glyph[star] ( x ) -ˆ h S ( x ), we have

Then, let's define h avg ( x ) = E S [ h S ( x )] as the 'average model'-the model obtained by drawing an infinite number of datasets, training on them, and averaging their predictions on x . Note that h avg is a hypothetical model for analytical purposes that can not be obtained in reality (because we don't

5 For simplicity, the test input x is considered to be fixed here, but the same conceptual message holds when we average over the choice of x 's.

6 The subscript under the expectation symbol is to emphasize the variables that are considered as random by the expectation operation.

have infinite number of datasets). It turns out that for many cases, h avg is (approximately) equal to the the model obtained by training on a single dataset with infinite samples. Thus, we can also intuitively interpret h avg this way, which is consistent with our intuitive definition of bias in the previous subsection.

We can further decompose MSE( x ) by letting c = h glyph[star] ( x ) -h avg ( x ) (which is a constant that does not depend on the choice of S !) and A = h avg ( x ) -h S ( x ) in the corollary part of Claim 8.1.1:

We call the second term the bias (square) and the third term the variance. As discussed before, the bias captures the part of the error that are introduced due to the lack of expressivity of the model. Recall that h avg can be thought of as the best possible model learned even with infinite data. Thus, the bias is not due to the lack of data, but is rather caused by that the family of models fundamentally cannot approximate the h glyph[star] . For example, in the illustrating example in Figure 8.2, because any linear model cannot approximate the true quadratic function h glyph[star] , neither can h avg , and thus the bias term has to be large.

The variance term captures how the random nature of the finite dataset introduces errors in the learned model. It measures the sensitivity of the learned model to the randomness in the dataset. It often decreases as the size of the dataset increases.

There is nothing we can do about the first term σ 2 as we can not predict the noise ξ by definition.

Finally, we note that the bias-variance decomposition for classification is much less clear than for regression problems. There have been several proposals, but there is as yet no agreement on what is the 'right' and/or the most useful formalism.

8.2 The double descent phenomenon

Model-wise double descent. Recent works have demonstrated that the test error can present a 'double descent' phenomenon in a range of machine

learning models including linear models and deep neural networks. 7 The conventional wisdom, as discussed in Section 8.1, is that as we increase the model complexity, the test error first decreases and then increases, as illustrated in Figure 8.8. However, in many cases, we empirically observe that the test error can have a second descent-it first decreases, then increases to a peak around when the model size is large enough to fit all the training data very well, and then decreases again in the so-called overparameterized regime, where the number of parameters is larger than the number of data points. See Figure 8.10 for an illustration of the typical curves of test errors against model complexity (measured by the number of parameters). To some extent, the overparameterized regime with the second descent is considered as new to the machine learning community-partly because lightly-regularized, overparameterized models are only extensively used in the deep learning era. A practical implication of the phenomenon is that one should not hold back from scaling into and experimenting with over-parametrized models because the test error may well decrease again to a level even smaller than the previous lowest point. Actually, in many cases, larger overparameterized models always lead to a better test performance (meaning there won't be a second ascent after the second descent).


> [Vision Analysis]: ```json
{
  "chart": {
    "title": "Bias-Variance Tradeoff",
    "x_axis": {
      "label": "# parameters",
      "scale": "linear",
      "units": ""
    },
    "y_axis": {
      "label": "test error",
      "scale": "linear",
      "units": ""
    },
    "data_series": [
      {
        "name": "classical regime",
        "line": "smooth curve",
        "description": "bias-variance tradeoff"
      },
      {
        "name": "modern regime",
        "line": "sharp curve",
        "description": "over-parameterization"
      }
    ],
    "key_points": [
      {
        "label": "bias-variance tradeoff",
        "description": "typically when # parameters fit the data"
      },
      {
        "label": "over-parameterization",
        "description": "when # parameters fit the data"
      }
    ]
  }
}
```


Figure 8.10: A typical model-wise double descent phenomenon. As the number of parameters increases, the test error first decreases when the number of parameters is smaller than the training data. Then in the overparameterized regime, the test error decreases again.

7 The discovery of the phenomenon perhaps dates back to Opper [1995, 2001], and has been recently popularized by Belkin et al. [2020], Hastie et al. [2019], etc.

with non-linear models. Understanding it will likely give you the language and backgrounds to understand various recent papers related to it.

As a running example, we will consider the following parameterization of p ( x, z ; θ ) by a neural network. Let θ be the collection of the weights of a neural network g ( z ; θ ) that maps z ∈ R k to R d . Let

Here I k × k denotes identity matrix of dimension k by k , and σ is a scalar that we assume to be known for simplicity.

For the Gaussian mixture models in Section 11.4, the optimal choice of Q ( z ) = p ( z | x ; θ ) for each fixed θ , that is the posterior distribution of z , can be analytically computed. In many more complex models such as the model (11.19), it's intractable to compute the exact the posterior distribution p ( z | x ; θ ).

Recall that from equation (11.10), ELBO is always a lower bound for any choice of Q , and therefore, we can also aim for finding an approximation of the true posterior distribution. Often, one has to use some particular form to approximate the true posterior distribution. Let Q be a family of Q 's that we are considering, and we will aim to find a Q within the family of Q that is closest to the true posterior distribution. To formalize, recall the definition of the ELBO lower bound as a function of Q and θ defined in equation (11.14)

Recall that EM can be viewed as alternating maximization of ELBO( Q,θ ). Here instead, we optimize the EBLO over Q ∈ Q

Now the next question is what form of Q (or what structural assumptions to make about Q ) allows us to efficiently maximize the objective above. When the latent variable z are high-dimensional discrete variables, one popular assumption is the mean field assumption , which assumes that Q i ( z ) gives a distribution with independent coordinates, or in other words, Q i can be decomposed into Q i ( z ) = Q 1 i ( z 1 ) · · · Q k i ( z k ). There are tremendous applications of mean field assumptions to learning generative models with discrete latent variables, and we refer to Blei et al. [2017] for a survey of these models and