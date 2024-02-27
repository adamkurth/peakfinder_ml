## Support Vector Machine

[Wikipedia SVM](https://en.wikipedia.org/wiki/Support_vector_machine)

Support vector machines (SVMs), are a supervised machine learning algorithm that can be used for both classification and regression tasks. SVMs are more commonly used in classification problems. In this article, we will focus on the classification problem.

### Motivation

Classification of data is a common task in machine learning. For example, we may want to classify an email as spam or not spam, or we may want to classify a tumor as malignant or benign. In these cases, we have a set of data points, and we want to classify them into two or more classes.

Much the same, we have a set of data points and the desire to classify them. In this repo, we want to determine what is and is not a Bragg peak in analyzing crystallography images. We have a set of data points, and we want to classify them into two classes: Bragg peak and not Bragg peak.

### How does SVM work?

With the data that we do have, the *goal* is to determine which class the *new* data points will be in. In the case of support vector machines, a data point is viewed as a $p$ dimensional vector (list of $p$ numbers), abd want to know whether we can seperate such points with a $(p-1)$-dimensional hyperplane. This is called the *linear classifier*. Its important to know that there are many hyperplanes that can classify this data. So we chose the hyperplane so that the distance from it to the nearest data point on each side is maximized. This is called the *maximal margin classifier* (equivalently *perceptron of optimal stability*), and if this hyperplane exists, its known as the *maximum margin hyperplane*.

Intuitively, a good seperation is achieved by the hyperplane that has the largest distance to the nearest training data point of any class (so-called *functional margin*) since in general the larger the margin the lower the *generalization error* of the classifier. A lower *generalization error* means that the implementer is less likely to experience overfitting.


If the data is not linearly seperable in the finite space, we can map the finite dimensional space into a higher-dimensional space, with the assumption that the data this will make the seperation easier. To keeps the compuational cost down, the mappings used by SVM schemes are designed to ensure the dot products of pairs of input data vectors to be computed easily in terms of the variables in the original space, by defining them in terms of a *kernel function* $K(x,y)$ selected to suit the problem. The hyperplanes in the higher-dimensional space are defined as the set of points whose dot product with a vector in that space is constant, where the vector is given by the sum of the input vector multiplied by the corresponding *dual variable* (see [Wikipedia SVM](https://en.wikipedia.org/wiki/Support_vector_machine) for more information).


### Linear SVM 

Given the training data of $n$ points: 

\[(x_{1}, y_{1}), \cdots , (x_{n}, y_{n}) \]

where $y_i$ is either 1 or -1, indicating the class of $\mathbf{x_i}$, and each $\mathbf{x_i}$ is a $p$-dimensional real vector. We want to find the *maximum-margin hyperplane* that divides the group of points $\mathbf{x_i}$ for which $y_i = 1$ from the group of points for which $y_i = -1$, which is defined so that the distance between the hyperplane and the nearest point from either group is maximized. Any hyperplane can be written as the set of points $\mathbf{x}$ satisfying 

\[\mathbf{w}^T \mathbf{x} - b = 0\]

Such that $\mathbf{w}$ is not necessarily a normalized, normal vector to the hyperplane. The parameter $\frac{b}{\|\ \mathbf{w} \|\}$ determines the offset of the hyperplane from the origin along the normal vector $\mathbf{w}$.

#### Hard Margin 

Suppose that the training data *is* linearly seperable, we select two parallel hyperplanes that seperate the two classes, so that the distance between them is as large as possible. The *margin* is the region bounded by the two hyperplanes, and the maximum margin hyperplane is the hyperplane that lies halfway between them. Such that 

$$
\begin{align*}
    \mathbf{w}^T \mathbf{x} - b &= 1 \\
    \mathbf{w}^T \mathbf{x} - b &= -1
\end{align*}
$$

The distance between the hyperplanes is $\frac{2}{\|\ \mathbf{w} \|}$, thus to maximize the distance between the planes we want to minimize $\|\ \mathbf{w} \|$. To prevent th data points from falling into the margin, the following contraint is added: 

For each $i$, these constraints state that each data point must lie on the correct side of the margin.

$$
\begin{align*}
    \mathbf{w}^T & \mathbf{x_i} - b &\geq 1 \quad \text{if} \quad y_i = 1 \\
    \mathbf{w}^T & \mathbf{x_i} - b &\leq -1 \quad \text{if} \quad y_i = -1\\
    \text{Rewriting this as:} \\
    y_i(\mathbf{w}^T & \mathbf{x_i} - b) &\geq 1
\end{align*}
$$

Thus the optimization problem to solve is:

$$
\begin{align*}
    \underset{\mathbf{w}, b}{\text{minimize}} & \quad  \|\ \mathbf{w} \|\_2^2 \\ 
    \text{subject to} & \quad y_i(\mathbf{w}^T \mathbf{x_i} - b) \geq 1 \quad \text{for} \quad \forall_{i = 1, \cdots, n}
\end{align*}
$$

The $\mathbf{w}, b$ that solve this problem determine the classifier, because the sign of $\mathbf{x} \mapsto \text{sgn}(\mathbf{w^T}\mathbf{x} - b)$. This geometric descriptio is that the max-margin hyperplane is completely determined by those $\mathbf{x_i}$ that are closest to it. These $\mathbf{x_i}$ are called *support vectors*.

#### Soft Margin

