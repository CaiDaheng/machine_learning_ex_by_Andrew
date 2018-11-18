# A note on collaborative filtering gradient

## The cost function

$\begin{array} { l } { X: n _ { m } \times n _ { t } }, \\ { \theta : n _ { u } \times n _ { f } },\\{Y : n _ { m } \times n _ { u }},\\{R:n _ { m } \times n _ { u }}. \end{array}$
$$
\begin{align}
J ( X , \theta ) &= \frac{1}{2} \sum_{i,j}\left( x _ { i } \theta _ { j } ^ { T } - y _ { i } ^ { j } \right) ^ { 2 }\\
&=\frac{1}{2} \sum_{i} \sum_{j}(x _ { i } \theta _ { j } ^ { T } - y _ { i } ^ { j } ) ^ { 2 }\\
&= \frac{1}{2} \sum_{i} \left( x _ { i } \theta ^ { T } - y _ { i } \right) \left( x _ { i } \theta ^ { T } - y _ { i } \right) ^ { T }\\
&=\frac{1}{2} \sum_{j} \left( X \theta _ { j } ^ { T } - y ^ { j } \right) ^ { T } \left( X \theta _ { j } ^ { T } - y ^ { j } \right).
\end{align}
$$


From $J(X,\theta) = \frac{1}{2} \sum_{i} \left( x _ { i } \theta ^ { T } - y _ { i } \right) \left( x _ { i } \theta ^ { T } - y _ { i } \right) ^ { T }$，
$$
\frac{\partial J}{\partial \theta} = \sum_{i} \left( x _ { i } \theta ^ { T } - y _ { i } \right)  x_i = (X\theta^T - Y)X.
$$
From $J(X,\theta) =\frac{1}{2} \sum_{j} \left( X \theta _ { j } ^ { T } - y ^ { j } \right) ^ { T } \left( X \theta _ { j } ^ { T } - y ^ { j } \right) ​$,
$$
\frac{\partial J}{\partial X} = \sum_{j} \left( x \theta _ { j } ^ { T } - y ^ { j } \right)   \theta _ { j } = (X\theta^T - Y)\theta.
$$

```matlab
predictions = X*Theta';
pre_cost = predictions - Y;

J = sum(sum(R.*pre_cost.^2))/2 +...
	sum(sum(Theta.^2))*lambda/2 + sum(sum(X.^2))*lambda/2;


Theta_grad = (R.*pre_cost)' * bsxfun(@times,X,sum(R,2) ~= 0) + lambda*Theta;
X_grad = (R.*pre_cost) * bsxfun(@times,Theta,(sum(R) ~= 0)') + lambda*X;
```



## Something in matrix operator

$A = [\alpha_1,\cdots,\alpha_m],B = [\beta_1,\cdots,\beta_m]^T~\Rightarrow ~C = AB = \sum_{k = 1}^m \alpha_k\beta_k^T$, where $\alpha_k \in R^{s},~\beta_k\in R^t$.
$$
\begin{align}
c_{i,j} &= \sum_{k = 1}^m \alpha_{k,i} \beta_{k,j}\\
&= \alpha_{1,i} \beta_{1,j} +\cdots+\alpha_{m,i} \beta_{m,j}\\
&= (\alpha_1\beta_1^T)_{i,j}+\cdots+(\alpha_m\beta_m^T)_{i,j}\\
& = (\alpha_1\beta_1^T+\cdots+\alpha_m\beta_m^T)_{i,j}.
\end{align}
$$


Hence we have $C = AB = \sum_{k = 1}^m \alpha_k\beta_k^T$.

Which give us the result of $\sum_{i} \left( x _ { i } \theta ^ { T } - y _ { i } \right) x_i = (X\theta^T - Y)X$ and $\sum_{j} \left( x \theta _ { j } ^ { T } - y ^ { j } \right)   \theta _ { j } = (X\theta^T - Y)\theta$.