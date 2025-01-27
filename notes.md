# Setup
Conda
```
mamba env create -f environment.yaml

```
# Model Notes

### Additive and Multiplicative interaction

According to [forecaster.py](https://github.com/facebook/prophet/blob/101dd50e3195c1875ee856cdf49ee9fcd6a87fa3/python/fbprophet/forecaster.py#L1191-L1194) the regressors combine as 

$$
y = \text{trend} * (1 + \text{multiplicative terms}) + \text{additive terms}
$$


$$
\begin{align*}
    y(t) &= \text{trend}(t) &&+ \text{seasonality}(t) &&+ \text{holiday}(t)
    \\
    &= g(t) &&+ s(t) &&+h(t)
\end{align*}
$$


## Growth
### Logitistic
Start with
$$
g(t) = \frac{C(t)}{1+\exp{(-k(t-m))}}
$$

Allow the growth rate $k$ to change by $\delta_j$ at time $s_j$ for $j=1,...,S$. 

The growth rate at time $t$ is then defined as 
$$
\begin{align*}
    k(t)
    &= k+\sum_{j<t}\delta_j \\
    &= k+\mathbf{a(t)^T \delta} & \text{for } \mathbf{a(t)}\in \{0,1\}^S, && a_j(t) = \mathbb{I}\{t>s_j\}
\end{align*}
$$

Equivalently,
$
k(t) = k_j
$
where $s_j$ is the last changepoint before $t$ and $k_j = k+\sum_{l=1}^j\delta_l$ .

$$
\begin{align*}
    k(t) &= 
    \begin{cases}
        k_j &\text{ for } j=\max{\{ s_l : l<t \}} & \text{if } t\geq s_1 \\
        k  && \text{if } t < s_1
    \end{cases} 
\end{align*}
$$

To maintain continuity we need to adjust the offset parameter $m$. Define $m(t)$ in the same way. Need to enforce continuity at the changepoints. Require when $t=s_j$ that,

$$
\begin{align*}
    k_{j-1} (t - m_{j-1}) &= k_j(t-m_j) \\
    \frac{k_{j-1}}{k_j}(t-m_{j-1}) &= t-m_j \\
    m_j &=t - \frac{k_{j-1}}{k_j}(t-m_{j-1}) \\
    \iff m_j-m_{j-1} &= t(1-\frac{k_{j-1}}{k_j}) + m_{j-1}(\frac{k_{j-1}}{k_j} - 1) \\
    &= (t-m_{j-1}) (1-\frac{k_{j-1}}{k_j})
\end{align*}
$$

Hence the changepoints in offset are 
$$
\gamma_j = (s_j-m_{j-1})(1-\frac{k_{j-1}}{k_j})
$$

### Linear
If we generalise rate and offset to $g(t)=k(t)t + m(t)$,
then we again need the offset to ensure continuity at changepoints which can be done with offset
$-s_j k_j +s_j k_{j-1} = -s_j \delta_j$.