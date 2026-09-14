# Optimizer mathematics

Every rule here minimizes the same loss with the same gradients. What changes is how a gradient
becomes a step. Write $g_t$ for the gradient at step $t$, $\eta$ for the learning rate, and
$\theta_t$ for the parameters.

## What a mini-batch gradient is

The loss is a mean over $N$ samples, so its exact gradient is a mean of $N$ per-sample gradients.
A batch $B$ estimates it:

$$
\hat{g} = \frac{1}{|B|}\sum_{i \in B} \nabla L_i .
$$

This estimate is unbiased for any batch size, and for samples drawn independently its variance
falls like $1/|B|$. So the standard deviation of the noise shrinks only as $\sqrt{|B|}$ while the
cost of computing it grows as $|B|$: quadrupling the batch buys one halving of the noise for four
times the work. That is the whole argument for mini-batches over full-batch descent.

The three classical regimes are the same algorithm at different $|B|$:

- **Batch gradient descent** uses $|B| = N$. Each step is the exact gradient, and one pass over the
  data yields exactly one update.
- **Stochastic gradient descent** uses $|B| = 1$. One pass yields $N$ updates, each very noisy.
- **Mini-batch** sits between, and is what is actually used.

Two consequences follow, and the experiment measures both. First, progress per *pass over the data*
is governed by the number of updates, not by the accuracy of each one — which is why full-batch
descent is hopeless on a fixed epoch budget. Second, the batch size and the learning rate are
coupled: a noisier gradient needs a smaller step, so the same $\eta$ does not mean the same thing at
$|B| = 1$ and $|B| = 64$.

It is tempting to think a larger step can compensate for having few updates. It cannot. For a
quadratic with largest curvature $\lambda_{\max}$, gradient descent diverges unless
$\eta < 2/\lambda_{\max}$. The stable step size is bounded by the curvature of the problem, not by
how patient the practitioner is, so raising $\eta$ past that bound makes things worse rather than
faster.

## Momentum

$$
v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta v_t .
$$

For a gradient that keeps its sign, $v$ becomes a geometric series and converges to $g/(1-\mu)$, so
the effective step grows to $\eta/(1-\mu)$ times the gradient — a factor of 10 at $\mu = 0.9$. Along
a direction where the gradient keeps flipping sign, the terms cancel instead. Momentum therefore
accelerates along consistent directions and damps oscillation across a narrow valley, which is
exactly the situation plain descent handles worst.

That factor of $1/(1-\mu)$ also means momentum's usable learning-rate range is shifted down by
roughly an order of magnitude: at $\mu = 0.9$, running momentum at $\eta$ resembles running plain
descent at $10\eta$.

**Nesterov momentum** updates the velocity first and takes the step from the look-ahead point:

$$
v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\left(g_t + \mu v_t\right).
$$

While the gradient is consistent this steps further than classical momentum; when the gradient
reverses, the look-ahead term responds in the same step rather than one step later, so it brakes
sooner. It is a correction to momentum's overshoot, not a different idea.

## Per-coordinate scaling

A single global $\eta$ must serve every parameter, so it is bounded by the steepest direction while
the flattest directions barely move. **RMSProp** removes the scale from each coordinate separately:

$$
s_t = \rho s_{t-1} + (1-\rho) g_t^2, \qquad
\theta_t = \theta_{t-1} - \eta \frac{g_t}{\sqrt{s_t} + \varepsilon} .
$$

Since $\sqrt{s_t}$ tracks the typical magnitude of that coordinate's gradient, the step is roughly
$\eta \cdot \operatorname{sign}(g)$ regardless of whether the gradient is $100$ or $0.01$. The
$\varepsilon$ floor keeps a coordinate with a vanishing gradient history from dividing by zero.

This normalization has a consequence that is easy to miss: because the step no longer shrinks as the
gradient shrinks, RMSProp with a fixed $\eta$ does not converge to a minimum — it settles into a
limit cycle around it at a radius proportional to $\eta$. On $f(x) = x^2/2$ the residual is exactly
$\eta/2$, which the tests assert directly. Reaching the minimum requires decaying $\eta$, and that
is the concrete reason learning-rate schedules exist.

**Adam** combines the two ideas and corrects the bias that zero-initialized averages introduce:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} .
$$

The correction matters most at the start. Both averages begin at zero, so $m_1 = (1-\beta_1)g_1$ is
ten times too small at $\beta_1 = 0.9$ and $v_1$ is a thousand times too small at
$\beta_2 = 0.999$. Uncorrected, the first step would be scaled by
$(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2$ — wrong in an unhelpful direction and wrong by a
different factor at every step. Dividing by $1-\beta^t$ removes exactly that bias, and at $t = 1$
gives $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, so the first step is
$\eta\, g/(|g| + \varepsilon) \approx \eta \operatorname{sign}(g)$ whatever the gradient's scale.
The tests check that property directly, including for a gradient of $10^{-6}$.

## What the comparison can and cannot show

Adam is often described as needing no tuning. The experiment measures something narrower and more
defensible. Across the decade Adam is normally run in, $\eta \in [10^{-3}, 10^{-2}]$, its result
barely moves while plain descent's changes by a large margin. But Adam's usable range is *shifted
down and narrower in absolute terms*, not wider: because its step is about $\eta$ per coordinate
regardless of the gradient, $\eta = 1$ means a step of size one in every parameter, and it diverges
where plain descent still works. The same shift applies to momentum, for the $1/(1-\mu)$ reason
above.

So a fair comparison gives each rule a rate suited to it, and reports both the result and the
sensitivity. Comparing rules at one shared learning rate measures which rule happens to like that
rate, not which rule is better. A fixed budget also has to be stated in the right units: equal
epochs means equal passes over the data and very unequal numbers of updates, while equal updates
means very unequal amounts of computation. The experiment fixes epochs, reports the update count
alongside, and records wall-clock time so the cost of many small updates is visible.
