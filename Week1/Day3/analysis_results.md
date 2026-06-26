# Statistical and Theoretical Analysis: Confidence Intervals for Erlang Loss System

This report compares the simulated blocking fractions and $95\%$ Confidence Intervals (CIs) for the Erlang loss systems simulated in Part 1, Part 2, and Part 3 of the Week 1 Day 3 exercises.

The simulations are based on a $10$-server system ($m = 10$) with an arrival rate of $\lambda = 1.0$ and a mean service time $E[S] = 8.0$ (giving an offered load of $A = \lambda E[S] = 8.0$). Each simulation run models $10,000$ customers, and results are averaged over $10$ independent replications.

---

## 1. Simulation Results Summary

The table below summarizes the simulated mean blocking fractions, standard deviations, and $95\%$ confidence intervals obtained across all scenarios.

| Scenario | Service Dist. | Arrival Dist. | Simulated Mean | Std. Deviation | 95% Confidence Interval | Covers Theoretical ($0.121661$)? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Part 1** | Exponential | Exponential | $0.118430$ | $0.006754$ | $[0.113599, 0.123261]$ | **Yes** |
| **Part 2** (Exp) | Exponential | Exponential | $0.123520$ | $0.006585$ | $[0.118809, 0.128231]$ | **Yes** |
| **Part 2** (Erl-4) | Exponential | Erlang-4 | $0.076230$ | $0.003587$ | $[0.073664, 0.078796]$ | No (Different system) |
| **Part 2** (Hyper) | Exponential | Hyperexponential | $0.136240$ | $0.007172$ | $[0.131109, 0.141371]$ | No (Different system) |
| **Part 3** (Const) | Constant | Exponential | $0.122050$ | $0.002609$ | $[0.120184, 0.123916]$ | **Yes** |
| **Part 3** (Erl-2.05) | Erlang-2.05 | Exponential | $0.121400$ | $0.006049$ | $[0.117073, 0.125727]$ | **Yes** |
| **Part 3** (Erl-1.05) | Erlang-1.05 | Exponential | $0.121570$ | $0.004328$ | $[0.118474, 0.124666]$ | **Yes** |
| **Part 3** (Exp) | Exponential | Exponential | $0.120970$ | $0.007072$ | $[0.115911, 0.126029]$ | **Yes** |
| **Part 3** (Pareto-1.05) | Pareto-1.05 | Exponential | $0.000640$ | $0.000570$ | $[0.000232, 0.001048]$ | **No (Underestimated)** |

> [!NOTE]
> The analytical blocking probability for the $M/G/10/10$ queue with an offered load of $A = 8.0$ is computed using the **Erlang-B formula**:
> $$B(10, 8) = \frac{\frac{8^{10}}{10!}}{\sum_{k=0}^{10} \frac{8^k}{k!}} \approx 0.121661$$

---

## 2. Methodology

The comparison results were gathered by running a standardized Python test harness ([compare_cis.py](file:///home/isako/.gemini/antigravity-cli/brain/ce65a328-d9f2-4601-a466-29c9f2af5718/scratch/compare_cis.py)) that replicates the models described in:
- [e1.py](file:///home/isako/Stochastic-Simulation/Week1/Day3/e1.py#L14-L60) (Part 1: M/M/10/10)
- [e2.py](file:///home/isako/Stochastic-Simulation/Week1/Day3/e2.py#L7-L57) (Part 2: G/M/10/10)
- [e3.py](file:///home/isako/Stochastic-Simulation/Week1/Day3/e3.py#L25-L71) (Part 3: M/G/10/10)

### A. Discrete Event Simulation (DES) Engine
The system was simulated using a discrete event model with a min-heap priority queue tracking events ordered by time. 
* **State Variables**: Number of busy servers ($0 \le \text{busy\_servers} \le m$), total arrivals ($C$), and blocked customers.
* **Events**:
  1. **Arrival**: 
     - If $\text{busy\_servers} < m$, the customer is accepted, $\text{busy\_servers}$ is incremented by 1, and a departure event is scheduled at $\text{current\_time} + \text{service\_duration}$.
     - If $\text{busy\_servers} = m$, the customer is blocked and added to the blocked count.
     - If total arrivals $C < 10,000$, the next arrival is scheduled at $\text{current\_time} + \text{interarrival\_time}$.
  2. **Departure**: Decrements $\text{busy\_servers}$ by 1.

### B. Generation of Random Variables
Let $U \sim \text{Uniform}(0, 1)$ be a pseudo-random floating-point number.
* **Exponential Distribution**: Used for standard inter-arrival and service times.
  $$X = -\mu_{mean} \ln(U)$$
* **Erlang-$k$ Distribution**: Sum of $k$ independent exponential stages. Generated via the Gamma distribution:
  $$X \sim \text{Gamma}\left(k, \frac{\text{mean}}{k}\right)$$
* **Hyperexponential Distribution ($H_2$)**: A mixture distribution with a probability $p_1 = 0.8$ of selecting an Exponential rate $\lambda_1 = 0.8333$, and probability $p_2 = 0.2$ of selecting Exponential rate $\lambda_2 = 5.0$.
* **Pareto Distribution**: Heavy-tailed distribution with shape parameter $k$ and scale (minimum value) $x_m$.
  $$X = x_m U^{-1/k}$$
  To achieve a target mean $E[X]$, we compute the scale parameter as:
  $$x_m = E[X] \cdot \frac{k - 1}{k}$$
  For Part 3 (Pareto-1.05), $k = 1.05$ and $E[X] = 8.0$, giving $x_m = 8.0 \cdot \frac{0.05}{1.05} \approx 0.38095$.

### C. Statistical Confidence Intervals
For each scenario, $N = 10$ independent replications are executed with independent random seeds.
1. The blocking fraction $Y_i$ is recorded for each replication $i \in \{1, \dots, N\}$.
2. The **Sample Mean** ($\bar{Y}$) is computed:
   $$\bar{Y} = \frac{1}{N} \sum_{i=1}^N Y_i$$
3. The **Sample Standard Deviation** ($S_Y$) is computed:
   $$S_Y = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (Y_i - \bar{Y})^2}$$
4. The **$95\%$ Confidence Interval** is calculated using the Student's t-distribution with $df = N - 1 = 9$ degrees of freedom:
   $$\text{CI} = \left[ \bar{Y} - t_{crit} \frac{S_Y}{\sqrt{N}}, \bar{Y} + t_{crit} \frac{S_Y}{\sqrt{N}} \right]$$
   For a $95\%$ two-tailed CI with $9$ degrees of freedom, the critical value is $t_{crit} = t_{0.025, 9} \approx 2.262157$.

---

## 3. Key Comparisons and Interpretations

### A. Part 1 vs. Part 3: The Insensitivity Theorem
In queueing theory, the **Erlang Loss System (M/G/m/m)** exhibits a remarkable property known as **insensitivity**: the steady-state probability of blocking depends *only* on the mean of the service time distribution ($E[S]$) and is completely independent of its shape or variance.

* **Theoretical Verification**: 
  * The scenarios in Part 3 (Constant, Erlang, Exponential service times) all have a theoretical blocking probability of exactly $B(10, 8) \approx 0.121661$.
  * The simulated 95% CIs for Constant, Erlang-2.05, Erlang-1.05, and Exponential service times all overlap significantly, and **all successfully contain the theoretical Erlang-B value of 0.121661**.
* **Effect of Service Variance on CI Width**: 
  * While the *mean* blocking probability is insensitive to the service distribution, the **estimator variance** (and thus the width of the CI) is highly sensitive to it.
  * **Constant service time** (variance = $0$) has the most stable sample paths, leading to the smallest standard deviation ($0.002609$) and the narrowest 95% CI ($[0.120184, 0.123916]$).
  * **Exponential service time** (variance = $64.0$) has high variability in service durations, resulting in a larger standard deviation ($0.007072$) and a much wider 95% CI ($[0.115911, 0.126029]$).

### B. The Pareto-1.05 Anomaly: Heavy Tails and Infinite Variance
The Pareto-1.05 service distribution represents an extreme outlier. Its simulated blocking fraction is nearly zero ($0.000640$), and its confidence interval does not cover the theoretical value of $0.121661$.

* **Why it fails**:
  * For $k = 1.05$, the Pareto distribution is extremely heavy-tailed. It has a finite mean of $8.0$ but **infinite variance**.
  * Due to the infinite variance, the sample mean converges extremely slowly (governed by the generalized central limit theorem). For a sample size of $N = 10,000$, the sample mean service time is severely underestimated (experimentally measured at $\approx 3.16$ instead of $8.0$).
  * The theoretical mean is dominated by rare, massive values. In a finite simulation run, these values either do not appear at all or are truncated at the end of the simulation, meaning the queue operates at a much lower effective offered load ($A \approx 3.16$ instead of $8.0$).
  * Even when simulated with $1,000,000$ samples, the service mean only rises to $\approx 3.74$ (blocking fraction $\approx 0.0031$), confirming that standard discrete event simulation lengths are highly inadequate for heavy-tailed service distributions with $k \approx 1$.

### C. Part 1 vs. Part 2: Sensitivity to the Arrival Process
Unlike the service distribution, the Erlang loss system is **highly sensitive** to the arrival distribution. The insensitivity theorem does *not* apply to the arrival process ($G/M/m/m$).

* **Baseline (Exponential Arrivals)**:
  * For both Part 1 and Part 2 (Exponential), the arrival process is Poisson ($C_a^2 = 1.0$). Both simulations converge near the theoretical Erlang-B value of $0.121661$, and their CIs overlap heavily.
* **Erlang-4 Arrivals (Low Variance)**:
  * Erlang-4 arrivals are more regular ($C_a^2 = 0.25$). Because arrivals are spaced out more evenly, the probability of sudden "bursts" overloading the servers is reduced.
  * Consequently, the blocking probability drops significantly to $\approx 0.076230$ ($95\%$ CI: $[0.073664, 0.078796]$).
* **Hyperexponential Arrivals (High Variance)**:
  * Hyperexponential arrivals are highly variable and bursty ($C_a^2 > 1.0$).
  * Arrivals tend to cluster together, causing temporary server saturation and high blocking rates during bursts, while leaving the servers idle between bursts.
  * This raises the blocking probability to $\approx 0.136240$ ($95\%$ CI: $[0.131109, 0.141371]$).
