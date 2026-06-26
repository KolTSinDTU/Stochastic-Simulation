# Hospital Bed Simulation & Optimization Report

This report presents the findings from the stochastic simulation models developed to evaluate the utilization of hospital beds in three wards (Ward A, Ward B, and Ward C) over a 365-day epidemic period.

We implemented and compared two exact algorithms for generating Non-Homogeneous Poisson Process (NHPP) arrivals:

1. **Option A: Time-Transformation (Inversion) Method** (via numerical Bisection)
2. **Option B: Acceptance-Rejection (Thinning) Method**

Both models were run for **100 replications** per configuration to estimate the performance measures.

---

## 1. NHPP Arrival Generation Comparison

Both methods yield identical statistics in the limit, but approach the problem differently:

- **Option A (Inversion):** Generates arrivals on a homogeneous timeline ($S \sim \text{Poisson}(1)$) and uses bisection search to map them to the real timeline using the inverse cumulative intensity function $t = \Lambda^{-1}(S)$. This is mathematically exact and builds directly on the **Inverse Transform Method**.
- **Option B (Rejection):** Proposes arrival times using a homogeneous process running at the peak intensity $\lambda_{\max}$, then accepts each proposal with probability $p(t) = \lambda(t)/\lambda_{\max}$. This builds directly on the **Acceptance-Rejection Method**.

###

When a Ward B patient overflows into Ward A, the code samples LOS from Ward B's distribution, which is correct (the patient's care needs don't change). However, the departure is tagged as ward 'A', meaning it will decrement occupied_A on departure. This is correct for bed counting purposes, but worth being explicit about in your report since it's a modeling assumption.

Blocking probability interpretation for Ward B — A Ward B patient is only counted as blocked if both Ward B and Ward A are full. This is correct per the spec, but when reporting "blocking probability for Ward B patients" you should be clear this means the probability of relocation to another hospital, not just the probability of not getting a Ward B bed.

No warm-up / transient handling — The simulation starts at t=0 with empty wards, which matches the problem statement ("system starts at t=0"), so this is intentional and correct. Just confirm your report acknowledges the system starts empty.

The problem asks you to find the optimal allocation minimising sum of relocated patients. Your objective function is exactly that, which is correct. However, you might want to briefly justify why you're not weighting Ward B relocations more heavily than Ward A or C relocations, since intensive care patients presumably face worse outcomes. The spec doesn't ask for this, but it shows critical thinking.

### Arrival Intensity Curves

Both scripts produced identical arrival profiles over the year:

```carousel
![Inversion Arrival Rates](inversion_arrival_rates.png)
<!-- slide -->
![Rejection Arrival Rates](rejection_arrival_rates.png)
```

---

## 2. Bed Allocation Optimization ($N = 75$)

We optimized the bed distribution $(N_A, N_B, N_C)$ to minimize the **total number of relocated patients** (the sum of relocated patients across all three types).

### Baseline vs. Optimal Results

| Metric                         | Baseline: Inversion ($25, 15, 35$) | Optimal: Inversion ($30, 1, 44$) | Baseline: Rejection ($25, 15, 35$) | Optimal: Rejection ($34, 1, 40$) |
| :----------------------------- | :--------------------------------: | :------------------------------: | :--------------------------------: | :------------------------------: |
| **Ward A (Regular) Beds**      |                 25                 |                30                |                 25                 |                34                |
| **Ward B (Intensive) Beds**    |                 15                 |                1                 |                 15                 |                1                 |
| **Ward C (Other) Beds**        |                 35                 |                44                |                 35                 |                40                |
| **Mean Relocated (Regular)**   |              1270.81               |             1263.31              |              1265.09               |             1158.66              |
| **Mean Relocated (Intensive)** |               75.61                |              241.19              |               77.22                |              221.03              |
| **Mean Relocated (Other)**     |               931.52               |              635.43              |               941.62               |              762.69              |
| **Total Mean Relocated**       |            **2277.94**             |           **2139.94**            |            **2283.93**             |           **2142.39**            |
| **Ward A Utilization**         |               87.42%               |              88.08%              |               87.43%               |              86.91%              |
| **Ward B Utilization**         |               71.16%               |              89.17%              |               71.69%               |              89.90%              |
| **Ward C Utilization**         |               95.64%               |              94.17%              |               95.75%               |              94.91%              |

### Key Discussion Point: Why is $N_B = 1$ Optimal?

At first glance, allocating only **1 bed** to Ward B (Intensive Care) seems counter-intuitive and dangerous. However, it is mathematically optimal under the routing rules:

1. **Flexible Spillover:** If an intensive care patient arrives and Ward B is full, they can spill over into Ward A. However, if a regular patient arrives and Ward A is full, they are immediately relocated (lost). Ward C patients also have no spillover.
2. **Resource Pooling:** By shifting beds from B to A, we make those beds "flexible" (available to both regular and intensive patients). Keeping beds in Ward B is restrictive because regular patients cannot use them.
3. **Optimizing the Sum:** Minimizing the sum of relocated patients pushes resources to where they are most flexible (Ward A) and where demand is highest relative to capacity (Ward C).

### Optimization Heatmaps

The following heatmaps show the total relocated patients as a function of the Ward A and Ward B bed allocation. The red dot marks the optimal configuration:

```carousel
![Inversion Optimization Heatmap](inversion_optimal_allocation_heatmap.png)
<!-- slide -->
![Rejection Optimization Heatmap](rejection_optimal_allocation_heatmap.png)
```

---

## 3. Length-of-Stay (LOS) Sensitivity Analysis

We evaluated the system's sensitivity to the LOS distribution by replacing the **Log-normal** distribution with an **Exponential** distribution with the same means ($Mean_A = 8, Mean_B = 12, Mean_C = 10$).

As shown in the bar charts, the blocking probabilities are highly sensitive to the distribution type:

- **Log-normal** distributions, which have higher variance and a heavier right tail, generally lead to higher congestion and slightly higher blocking probabilities in bottleneck wards compared to **Exponential** service times.
- This highlights that assuming exponential LOS for mathematical convenience (as is common in classical queueing theory) would lead to underestimating patient relocation rates in hospital planning.

```carousel
![Inversion Sensitivity](inversion_los_sensitivity.png)
<!-- slide -->
![Rejection Sensitivity](rejection_los_sensitivity.png)
```

---

## 4. Bed Capacity Impact Analysis

Lastly, we evaluated the impact of varying the total bed capacity ($N = 60, 75, 80, 100$). The plots show the minimum relocated patients as a function of total capacity:

- Reducing capacity to **60 beds** drastically increases relocated patients.
- Increasing capacity to **100 beds** reduces relocations significantly, though a non-zero number remains due to peak epidemic spikes.
- The optimal bed distribution adjusts dynamically (e.g., as total beds increase, more beds are allocated to Ward C to handle its high load).

```carousel
![Inversion Capacity Impact](inversion_capacity_impact.png)
<!-- slide -->
![Rejection Capacity Impact](rejection_capacity_impact.png)
```

Here is a structured, concise report detailing the implementation, results, and analyses of the stochastic simulation project, based on your code and provided documentation.

---

# Technical Report: Stochastic Simulation of Hospital Bed Utilization During an Epidemic

## 1. Introduction

This project aims to optimize the allocation of 75 hospital beds across three wards (Ward A for regular care, Ward B for intensive care, and Ward C for other patients) during a 365-day countrywide epidemic. The primary goal of the simulation model is to estimate the probability of beds being fully occupied upon patient arrival and to find the optimal bed distribution that minimizes the total number of patients relocated to other hospitals.

## 2. Model Implementation and Methodology

The simulation was built from a purely technical perspective, employing a Discrete Event Simulation (DES) engine driven by stochastic patient arrivals and lengths of stay (LOS). Each configuration was evaluated using 100 replications to ensure statistical reliability.

### 2.1 Non-Homogeneous Poisson Process (NHPP) Arrivals

Patient arrivals to the hospital fluctuate depending on the epidemic's intensity, modeled as a second-order polynomial peaking at 182.5 days:

- **Ward A:** $\lambda_1(t) = -(1/3650)t^2 + (1/10)t$
- **Ward B:** $\lambda_2(t) = 0.2\lambda_1(t)$
- **Ward C:** $\lambda_3 = 6$ (constant)

The Python implementation utilizes the **Time-Transformation (Inversion) Method** to generate these arrivals. By calculating the exact cumulative intensity function $\Lambda(t)$ mathematically, arrivals generated from a standard homogeneous Poisson process with rate 1 are mapped to the real timeline using the inverse cumulative intensity.

### 2.2 Routing Logic and Length-of-Stay (LOS)

The DES routing logic dictates that if Ward B is full, intensive care patients spill over into Ward A. If a regular patient arrives and Ward A is full, or a Ward C patient arrives and Ward C is full, the patient is relocated to another hospital.

The primary LOS follows a log-normal distribution, sampled using `np.random.lognormal` with specific $\mu$ and $\sigma$ parameters configured to yield means of 8, 12, and 10 days for Wards A, B, and C, respectively.

## 3. Optimization and Results

To minimize the total number of relocated patients, a hierarchical optimization engine was implemented to test different integer distributions of the 75 beds.

### 3.1 Baseline vs. Optimal Bed Allocation

The simulation reveals a significant improvement in system performance when shifting beds from the baseline to the optimal configuration:

| Metric                      | Baseline: Inversion ($25, 15, 35$) | Optimal: Inversion ($30, 1, 44$) |
| :-------------------------- | :--------------------------------- | :------------------------------- |
| **Ward A (Regular) Beds**   | 25                                 | 30                               |
| **Ward B (Intensive) Beds** | 15                                 | 1                                |
| **Ward C (Other) Beds**     | 35                                 | 44                               |
| **Total Mean Relocated**    | **2277.94**                        | **2139.94**                      |
| **Ward A Utilization**      | 87.42%                             | 88.08%                           |
| **Ward B Utilization**      | 71.16%                             | 89.17%                           |
| **Ward C Utilization**      | 95.64%                             | 94.17%                           |

_Note: Table extracted from the simulated results data._

### 3.2 Analysis of the Optimal Configuration

The mathematically optimal allocation assigns only **1 bed** to Ward B. While counter-intuitive for an intensive care ward, this is the logical result of the system's routing rules and resource pooling. Because Ward B patients can safely spill over into Ward A, beds placed in Ward A are strictly more "flexible" (available to both intensive and regular patients). Conversely, beds isolated in Ward B cannot be utilized by regular patients. Moving beds to Ward A maximizes total utility, effectively acting as a flexible pool that minimizes overall relocations.

## 4. Sensitivity Analyses

### 4.1 Sensitivity to LOS Distribution

When comparing the default Log-normal LOS to an Exponential distribution (with identical means), the Log-normal model exhibits higher congestion and blocking probabilities. The heavy right tail and higher variance inherent to the log-normal distribution cause beds to be occupied for extended outlier periods. This demonstrates that modeling hospital stays using an Exponential distribution—a common convenience in classical queueing theory—would dangerously underestimate the actual patient relocation rates.

### 4.2 Sensitivity to Total Capacity

Varying the total bed capacity away from 75 beds ($N = 60, 80, 100$) drastically alters the system state. A reduction to 60 beds yields a massive surge in relocations, while increasing capacity to 100 heavily mitigates the issue, though a non-zero relocation rate persists due to the harsh spikes in peak epidemic intensity. As total capacity increases, the optimal distribution dynamically shifts proportionally more beds toward the constantly loaded Ward C.

## 5. Conclusions and Key Takeaways

1. **Flexibility outperforms siloing:** Permitting spillover from specialized wards to general wards drastically reduces overall hospital rejections, provided general wards are given the bulk of the bed capacity.
2. **Variance drives congestion:** High-variance patient stays (log-normal) result in worse operational performance metrics than memoryless (exponential) stays of the same average length.
3. **Data-driven validation is critical:** As highlighted by the principles of building credible simulation models, relying on appropriate distributions rather than convenient theoretical ones is necessary to prevent invalid real-world decision-making.

---

## Appendix: Core Implementation Snippets

**A.1. Arrival Rate and Cumulative Intensity Functions**
Relevant material for generating NHPP arrivals via the inversion method:

```python
def lambda1_rate(t):
    # Arrival rate function for Ward A patients.
    if 0 <= t <= 365:
        return -(1.0 / 3650.0) * (t ** 2) + 0.1 * t
    return 0.0

def Lambda1_cum(t):
    # Cumulative intensity function for Ward A.
    if t < 0:
        return 0.0
    if t > 365:
        t = 365.0
    return -(t ** 3) / 10950.0 + (t ** 2) / 20.0
```

**A.2. Stochastic LOS Sampling**
Material mapping the respective distributions depending on the sensitivity analysis settings:

```python
def sample_los(ward, dist_type):
    # Samples length of stay for a given ward and distribution type.
    if dist_type == 'lognormal':
        if ward == 'A': return np.random.lognormal(MU_A, SIGMA_A)
        elif ward == 'B': return np.random.lognormal(MU_B, SIGMA_B)
        elif ward == 'C': return np.random.lognormal(MU_C, SIGMA_C)
    elif dist_type == 'exponential':
        if ward == 'A': return random.expovariate(1.0 / 8.0)
        elif ward == 'B': return random.expovariate(1.0 / 12.0)
        elif ward == 'C': return random.expovariate(1.0 / 10.0)
```
