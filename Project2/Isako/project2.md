DTU 02443 Stochastic Simulation
2021-06-21
ARA/ara,bfn
Utilization of hospital beds during epidemics
Background
An essential element of hospital planning is the decision of how many resources each ward should
have. This decision has both technical, organizational as well as political complications making the
problem incredibly difficult to solve. In this project, the aim is to approach the problem from a purely
technical perspective by deriving a simulation model that evaluates the implications from employing
a certain resource distribution.
The Emergency Department (ED) is often regarded as the main entrance to the hospital, and
can receive hundreds of patients with various diagnoses each day. The arrival intensity to the ED
fluctuates substantially, but in a weekly cyclical pattern. Fortunately, the majority of acute patients
are discharged directly from the ED, whereas a smaller fraction are transferred to the inpatient wards.
Hence, the arrival stream to inpatient wards consists of both transferred former acute patients as well
as elective admissions, and is time-independent – under normal circumstances.
In this project, we consider a particular year where a hospital has been forced to create two new
temporary wards due to a countrywide epidemic. The new wards admit patients requiring both regular
(Ward A) and intensive care (Ward B), respectively. If an intensive care patient arrives, and all the
beds in Ward B are occupied, the patient is admitted in Ward A. If a regular patient arrives when
Ward A is completely occupied, the patient is relocated to a different hospital. Furthermore, in order
to accommodate the need for the temporary wards, the hospital has been forced to move staff and
beds from an inpatient ward (Ward C) where patients also arrive on a daily basis, and are relocated
to a different hospital in the case of insufficient beds.
Various studies have found that patient arrivals are governed by a Poisson process (independent
of the receiving ward) and that the patient’s Length-Of-Stay (LOS) is often governed by a log-normal
distribution. Although in most applications of queueing theory for hospital planning, the LOS distribution is assumed exponential for convenience. In this project, we will assume the arrival processes of
all patient types are Poisson and that the LOS distribution is log-normal, but investigate the system’s
sensitivity to different distributions.
We assume (for convenience) that the intensity of the epidemic behaves as a second-order polynomial starting at t = 0 and ending in t = 365. A description of parameters for each patient type and
ward is found below.
Ward A – Regular care
Arrival rate: λ1(t) = −(1/3650)t
2 + (1/10)t patients per day, where 0 ≤ t ≤ 365.
Length-of-stay: Lognormal(µ, σ2
) with µ = log(4√ 2) and σ
2 = log(2), corresponding to a mean
and standard deviation of 8 days.
Ward B – Intensive care
Arrival rate: λ2(t) = 1
5
λ1(t) patients per day.
Length-of-stay: Lognormal(µ, σ2
) with µ = log(6√ 2) and σ
2 = log(2), corresponding to a mean
and standard deviation of 12 days.
Ward C – Other
Arrival rate: λ3 = 6 patients per day.
Length-of-stay: Lognormal(µ, σ2
) with µ = log(5√ 2) and σ
2 = log(2), corresponding to a mean
and standard deviation of 10 days.
Moreover, Ward C contained 75 beds, but some of these are now moved to Ward A and B.
Primary task
Assuming the system starts in t = 0 and ends in t = 365, build a simulation model that simulates
the patient flow for all three patient types and wards as a function of the bed distribution and the
aforementioned parameters.
Primary performance measures
Estimate the probability that all beds are occupied on arrival for each of the three patient types as
well as the mean number of patients that are relocated due to shortage of beds. Estimate the latter
for each individual patient type and as a sum over all types.
Furthermore, estimate the mean fraction of beds that are utilized (occupied) in each ward.
Sensitivity analysis
Test the sensitivity to the distribution of the 75 beds and try to find the optimal bed distribution
considering the sum of the relocated patients. Assume the bed distribution is fixed during the entire
365 day period. Furthermore, test the system’s sensitivity on the different performance measures to
the LOS distribution by employing an exponential distribution instead.
Lastly, evaluate the impact of increasing the total amount of beds in the system to for instance 80
or 100 beds. Also, what would be the impact of having fewer beds?
