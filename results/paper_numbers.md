# Numbers reported in the paper

## Data (Section 5, Table 1, Appendix B)
WCS languages kept: 109 of 110
WCS speakers kept: 2441 of 2616
terms per language: median 6, range 3-13
kept speakers per language: median 25
labeled chips per kept speaker: mean share 0.91
xkcd sessions: 13792 eligible, 1500 fitted
color check, archive Illuminant-C values vs the colour-science package's Munsell renotation: mean dE 0.72, max 1.23 (n = 320)
color check, shift from Illuminant C to D65 (Bradford), all 330 chips: mean dE 1.00, max 2.46 (n = 330)

## Synthetic validation (Section 4, Appendix B, Table 4)
relative error, kappa = 2, rows = flip rate nu, columns = T:
T       50     100    330    1000
noise                            
0.00   0.133  0.061  0.024  0.008
0.05   0.272  0.201  0.120  0.098
0.15   0.312  0.310  0.287  0.266
at nu = 0.15, mean of (kappa-hat minus kappa): 0.87

## Fit quality (Section 6, Table 2)
Euclidean, no weights:           0.647
fitted metric, no weights:       0.664
fitted metric, then weights:     0.746
metric and weights fitted jointly: 0.762
private prototypes:              0.705
focal-chip prototypes:           0.575
fitted kappa-hat: median 4.48, share at the cap 0.094
most compressed axis: L 0.22, a 0.10, b 0.68
xkcd: accuracy Euclidean 0.68, fitted 0.82; kappa-hat median 4.4; share at the cap 0.031

## Worked example (Figure 1)
lang 87 speakers 21 terms 5
speaker 1 (k=15): kappa_hat 1.60 accuracy 0.81
speaker 2 (k=9): kappa_hat 7.48 accuracy 0.76
chips where speakers 1 and 2 disagree: 54 of 330
model common ground of all speakers: 159 chips
empirical common ground of all speakers: 35 chips
empirical common ground inside model common ground: 33
kappa_max 7.62 q 0.966
('C', 0, 0, 17, 1, True, True)
('I', 0, 0, 34, 17, True, True)
('O', 0, 2, 30, 0, True, True)
('R', 0, 0, 30, 0, True, True)
('S', 0, 7, 48, 17, True, True)
totals [0, 9, 159, 35]

## H1 (Section 6)
terms with at least 3 common-ground chips: 229; testable: 173
not testable: {'degenerate': 47, 'achromatic': 9}
scattered baseline: hull-fill above the baseline median in 173 of 173
median hull-fill 0.72, median scattered baseline 0.14, share with p < 0.05: 0.95
connected baseline: testable 173 of 173; hull-fill above the baseline median in 11, equal in 27, below in 135
median hull-fill 0.72, median connected baseline 0.91, share with p < 0.05: 0.00
decision rule (at least 80% of testable terms above the baseline median): scattered met, connected not met
model vs empirical common ground, median Jaccard: 0.20

## H2 (Section 6)
languages: 109
r_S(kappa-hat, relative core) = -0.26, p = 0.007
five speakers: r_S = -0.26, p = 0.005
smallest margin of the core bound: 0.02 CIELAB units
refit with M >= 0.01 I: r_S = -0.44 (p = 1.4e-06); five speakers: r_S = -0.33 (p = 5.5e-04)

## H3 (Section 6, Figure 2 right)
languages: 104; slope median 1.88, IQR 1.81-1.98, max 2.38
xkcd: 61 patterns at n = 2, 17191 at n = 1024

## Follow-up predictions on sampling (Section 6, Figure 3)
languages: 95
N = 2000: exponent median 1.34, IQR 1.26-1.44
N = 20000: exponent median 1.94, IQR 1.85-2.01
N = 100000: exponent median 2.27, IQR 2.20-2.31
N = 100000, local exponents between n = 2-4, 4-8, 8-16, 16-24: [2.7 2.4 2.  1.6]
share of the box in D_1 by n (median): {2: 0.26, 4: 0.46, 8: 0.61, 16: 0.7, 24: 0.74}
observed count / (m + N * share of D_1): max 0.76; median at n = 24: {2000: 0.58, 20000: 0.27, 100000: 0.12}

## Tilts of the fitted walls (Appendix B)
tilt from the base diagram: mean 37.1 deg, max 75.0 deg
tilt between two speakers, over all speaker pairs of all languages: mean 29.2 deg

## H4, exploratory (Appendix C)
mean core radius by number of pooled languages: {1: 3.4, 2: 2.8, 4: 1.1, 8: 1.4, 16: 0.4}
share of measurements with radius zero by number of pooled languages: {1: 0.63, 2: 0.7, 4: 0.85, 8: 0.75, 16: 0.87}
r_S(number of pooled languages, core radius) = -0.20, p = 5.8e-04, measurements: 300

## H5 (Appendix C)
languages: 109; r_S median 0.057, IQR 0.03-0.11
positive in 95 of 109 languages; positive with p < 0.05 in 54; positive with p < 0.01 in 43; Wilcoxon p = 1.9e-13
mean over languages of the mean tilt between sampled speaker pairs: 28 deg
