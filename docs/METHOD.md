# Proposed Method: U-FusePF

## 1. Paper-faithful modality experts

Two residual CNNs reproduce the original method: one receives dynamic/background CIR and the other receives dynamic/background variance. They output excess-delay estimates `mu_c` and `mu_v`. Keeping these experts explicit provides a strong and interpretable connection to the original paper.

## 2. Local reliability network

A compact six-channel network receives dynamic, background and absolute-difference profiles for CIR and variance. Its modality branches predict local uncertainty and local reliability signals. It is trained with a heteroscedastic Student-t objective, robust location loss, auxiliary branch losses and a confidence-weighted consistency term.

## 3. Validation-prior reliability fusion

The global prior reliability of modality `m` is obtained only from validation data:

```text
r_m = 1 / (MAE_m,val^2 + epsilon)
```

The local network produces nonnegative sample-specific reliability `q_m(t,l)`. The final weights are:

```text
w_m(t,l) = q_m(t,l) r_m / sum_j q_j(t,l) r_j
mu(t,l) = w_c mu_c + w_v mu_v
```

No test labels are used. The global prior prevents an unstable local gate from ignoring the consistently stronger modality, while local reliability permits adaptation to a particular NLoS link.

## 4. Predictive uncertainty

The final scale combines learned aleatoric uncertainty and expert disagreement:

```text
sigma_total^2 = sigma_local^2 + [0.25 |mu_c - mu_v|]^2
```

The disagreement term is an epistemic proxy: when the two independently trained experts disagree, the observation should have less influence on tracking.

## 5. Adaptive Particle Filter

For a particle position `x`, the likelihood is:

```text
p(z_l | x) = (1-epsilon) StudentT(z_l - tau_l(x); sigma_l, nu)
             + epsilon StudentT(z_l - tau_l(x); sigma_out, 3)
```

A link with large predicted uncertainty contributes a broader likelihood and cannot dominate reliable links.

## 6. Scientific hypothesis

Compared with either paper expert and with equal-link PF, U-FusePF should preserve LoS accuracy and reduce P90/RMSE under NLoS, outliers and dropout. The hypothesis is rejected if improvements are not stable across cases and seeds or if uncertainty cannot identify corrupted links above chance.
