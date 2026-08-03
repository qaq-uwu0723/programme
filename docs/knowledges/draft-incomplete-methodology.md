(Updated from your current draft; no benchmark-metric details are introduced here, as requested.) 

## Methodology

Industrial control system (ICS) telemetry is intrinsically **mixed-type** and **mechanistically heterogeneous**: continuous process trajectories (e.g., sensor and actuator signals) coexist with discrete supervisory states (e.g., modes, alarms, interlocks), and the underlying generating mechanisms range from physical inertia to program-driven step logic. This heterogeneity is not cosmetic—it directly affects what “realistic” synthesis means, because a generator must jointly satisfy (i) temporal coherence, (ii) distributional fidelity, and (iii) discrete semantic validity (i.e., every discrete output must belong to its legal vocabulary by construction). These properties are emphasized broadly in operational-technology security guidance and ICS engineering practice, where state logic and physical dynamics are tightly coupled. [12]

We formalize each training instance as a fixed-length window of length (L), consisting of (i) continuous channels (X\in\mathbb{R}^{L\times d_c}) and (ii) discrete channels (Y={y^{(j)}*{1:L}}*{j=1}^{d_d}), where each discrete variable (y^{(j)}_t\in\mathcal{V}_j) belongs to a finite vocabulary (\mathcal{V}_j). Our objective is to learn a generator that produces synthetic ((\hat{X},\hat{Y})) that are simultaneously temporally coherent and distributionally faithful, while also ensuring (\hat{y}^{(j)}_t\in\mathcal{V}_j) for all (j,t) by construction (rather than via post-hoc rounding or thresholding).

A key empirical and methodological tension in ICS synthesis is that *temporal realism* and *marginal/distributional realism* can compete when optimized monolithically: sequence models trained primarily for regression often over-smooth heavy tails and intermittent bursts, while purely distribution-matching objectives can erode long-range structure. Diffusion models provide a principled route to rich distribution modeling through iterative denoising, but they do not, by themselves, resolve (i) the need for a stable low-frequency temporal scaffold, nor (ii) the discrete legality constraints for supervisory variables. [2,8] Recent time-series diffusion work further suggests that separating coarse structure from stochastic refinement can be an effective inductive bias for long-horizon realism. [6,7]

Motivated by these considerations, we propose **Mask-DDPM**, organized in the following order:

1. **Transformer trend module**: learns the dominant temporal backbone of continuous dynamics via attention-based sequence modeling. [1]
2. **Residual DDPM for continuous variables**: models distributional detail as stochastic residual structure conditioned on the learned trend. [2,6]
3. **Masked diffusion for discrete variables**: generates discrete ICS states with an absorbing/masking corruption process and categorical reconstruction. [3,4]
4. **Type-aware decomposition**: a **type-aware factorization and routing layer** that assigns variables to the most appropriate modeling mechanism and enforces deterministic constraints where warranted.

This ordering is intentional. The trend module establishes a macro-temporal scaffold; residual diffusion then concentrates capacity on micro-structure and marginal fidelity; masked diffusion provides a native mechanism for discrete legality; and the type-aware layer operationalizes the observation that not all ICS variables should be modeled with the same stochastic mechanism. Importantly, while diffusion-based generation for ICS telemetry has begun to emerge, existing approaches remain limited and typically emphasize continuous synthesis or augmentation; in contrast, our pipeline integrates (i) a Transformer-conditioned residual diffusion backbone, (ii) a discrete masked-diffusion branch, and (iii) explicit type-aware routing for heterogeneous variable mechanisms within a single coherent generator. [10,11]

---

## Transformer trend module for continuous dynamics

We instantiate the temporal backbone as a **causal Transformer** trend extractor, leveraging self-attention’s ability to represent long-range dependencies and cross-channel interactions without recurrence. [1] Compared with recurrent trend extractors (e.g., GRU-style backbones), a Transformer trend module offers a direct mechanism to model delayed effects and multivariate coupling—common in ICS, where control actions may influence downstream sensors with nontrivial lags and regime-dependent propagation. [1,12] Crucially, in our design the Transformer is *not* asked to be the entire generator; instead, it serves a deliberately restricted role: providing a stable, temporally coherent conditioning signal that later stochastic components refine.

For continuous channels (X), we posit an additive decomposition
[
X = S + R,
]
where (S\in\mathbb{R}^{L\times d_c}) is a smooth **trend** capturing predictable temporal evolution and (R\in\mathbb{R}^{L\times d_c}) is a **residual** capturing distributional detail (e.g., bursts, heavy tails, local fluctuations) that is difficult to represent robustly with a purely regression-oriented temporal objective. This separation reflects an explicit *division of labor*: the trend module prioritizes temporal coherence, while diffusion (introduced next) targets distributional realism at the residual level—a strategy aligned with “predict-then-refine” perspectives in time-series diffusion modeling. [6,7]

We parameterize the trend (S) using a causal Transformer (f_{\phi}). With teacher forcing, we train (f_{\phi}) to predict the next-step trend from past observations:
[
\hat{S}*{t+1} = f*{\phi}(X_{1:t}), \qquad t=1,\dots,L-1,
]
using the mean-squared error objective
[
\mathcal{L}*{\text{trend}}(\phi)=\frac{1}{(L-1)d_c}\sum*{t=1}^{L-1}\left| \hat{S}*{t+1} - X*{t+1}\right|_2^2.
]
At inference, we roll out the Transformer autoregressively to obtain (\hat{S}), and define the residual target for diffusion as (R = X - \hat{S}). This setup intentionally “locks in” a coherent low-frequency scaffold before any stochastic refinement is applied, thereby reducing the burden on downstream diffusion modules to simultaneously learn both long-range structure and marginal detail. In this sense, our use of Transformers is distinctive: it is a *conditioning-first* temporal backbone designed to stabilize mixed-type diffusion synthesis in ICS, rather than an end-to-end monolithic generator. [1,6,10]

---

## DDPM for continuous residual generation

We model the residual (R) with a denoising diffusion probabilistic model (DDPM) conditioned on the trend (\hat{S}). [2] Diffusion models learn complex data distributions by inverting a tractable noising process through iterative denoising, and have proven effective at capturing multimodality and heavy-tailed structure that is often attenuated by purely regression-based sequence models. [2,8] Conditioning the diffusion model on (\hat{S}) is central: it prevents the denoiser from re-learning the low-frequency scaffold and focuses capacity on residual micro-structure, mirroring the broader principle that diffusion excels as a distributional corrector when a reasonable coarse structure is available. [6,7]

Let (K) denote the number of diffusion steps, with a noise schedule ({\beta_k}_{k=1}^K), (\alpha_k = 1-\beta_k), and (\bar{\alpha}*k=\prod*{i=1}^k \alpha_i). The forward corruption process is
[
q(r_k\mid r_0)=\mathcal{N}!\left(\sqrt{\bar{\alpha}_k},r_0,\ (1-\bar{\alpha}_k)\mathbf{I}\right),
]
equivalently,
[
r_k = \sqrt{\bar{\alpha}_k},r_0 + \sqrt{1-\bar{\alpha}_k},\epsilon,\qquad \epsilon\sim\mathcal{N}(0,\mathbf{I}),
]
where (r_0\equiv R) and (r_k) is the noised residual at step (k).

The learned reverse process is parameterized as
[
p_{\theta}(r_{k-1}\mid r_k,\hat{S})=\mathcal{N}!\left(\mu_{\theta}(r_k,k,\hat{S}),\ \Sigma(k)\right),
]
where (\mu_\theta) is implemented by a **Transformer denoiser** that consumes (i) the noised residual (r_k), (ii) a timestep embedding for (k), and (iii) conditioning features derived from (\hat{S}). This denoiser architecture is consistent with the growing use of attention-based denoisers for long-context time-series diffusion, while our key methodological emphasis is the *trend-conditioned residual* factorization as the object of diffusion learning. [2,7]

We train the denoiser using the standard DDPM (\epsilon)-prediction objective:
[
\mathcal{L}*{\text{cont}}(\theta)
= \mathbb{E}*{k,r_0,\epsilon}!\left[
\left|
\epsilon - \epsilon_{\theta}(r_k,k,\hat{S})
\right|*2^2
\right].
]
Because diffusion optimization can exhibit timestep imbalance (i.e., some timesteps dominate gradients), we optionally apply an SNR-based reweighting consistent with Min-SNR training:
[
\mathcal{L}^{\text{snr}}*{\text{cont}}(\theta)
= \mathbb{E}*{k,r_0,\epsilon}!\left[
w_k\left|
\epsilon - \epsilon*{\theta}(r_k,k,\hat{S})
\right|_2^2
\right],
\qquad
w_k=\frac{\min(\mathrm{SNR}_k,\gamma)}{\mathrm{SNR}_k},
]
where (\mathrm{SNR}_k=\bar{\alpha}_k/(1-\bar{\alpha}_k)) and (\gamma>0) is a cap parameter. [5]

After sampling (\hat{R}) by reverse diffusion, we reconstruct the continuous output as
[
\hat{X} = \hat{S} + \hat{R}.
]
Overall, the DDPM component serves as a **distributional corrector** on top of a temporally coherent backbone, which is particularly suited to ICS where low-frequency dynamics are strong and persistent but fine-scale variability (including bursts and regime-conditioned noise) remains important for realism. Relative to prior ICS diffusion efforts that primarily focus on continuous augmentation, our formulation elevates *trend-conditioned residual diffusion* as a modular mechanism for disentangling temporal structure from distributional refinement. [10,11]

---

## Masked diffusion for discrete ICS variables

Discrete ICS variables must remain categorical, making Gaussian diffusion inappropriate for supervisory states and mode-like channels. While one can attempt continuous relaxations or post-hoc discretization, such strategies risk producing semantically invalid intermediate states (e.g., “in-between” modes) and can distort the discrete marginal distribution. Discrete-state diffusion provides a principled alternative by defining a valid corruption process directly on categorical variables. [3,4] In the ICS setting, this is not a secondary detail: supervisory tags often encode control logic boundaries (modes, alarms, interlocks) that must remain within a finite vocabulary to preserve semantic correctness. [12]

We therefore adopt **masked (absorbing) diffusion** for discrete channels, where corruption replaces tokens with a special (\texttt{[MASK]}) symbol according to a schedule. [4] For each variable (j), define a masking schedule ({m_k}_{k=1}^K) (with (m_k\in[0,1]) increasing in (k)). The forward corruption process is
[
q(y^{(j)}_k \mid y^{(j)}_0)=
\begin{cases}
y^{(j)}*0, & \text{with probability } 1-m_k,\
\texttt{[MASK]}, & \text{with probability } m_k,
\end{cases}
]
applied independently across (j) and (t). Let (\mathcal{M}) denote the set of masked positions at step (k). The denoiser (h*{\psi}) predicts a categorical distribution over (\mathcal{V}*j) for each masked token, conditioned on (i) the corrupted discrete sequence, (ii) the diffusion step (k), and (iii) continuous context. Concretely, we condition on (\hat{S}) and (optionally) (\hat{X}) to couple supervisory reconstruction to the underlying continuous dynamics:
[
p*{\psi}!\left(y^{(j)}*0 \mid y_k, k, \hat{S}, \hat{X}\right)
= h*{\psi}(y_k,k,\hat{S},\hat{X}).
]
This conditioning choice is motivated by the fact that many discrete ICS states are not standalone—they are functions of regimes, thresholds, and procedural phases that manifest in continuous channels. [12]

Training uses a categorical denoising objective:
[
\mathcal{L}*{\text{disc}}(\psi)
= \mathbb{E}*{k}!\left[
\frac{1}{|\mathcal{M}|}
\sum_{(j,t)\in\mathcal{M}}
\mathrm{CE}!\left(
h_{\psi}(y_k,k,\hat{S},\hat{X})*{j,t},
y^{(j)}*{0,t}
\right)
\right],
]
where (\mathrm{CE}(\cdot,\cdot)) is cross-entropy. At sampling time, we initialize all discrete tokens as (\texttt{[MASK]}) and iteratively unmask them using the learned conditionals, ensuring that every output token lies in its legal vocabulary by construction. This discrete branch is a key differentiator of our pipeline: unlike typical continuous-only diffusion augmentation in ICS, we integrate masked diffusion as a first-class mechanism for supervisory-variable legality within the same end-to-end synthesis workflow. [4,10]

---

## Type-aware decomposition as a performance refinement layer

Even with a trend-conditioned residual DDPM and a discrete masked-diffusion branch, a single uniform modeling treatment can remain suboptimal because ICS variables are generated by qualitatively different mechanisms. For example, program-driven setpoints exhibit step-and-dwell dynamics; controller outputs follow control laws conditioned on process feedback; actuator positions may show saturation and dwell; and some “derived tags” are deterministic functions of other channels. Treating all channels as if they were exchangeable stochastic processes can misallocate model capacity and induce systematic error concentration on a small subset of mechanistically distinct variables. [12]

We therefore introduce a **type-aware decomposition** that formalizes this heterogeneity as a routing and constraint layer. Let (\tau(i)\in{1,\dots,6}) assign each variable (i) to a type class. The type assignment can be initialized from domain semantics (tag metadata, value domains, and engineering meaning), and subsequently refined via an error-attribution workflow described in the Benchmark section.  Importantly, this refinement does **not** change the core diffusion backbone; it changes *which mechanism is responsible for which variable*, thereby aligning inductive bias with variable-generating mechanism while preserving overall coherence.

We use the following taxonomy:

* **Type 1 (program-driven / setpoint-like):** externally commanded, step-and-dwell variables. These variables can be treated as exogenous drivers (conditioning signals) or routed to specialized change-point / dwell-time models, rather than being forced into a smooth denoiser that may over-regularize step structure.
* **Type 2 (controller outputs):** continuous variables tightly coupled to feedback loops; these benefit from conditional modeling where the conditioning includes relevant process variables and commanded setpoints.
* **Type 3 (actuator states/positions):** often exhibit saturation, dwell, and rate limits; these may require stateful dynamics beyond generic residual diffusion, motivating either specialized conditional modules or additional inductive constraints.
* **Type 4 (process variables):** inertia-dominated continuous dynamics; these are the primary beneficiaries of the **Transformer trend + residual DDPM** pipeline. 
* **Type 5 (derived/deterministic variables):** algebraic or rule-based functions of other variables; we enforce deterministic reconstruction (\hat{x}^{(i)} = g_i(\hat{X},\hat{Y})) rather than learning a stochastic generator, improving logical consistency and sample efficiency. 
* **Type 6 (auxiliary/low-impact variables):** weakly coupled or sparse signals; we allow simplified modeling (e.g., calibrated marginals or lightweight temporal models) to avoid allocating diffusion capacity where it is not warranted. 

Type-aware decomposition improves synthesis quality through three mechanisms. First, it improves **capacity allocation** by preventing a small set of mechanistically atypical variables from dominating gradients and distorting the learned distribution for the majority class (typically Type 4). Second, it enables **constraint enforcement** by deterministically reconstructing Type 5 variables, preventing logically inconsistent samples that purely learned generators can produce. Third, it improves **mechanism alignment** by attaching inductive biases consistent with step/dwell or saturation behaviors where generic denoisers may implicitly favor smoothness. 

From a novelty standpoint, this layer is not merely an engineering “patch”; it is an explicit methodological statement that ICS synthesis benefits from **typed factorization**—a principle that has analogues in mixed-type generative modeling more broadly, but that remains underexplored in diffusion-based ICS telemetry synthesis. [9,10,12]

---

## Joint optimization and end-to-end sampling

We train the model in a staged manner consistent with the above factorization, which improves optimization stability and encourages each component to specialize in its intended role. Specifically: (i) we train the trend Transformer (f_{\phi}) to obtain (\hat{S}); (ii) we compute residual targets (R=X-\hat{S}) for the continuous variables routed to residual diffusion; (iii) we train the residual DDPM (p_{\theta}(R\mid \hat{S})) and masked diffusion model (p_{\psi}(Y\mid \text{masked}(Y), \hat{S}, \hat{X})); and (iv) we apply type-aware routing and deterministic reconstruction during sampling.  This staged strategy is aligned with the design goal of separating temporal scaffolding from distributional refinement, and it mirrors the broader intuition in time-series diffusion that decoupling coarse structure and stochastic detail can mitigate “structure vs. realism” conflicts. [6,7]

A simple combined objective is
[
\mathcal{L} = \lambda,\mathcal{L}*{\text{cont}} + (1-\lambda),\mathcal{L}*{\text{disc}},
]
with (\lambda\in[0,1]) controlling the balance between continuous and discrete learning. Type-aware routing determines which channels contribute to which loss and which are excluded in favor of deterministic reconstruction. In practice, this routing acts as a principled guardrail against negative transfer across variable mechanisms: channels that are best handled deterministically (Type 5) or by specialized drivers (Type 1/3, depending on configuration) are prevented from forcing the diffusion models into statistically incoherent compromises.

At inference time, generation follows the same structured order: (i) trend (\hat{S}) via the Transformer, (ii) residual (\hat{R}) via DDPM, (iii) discrete (\hat{Y}) via masked diffusion, and (iv) type-aware assembly with deterministic reconstruction for routed variables. This pipeline produces ((\hat{X},\hat{Y})) that are temporally coherent by construction (through (\hat{S})), distributionally expressive (through (\hat{R}) denoising), and discretely valid (through masked diffusion), while explicitly accounting for heterogeneous variable-generating mechanisms through type-aware routing. In combination, these choices constitute our central methodological contribution: a unified Transformer + mixed diffusion generator for ICS telemetry, augmented by typed factorization to align model capacity with domain mechanism. [2,4,10,12]

---

# References

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. *Attention Is All You Need.* NeurIPS, 2017. arXiv:1706.03762. ([arXiv][1])
[2] Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. arXiv:2006.11239. ([Proceedings of Machine Learning Research][2])
[3] Austin, J., Johnson, D. D., Ho, J., Tarlow, D., van den Berg, R. *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).* NeurIPS, 2021. arXiv:2107.03006. ([arXiv][3])
[4] Shi, J., Han, K., Wang, Z., Doucet, A., Titsias, M. K. *Simplified and Generalized Masked Diffusion for Discrete Data.* arXiv:2406.04329, 2024. ([arXiv][4])
[5] Hang, T., Wu, C., Zhang, H., et al. *Efficient Diffusion Training via Min-SNR Weighting Strategy.* arXiv:2303.09556, 2023. ([arXiv][5])
[6] Kollovieh, M., Ansari, A. F., Bohlke-Schneider, M., et al. *Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (TSDiff).* arXiv:2307.11494, 2023. ([arXiv][6])
[7] Sikder, M. F., Ramachandranpillai, R., Heintz, F. *TransFusion: Generating Long, High Fidelity Time Series using Diffusion Models with Transformers.* arXiv:2307.12667, 2023. ([arXiv][7])
[8] Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR, 2021. arXiv:2011.13456. ([arXiv][8])
[9] Zhang, H., Zhang, J., Li, J., et al. *TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation.* arXiv:2410.20626, 2024. ([arXiv][9])
[10] Yuan, H., Sha, K., Zhao, W. *CTU-DDPM: Conditional Transformer U-net DDPM for Industrial Control System Anomaly Data Augmentation.* ACM AICSS, 2025. DOI:10.1145/3776759.3776845.
[11] Sha, K., et al. *DDPM Fusing Mamba and Adaptive Attention: An Augmentation Method for Industrial Control Systems Anomaly Data.* SSRN, posted Jan 10, 2026. (SSRN 6055903). ([SSRN][10])
[12] NIST. *Guide to Operational Technology (OT) Security (SP 800-82r3).* 2023. ([NIST Computer Security Resource Center][11])

[1]: https://arxiv.org/abs/2209.15421 "https://arxiv.org/abs/2209.15421"
[2]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf "https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf"
[3]: https://arxiv.org/html/2209.15421v2 "https://arxiv.org/html/2209.15421v2"
[4]: https://arxiv.org/abs/2011.13456 "https://arxiv.org/abs/2011.13456"
[5]: https://arxiv.org/abs/2303.09556 "https://arxiv.org/abs/2303.09556"
[6]: https://arxiv.org/pdf/2401.03006 "https://arxiv.org/pdf/2401.03006"
[7]: https://arxiv.org/abs/2307.12667 "https://arxiv.org/abs/2307.12667"
[8]: https://arxiv.org/abs/2406.04329 "https://arxiv.org/abs/2406.04329"
[9]: https://arxiv.org/abs/2406.07524 "https://arxiv.org/abs/2406.07524"
[10]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6055903 "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6055903"
[11]: https://csrc.nist.gov/pubs/sp/800/82/r3/final "https://csrc.nist.gov/pubs/sp/800/82/r3/final"
