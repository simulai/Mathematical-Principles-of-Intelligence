# Latest Relevant arXiv Papers

*Last Updated: 2026-01-02 11:28:57*

This list is automatically generated using logic from `arXiv-mcp`.

## A Comprehensive Study of Deep Learning Model Fixing Approaches
- **Authors**: Hanmo You, Zan Wang, Zishuo Dong, Luanqi Mo, Jianjun Zhao, Junjie Chen
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23745](https://arxiv.org/abs/2512.23745)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23745v1 Announce Type: new 
Abstract: Deep Learning (DL) has been widely adopted in diverse industrial domains, including autonomous driving, intelligent healthcare, and aided programming. Like traditional software, DL systems are also prone to faults, whose malfunctioning may expose users to significant risks. Consequently, numerous approaches have been proposed to address these issues. In this paper, we conduct a large-scale empirical study on 16 state-of-the-art DL model fixing approaches, spanning model-level, layer-level, and neuron-level categories, to comprehensively evaluate their performance. We assess not only their fixing effectiveness (their primary purpose) but also their impact on other critical properties, such as robustness, fairness, and backward compatibility. To ensure comprehensive and fair evaluation, we employ a diverse set of datasets, model architectures, and application domains within a uniform experimental setup for experimentation. We summarize several key findings with implications for both industry and academia. For example, model-level approaches demonstrate superior fixing effectiveness compared to others. No single approach can achieve the best fixing performance while improving accuracy and maintaining all other properties. Thus, academia should prioritize research on mitigating these side effects. These insights highlight promising directions for future exploration in this field.

---

## Geometric Scaling of Bayesian Inference in LLMs
- **Authors**: Naman Aggarwal, Siddhartha R. Dalal, Vishal Misra
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23752](https://arxiv.org/abs/2512.23752)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.23752v1 Announce Type: cross 
Abstract: Recent work has shown that small transformers trained in controlled "wind-tunnel'' settings can implement exact Bayesian inference, and that their training dynamics produce a geometric substrate -- low-dimensional value manifolds and progressively orthogonal keys -- that encodes posterior structure. We investigate whether this geometric signature persists in production-grade language models. Across Pythia, Phi-2, Llama-3, and Mistral families, we find that last-layer value representations organize along a single dominant axis whose position strongly correlates with predictive entropy, and that domain-restricted prompts collapse this structure into the same low-dimensional manifolds observed in synthetic settings.
  To probe the role of this geometry, we perform targeted interventions on the entropy-aligned axis of Pythia-410M during in-context learning. Removing or perturbing this axis selectively disrupts the local uncertainty geometry, whereas matched random-axis interventions leave it intact. However, these single-layer manipulations do not produce proportionally specific degradation in Bayesian-like behavior, indicating that the geometry is a privileged readout of uncertainty rather than a singular computational bottleneck. Taken together, our results show that modern language models preserve the geometric substrate that enables Bayesian inference in wind tunnels, and organize their approximate Bayesian updates along this substrate.

---

## Generalized Regularized Evidential Deep Learning Models: Theory and Comprehensive Evaluation
- **Authors**: Deep Shankar Pandey, Hyomin Choi, Qi Yu
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23753](https://arxiv.org/abs/2512.23753)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23753v1 Announce Type: cross 
Abstract: Evidential deep learning (EDL) models, based on Subjective Logic, introduce a principled and computationally efficient way to make deterministic neural networks uncertainty-aware. The resulting evidential models can quantify fine-grained uncertainty using learned evidence. However, the Subjective-Logic framework constrains evidence to be non-negative, requiring specific activation functions whose geometric properties can induce activation-dependent learning-freeze behavior: a regime where gradients become extremely small for samples mapped into low-evidence regions. We theoretically characterize this behavior and analyze how different evidential activations influence learning dynamics. Building on this analysis, we design a general family of activation functions and corresponding evidential regularizers that provide an alternative pathway for consistent evidence updates across activation regimes. Extensive experiments on four benchmark classification problems (MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet), two few-shot classification problems, and blind face restoration problem empirically validate the developed theory and demonstrate the effectiveness of the proposed generalized regularized evidential models.

---

## A Granular Grassmannian Clustering Framework via the Schubert Variety of Best Fit
- **Authors**: Karim Salta, Michael Kirby, Chris Peterson
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23766](https://arxiv.org/abs/2512.23766)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.23766v1 Announce Type: new 
Abstract: In many classification and clustering tasks, it is useful to compute a geometric representative for a dataset or a cluster, such as a mean or median. When datasets are represented by subspaces, these representatives become points on the Grassmann or flag manifold, with distances induced by their geometry, often via principal angles. We introduce a subspace clustering algorithm that replaces subspace means with a trainable prototype defined as a Schubert Variety of Best Fit (SVBF) - a subspace that comes as close as possible to intersecting each cluster member in at least one fixed direction. Integrated in the Linde-Buzo-Grey (LBG) pipeline, this SVBF-LBG scheme yields improved cluster purity on synthetic, image, spectral, and video action data, while retaining the mathematical structure required for downstream analysis.

---

## TabMixNN: A Unified Deep Learning Framework for Structural Mixed Effects Modeling on Tabular Data
- **Authors**: Deniz Akdemir
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23787](https://arxiv.org/abs/2512.23787)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.23787v1 Announce Type: new 
Abstract: We present TabMixNN, a flexible PyTorch-based deep learning framework that synthesizes classical mixed-effects modeling with modern neural network architectures for tabular data analysis. TabMixNN addresses the growing need for methods that can handle hierarchical data structures while supporting diverse outcome types including regression, classification, and multitask learning. The framework implements a modular three-stage architecture: (1) a mixed-effects encoder with variational random effects and flexible covariance structures, (2) backbone architectures including Generalized Structural Equation Models (GSEM) and spatial-temporal manifold networks, and (3) outcome-specific prediction heads supporting multiple outcome families. Key innovations include an R-style formula interface for accessibility, support for directed acyclic graph (DAG) constraints for causal structure learning, Stochastic Partial Differential Equation (SPDE) kernels for spatial modeling, and comprehensive interpretability tools including SHAP values and variance decomposition. We demonstrate the framework's flexibility through applications to longitudinal data analysis, genomic prediction, and spatial-temporal modeling. TabMixNN provides a unified interface for researchers to leverage deep learning while maintaining the interpretability and theoretical grounding of classical mixed-effects models.

---

## Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding
- **Authors**: Yue Guan, Changming Yu, Shihan Fang, Weiming Hu, Zaifeng Pan, Zheng Wang, Zihan Liu, Yangjie Zhou, Yufei Ding, Minyi Guo, Jingwen Leng
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23858](https://arxiv.org/abs/2512.23858)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23858v1 Announce Type: new 
Abstract: Speculative decoding improves LLM inference by generating and verifying multiple tokens in parallel, but existing systems suffer from suboptimal performance due to a mismatch between dynamic speculation and static runtime assumptions. We present Yggdrasil, a co-designed system that enables latency-optimal speculative decoding through context-aware tree drafting and compiler-friendly execution. Yggdrasil introduces an equal-growth tree structure for static graph compatibility, a latency-aware optimization objective for draft selection, and stage-based scheduling to reduce overhead. Yggdrasil supports unmodified LLMs and achieves up to $3.98\times$ speedup over state-of-the-art baselines across multiple hardware setups.

---

## Probing the Limits of Compressive Memory: A Study of Infini-Attention in Small-Scale Pretraining
- **Authors**: Ruizhe Huang, Kexuan Zhang, Yihao Fang, Baifeng Yu
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23862](https://arxiv.org/abs/2512.23862)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23862v1 Announce Type: cross 
Abstract: This study investigates small-scale pretraining for Small Language Models (SLMs) to enable efficient use of limited data and compute, improve accessibility in low-resource settings and reduce costs. To enhance long-context extrapolation in compact models, we focus on Infini-attention, which builds a compressed memory from past segments while preserving local attention. In our work, we conduct an empirical study using 300M-parameter LLaMA models pretrained with Infini-attention. The model demonstrates training stability and outperforms the baseline in long-context retrieval. We identify the balance factor as a key part of the model performance, and we found that retrieval accuracy drops with repeated memory compressions over long sequences. Even so, Infini-attention still effectively compensates for the SLM's limited parameters. Particularly, despite performance degradation at a 16,384-token context, the Infini-attention model achieves up to 31% higher accuracy than the baseline. Our findings suggest that achieving robust long-context capability in SLMs benefits from architectural memory like Infini-attention.

---

## Improved Balanced Classification with Theoretically Grounded Loss Functions
- **Authors**: Corinna Cortes, Mehryar Mohri, Yutao Zhong
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23947](https://arxiv.org/abs/2512.23947)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23947v1 Announce Type: cross 
Abstract: The balanced loss is a widely adopted objective for multi-class classification under class imbalance. By assigning equal importance to all classes, regardless of their frequency, it promotes fairness and ensures that minority classes are not overlooked. However, directly minimizing the balanced classification loss is typically intractable, which makes the design of effective surrogate losses a central question. This paper introduces and studies two advanced surrogate loss families: Generalized Logit-Adjusted (GLA) loss functions and Generalized Class-Aware weighted (GCA) losses. GLA losses generalize Logit-Adjusted losses, which shift logits based on class priors, to the broader general cross-entropy loss family. GCA loss functions extend the standard class-weighted losses, which scale losses inversely by class frequency, by incorporating class-dependent confidence margins and extending them to the general cross-entropy family. We present a comprehensive theoretical analysis of consistency for both loss families. We show that GLA losses are Bayes-consistent, but only $H$-consistent for complete (i.e., unbounded) hypothesis sets. Moreover, their $H$-consistency bounds depend inversely on the minimum class probability, scaling at least as $1/\mathsf p_{\min}$. In contrast, GCA losses are $H$-consistent for any hypothesis set that is bounded or complete, with $H$-consistency bounds that scale more favorably as $1/\sqrt{\mathsf p_{\min}}$, offering significantly stronger theoretical guarantees in imbalanced settings. We report the results of experiments demonstrating that, empirically, both the GCA losses with calibrated class-dependent confidence margins and GLA losses can greatly outperform straightforward class-weighted losses as well as the LA losses. GLA generally performs slightly better in common benchmarks, whereas GCA exhibits a slight edge in highly imbalanced settings.

---

## DivQAT: Enhancing Robustness of Quantized Convolutional Neural Networks against Model Extraction Attacks
- **Authors**: Kacem Khaled, Felipe Gohring de Magalh\~aes, Gabriela Nicolescu
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23948](https://arxiv.org/abs/2512.23948)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23948v1 Announce Type: new 
Abstract: Convolutional Neural Networks (CNNs) and their quantized counterparts are vulnerable to extraction attacks, posing a significant threat of IP theft. Yet, the robustness of quantized models against these attacks is little studied compared to large models. Previous defenses propose to inject calculated noise into the prediction probabilities. However, these defenses are limited since they are not incorporated during the model design and are only added as an afterthought after training. Additionally, most defense techniques are computationally expensive and often have unrealistic assumptions about the victim model that are not feasible in edge device implementations and do not apply to quantized models. In this paper, we propose DivQAT, a novel algorithm to train quantized CNNs based on Quantization Aware Training (QAT) aiming to enhance their robustness against extraction attacks. To the best of our knowledge, our technique is the first to modify the quantization process to integrate a model extraction defense into the training process. Through empirical validation on benchmark vision datasets, we demonstrate the efficacy of our technique in defending against model extraction attacks without compromising model accuracy. Furthermore, combining our quantization technique with other defense mechanisms improves their effectiveness compared to traditional QAT.

---

## Information-Theoretic Quality Metric of Low-Dimensional Embeddings
- **Authors**: Sebasti\'an Guti\'errez-Bernal (Tecnol\'ogico de Monterrey, Monterrey, N.L., Mexico), Hector Medel Cobaxin (Tecnol\'ogico de Monterrey, Monterrey, N.L., Mexico), Abiel Galindo Gonz\'alez (Tecnol\'ogico de Monterrey, Monterrey, N.L., Mexico)
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.23981](https://arxiv.org/abs/2512.23981)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.23981v1 Announce Type: new 
Abstract: In this work we study the quality of low-dimensional embeddings from an explicitly information-theoretic perspective. We begin by noting that classical evaluation metrics such as stress, rank-based neighborhood criteria, or Local Procrustes quantify distortions in distances or in local geometries, but do not directly assess how much information is preserved when projecting high-dimensional data onto a lower-dimensional space. To address this limitation, we introduce the Entropy Rank Preservation Measure (ERPM), a local metric based on the Shannon entropy of the singular-value spectrum of neighborhood matrices and on the stable rank, which quantifies changes in uncertainty between the original representation and its reduced projection, providing neighborhood-level indicators and a global summary statistic. To validate the results of the metric, we compare its outcomes with the Mean Relative Rank Error (MRRE), which is distance-based, and with Local Procrustes, which is based on geometric properties, using a financial time series and a manifold commonly studied in the literature. We observe that distance-based criteria exhibit very low correlation with geometric and spectral measures, while ERPM and Local Procrustes show strong average correlation but display significant discrepancies in local regimes, leading to the conclusion that ERPM complements existing metrics by identifying neighborhoods with severe information loss, thereby enabling a more comprehensive assessment of embeddings, particularly in information-sensitive applications such as the construction of early-warning indicators.

---

## Autoregressivity in the Latent Space of a GP-VAE Language Model: An Empirical Ablation Study
- **Authors**: Yves Ruffenach
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24102](https://arxiv.org/abs/2512.24102)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24102v1 Announce Type: new 
Abstract: This paper provides an ablation-based analysis of latent autoregression in GP-VAE models, building upon our previous work introducing the architecture. Language models typically rely on an autoregressive factorization over tokens. In contrast, our prior work proposed shifting sequential structure to the latent space through a causal Gaussian process, while using a non-autoregressive decoder. Here, we conduct a systematic ablation study of the role played by latent autoregression. We compare (i) a full GP-VAE model with autoregressive latent dynamics, (ii) a non-autoregressive ablation in which latent variables are independent, and (iii) a standard token-level autoregressive Transformer. Our results show that, within the considered regime (medium-scale corpora and short training contexts), latent autoregression induces latent trajectories that are significantly more compatible with the Gaussian-process prior and exhibit greater long-horizon stability. In contrast, removing autoregression leads to degraded latent structure and unstable long-range behavior. These findings highlight the role of latent autoregression as an effective mechanism for organizing long-range structure, while remaining complementary to token-level autoregressive modeling. They should be interpreted as an empirical analysis of representational structure rather than as a proposal for a new architecture.

---

## Enhancing LLM Planning Capabilities through Intrinsic Self-Critique
- **Authors**: Bernd Bohnet, Pierre-Alexandre Kamienny, Hanie Sedghi, Dilan Gorur, Pranjal Awasthi, Aaron Parisi, Kevin Swersky, Rosanne Liu, Azade Nova, Noah Fiedel
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24103](https://arxiv.org/abs/2512.24103)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24103v1 Announce Type: cross 
Abstract: We demonstrate an approach for LLMs to critique their \emph{own} answers with the goal of enhancing their performance that leads to significant improvements over established planning benchmarks. Despite the findings of earlier research that has cast doubt on the effectiveness of LLMs leveraging self critique methods, we show significant performance gains on planning datasets in the Blocksworld domain through intrinsic self-critique, without external source such as a verifier. We also demonstrate similar improvements on Logistics and Mini-grid datasets, exceeding strong baseline accuracies. We employ a few-shot learning technique and progressively extend it to a many-shot approach as our base method and demonstrate that it is possible to gain substantial improvement on top of this already competitive approach by employing an iterative process for correction and refinement. We illustrate how self-critique can significantly boost planning performance. Our empirical results present new state-of-the-art on the class of models considered, namely LLM model checkpoints from October 2024. Our primary focus lies on the method itself, demonstrating intrinsic self-improvement capabilities that are applicable regardless of the specific model version, and we believe that applying our method to more complex search techniques and more capable models will lead to even better performance.

---

## Paired Seed Evaluation: Statistical Reliability for Learning-Based Simulators
- **Authors**: Udit Sharma
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24145](https://arxiv.org/abs/2512.24145)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24145v1 Announce Type: cross 
Abstract: Machine learning systems appear stochastic but are deterministically random, as seeded pseudorandom number generators produce identical realisations across executions. Learning-based simulators are widely used to compare algorithms, design choices, and interventions under such dynamics, yet evaluation outcomes often exhibit high variance due to random initialisation and learning stochasticity. We analyse the statistical structure of comparative evaluation in these settings and show that standard independent evaluation designs fail to exploit shared sources of randomness across alternatives. We formalise a paired seed evaluation design in which competing systems are evaluated under identical random seeds, inducing matched realisations of stochastic components and strict variance reduction whenever outcomes are positively correlated at the seed level. This yields tighter confidence intervals, higher statistical power, and effective sample size gains at fixed computational budgets. Empirically, seed-level correlations are typically large and positive, producing order-of-magnitude efficiency gains. Paired seed evaluation is weakly dominant in practice, improving statistical reliability when correlation is present and reducing to independent evaluation without loss of validity when it is not.

---

## Empower Low-Altitude Economy: A Reliability-Aware Dynamic Weighting Allocation for Multi-modal UAV Beam Prediction
- **Authors**: Haojin Li, Anbang Zhang, Chen Sun, Chenyuan Feng, Kaiqian Qu, Tony Q. S. Quek, Haijun Zhang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24324](https://arxiv.org/abs/2512.24324)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24324v1 Announce Type: cross 
Abstract: The low-altitude economy (LAE) is rapidly expanding driven by urban air mobility, logistics drones, and aerial sensing, while fast and accurate beam prediction in uncrewed aerial vehicles (UAVs) communications is crucial for achieving reliable connectivity. Current research is shifting from single-signal to multi-modal collaborative approaches. However, existing multi-modal methods mostly employ fixed or empirical weights, assuming equal reliability across modalities at any given moment. Indeed, the importance of different modalities fluctuates dramatically with UAV motion scenarios, and static weighting amplifies the negative impact of degraded modalities. Furthermore, modal mismatch and weak alignment further undermine cross-scenario generalization. To this end, we propose a reliability-aware dynamic weighting scheme applied to a semantic-aware multi-modal beam prediction framework, named SaM2B. Specifically, SaM2B leverages lightweight cues such as environmental visual, flight posture, and geospatial data to adaptively allocate contributions across modalities at different time points through reliability-aware dynamic weight updates. Moreover, by utilizing cross-modal contrastive learning, we align the "multi-source representation beam semantics" associated with specific beam information to a shared semantic space, thereby enhancing discriminative power and robustness under modal noise and distribution shifts. Experiments on real-world low-altitude UAV datasets show that SaM2B achieves more satisfactory results than baseline methods.

---

## Tubular Riemannian Laplace Approximations for Bayesian Neural Networks
- **Authors**: Rodrigo Pereira David
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24381](https://arxiv.org/abs/2512.24381)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24381v1 Announce Type: cross 
Abstract: Laplace approximations are among the simplest and most practical methods for approximate Bayesian inference in neural networks, yet their Euclidean formulation struggles with the highly anisotropic, curved loss surfaces and large symmetry groups that characterize modern deep models. Recent work has proposed Riemannian and geometric Gaussian approximations to adapt to this structure. Building on these ideas, we introduce the Tubular Riemannian Laplace (TRL) approximation. TRL explicitly models the posterior as a probabilistic tube that follows a low-loss valley induced by functional symmetries, using a Fisher/Gauss-Newton metric to separate prior-dominated tangential uncertainty from data-dominated transverse uncertainty. We interpret TRL as a scalable reparametrised Gaussian approximation that utilizes implicit curvature estimates to operate in high-dimensional parameter spaces. Our empirical evaluation on ResNet-18 (CIFAR-10 and CIFAR-100) demonstrates that TRL achieves excellent calibration, matching or exceeding the reliability of Deep Ensembles (in terms of ECE) while requiring only a fraction (1/5) of the training cost. TRL effectively bridges the gap between single-model efficiency and ensemble-grade reliability.

---

## HOLOGRAPH: Active Causal Discovery via Sheaf-Theoretic Alignment of Large Language Model Priors
- **Authors**: Hyunjun Kim
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24478](https://arxiv.org/abs/2512.24478)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.24478v1 Announce Type: cross 
Abstract: Causal discovery from observational data remains fundamentally limited by identifiability constraints. Recent work has explored leveraging Large Language Models (LLMs) as sources of prior causal knowledge, but existing approaches rely on heuristic integration that lacks theoretical grounding. We introduce HOLOGRAPH, a framework that formalizes LLM-guided causal discovery through sheaf theory--representing local causal beliefs as sections of a presheaf over variable subsets. Our key insight is that coherent global causal structure corresponds to the existence of a global section, while topological obstructions manifest as non-vanishing sheaf cohomology. We propose the Algebraic Latent Projection to handle hidden confounders and Natural Gradient Descent on the belief manifold for principled optimization. Experiments on synthetic and real-world benchmarks demonstrate that HOLOGRAPH provides rigorous mathematical foundations while achieving competitive performance on causal discovery tasks with 50-100 variables. Our sheaf-theoretic analysis reveals that while Identity, Transitivity, and Gluing axioms are satisfied to numerical precision (<10^{-6}), the Locality axiom fails for larger graphs, suggesting fundamental non-local coupling in latent variable projections. Code is available at [https://github.com/hyunjun1121/holograph](https://github.com/hyunjun1121/holograph).

---

## Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice
- **Authors**: Jiachen T. Wang, Tong Wu, Kaifeng Lyu, James Zou, Dawn Song, Ruoxi Jia, Prateek Mittal
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24503](https://arxiv.org/abs/2512.24503)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24503v1 Announce Type: cross 
Abstract: Data teams at frontier AI companies routinely train small proxy models to make critical decisions about pretraining data recipes for full-scale training runs. However, the community has a limited understanding of whether and when conclusions drawn from small-scale experiments reliably transfer to full-scale model training. In this work, we uncover a subtle yet critical issue in the standard experimental protocol for data recipe assessment: the use of identical small-scale model training configurations across all data recipes in the name of "fair" comparison. We show that the experiment conclusions about data quality can flip with even minor adjustments to training hyperparameters, as the optimal training configuration is inherently data-dependent. Moreover, this fixed-configuration protocol diverges from full-scale model development pipelines, where hyperparameter optimization is a standard step. Consequently, we posit that the objective of data recipe assessment should be to identify the recipe that yields the best performance under data-specific tuning. To mitigate the high cost of hyperparameter tuning, we introduce a simple patch to the evaluation protocol: using reduced learning rates for proxy model training. We show that this approach yields relative performance that strongly correlates with that of fully tuned large-scale LLM pretraining runs. Theoretically, we prove that for random-feature models, this approach preserves the ordering of datasets according to their optimal achievable loss. Empirically, we validate this approach across 23 data recipes covering four critical dimensions of data curation, demonstrating dramatic improvements in the reliability of small-scale experiments.

---

## CPR: Causal Physiological Representation Learning for Robust ECG Analysis under Distribution Shifts
- **Authors**: Shunbo Jia, Caizhi Liao
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24564](https://arxiv.org/abs/2512.24564)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24564v1 Announce Type: new 
Abstract: Deep learning models for Electrocardiogram (ECG) diagnosis have achieved remarkable accuracy but exhibit fragility against adversarial perturbations, particularly Smooth Adversarial Perturbations (SAP) that mimic biological morphology. Existing defenses face a critical dilemma: Adversarial Training (AT) provides robustness but incurs a prohibitive computational burden, while certified methods like Randomized Smoothing (RS) introduce significant inference latency, rendering them impractical for real-time clinical monitoring. We posit that this vulnerability stems from the models' reliance on non-robust spurious correlations rather than invariant pathological features. To address this, we propose Causal Physiological Representation Learning (CPR). Unlike standard denoising approaches that operate without semantic constraints, CPR incorporates a Physiological Structural Prior within a causal disentanglement framework. By modeling ECG generation via a Structural Causal Model (SCM), CPR enforces a structural intervention that strictly separates invariant pathological morphology (P-QRS-T complex) from non-causal artifacts. Empirical results on PTB-XL demonstrate that CPR significantly outperforms standard clinical preprocessing methods. Specifically, under SAP attacks, CPR achieves an F1 score of 0.632, surpassing Median Smoothing (0.541 F1) by 9.1%. Crucially, CPR matches the certified robustness of Randomized Smoothing while maintaining single-pass inference efficiency, offering a superior trade-off between robustness, efficiency, and clinical interpretability.

---

## Mobility-Assisted Decentralized Federated Learning: Convergence Analysis and A Data-Driven Approach
- **Authors**: Reza Jahani, Md Farhamdur Reza, Richeng Jin, Huaiyu Dai
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24694](https://arxiv.org/abs/2512.24694)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24694v1 Announce Type: new 
Abstract: Decentralized Federated Learning (DFL) has emerged as a privacy-preserving machine learning paradigm that enables collaborative training among users without relying on a central server. However, its performance often degrades significantly due to limited connectivity and data heterogeneity. As we move toward the next generation of wireless networks, mobility is increasingly embedded in many real-world applications. The user mobility, either natural or induced, enables clients to act as relays or bridges, thus enhancing information flow in sparse networks; however, its impact on DFL has been largely overlooked despite its potential. In this work, we systematically investigate the role of mobility in improving DFL performance. We first establish the convergence of DFL in sparse networks under user mobility and theoretically demonstrate that even random movement of a fraction of users can significantly boost performance. Building upon this insight, we propose a DFL framework that utilizes mobile users with induced mobility patterns, allowing them to exploit the knowledge of data distribution to determine their trajectories to enhance information propagation through the network. Through extensive experiments, we empirically confirm our theoretical findings, validate the superiority of our approach over baselines, and provide a comprehensive analysis of how various network parameters influence DFL performance in mobile networks.

---

## From Trial to Deployment: A SEM Analysis of Traveler Adoptions to Fully Operational Autonomous Taxis
- **Authors**: Yutong Cai, Hua Wang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.24767](https://arxiv.org/abs/2512.24767)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24767v1 Announce Type: new 
Abstract: Autonomous taxi services represent a transformative advancement in urban mobility, offering safety, efficiency, and round-the-clock operations. While existing literature has explored user acceptance of autonomous taxis through stated preference experiments and hypothetical scenarios, few studies have investigated actual user behavior based on operational AV services. This study addresses that gap by leveraging survey data from Wuhan, China, where Baidu's Apollo Robotaxi service operates at scale. We design a realistic survey incorporating actual service attributes and collect 336 valid responses from actual users. Using Structural Equation Modeling, we identify six latent psychological constructs, namely Trust \& Policy Support, Cost Sensitivity, Performance, Behavioral Intention, Lifestyle, and Education. Their influences on adoption behavior, measured by the selection frequency of autonomous taxis in ten scenarios, are examined and interpreted. Results show that Cost Sensitivity and Behavioral Intention are the strongest positive predictors of adoption, while other latent constructs play more nuanced roles. The model demonstrates strong goodness-of-fit across multiple indices. Our findings offer empirical evidence to support policymaking, fare design, and public outreach strategies for scaling autonomous taxis deployments in real-world urban settings.

---

## ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning
- **Authors**: Timo Kaufmann, Yannick Metz, Daniel Keim, Eyke H\"ullermeier
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.25023](https://arxiv.org/abs/2512.25023)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.25023v1 Announce Type: new 
Abstract: Binary choices, as often used for reinforcement learning from human feedback (RLHF), convey only the direction of a preference. A person may choose apples over oranges and bananas over grapes, but which preference is stronger? Strength is crucial for decision-making under uncertainty and generalization of preference models, but hard to measure reliably. Metadata such as response times and inter-annotator agreement can serve as proxies for strength, but are often noisy and confounded. We propose ResponseRank to address the challenge of learning from noisy strength signals. Our method uses relative differences in proxy signals to rank responses to pairwise comparisons by their inferred preference strength. To control for systemic variation, we compare signals only locally within carefully constructed strata. This enables robust learning of utility differences consistent with strength-derived rankings while making minimal assumptions about the strength signal. Our contributions are threefold: (1) ResponseRank, a novel method that robustly learns preference strength by leveraging locally valid relative strength signals; (2) empirical evidence of improved sample efficiency and robustness across diverse tasks: synthetic preference learning (with simulated response times), language modeling (with annotator agreement), and RL control tasks (with simulated episode returns); and (3) the Pearson Distance Correlation (PDC), a novel metric that isolates cardinal utility learning from ordinal accuracy.

---

## On the geometry and topology of representations: the manifolds of modular addition
- **Authors**: Gabriela Moisescu-Pareja, Gavin McCracken, Harley Wiltzer, Vincent L\'etourneau, Colin Daniels, Doina Precup, Jonathan Love
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.LG
- **Link**: [https://arxiv.org/abs/2512.25060](https://arxiv.org/abs/2512.25060)
- **Matched Keywords**: Manifold

**Abstract**:
arXiv:2512.25060v1 Announce Type: new 
Abstract: The Clock and Pizza interpretations, associated with architectures differing in either uniform or learnable attention, were introduced to argue that different architectural designs can yield distinct circuits for modular addition. In this work, we show that this is not the case, and that both uniform attention and trainable attention architectures implement the same algorithm via topologically and geometrically equivalent representations. Our methodology goes beyond the interpretation of individual neurons and weights. Instead, we identify all of the neurons corresponding to each learned representation and then study the collective group of neurons as one entity. This method reveals that each learned representation is a manifold that we can study utilizing tools from topology. Based on this insight, we can statistically analyze the learned representations across hundreds of circuits to demonstrate the similarity between learned modular addition circuits that arise naturally from common deep learning paradigms.

---

## Implicit geometric regularization in flow matching via density weighted Stein operators
- **Authors**: Shinto Eguchi
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: stat.ML
- **Link**: [https://arxiv.org/abs/2512.23956](https://arxiv.org/abs/2512.23956)
- **Matched Keywords**: Manifold, MPI

**Abstract**:
arXiv:2512.23956v1 Announce Type: new 
Abstract: Flow Matching (FM) has emerged as a powerful paradigm for continuous normalizing flows, yet standard FM implicitly performs an unweighted $L^2$ regression over the entire ambient space. In high dimensions, this leads to a fundamental inefficiency: the vast majority of the integration domain consists of low-density ``void'' regions where the target velocity fields are often chaotic or ill-defined. In this paper, we propose {$\gamma$-Flow Matching ($\gamma$-FM)}, a density-weighted variant that aligns the regression geometry with the underlying probability flow. While density weighting is desirable, naive implementations would require evaluating the intractable target density. We circumvent this by introducing a Dynamic Density-Weighting strategy that estimates the \emph{target} density directly from training particles. This approach allows us to dynamically downweight the regression loss in void regions without compromising the simulation-free nature of FM. Theoretically, we establish that $\gamma$-FM minimizes the transport cost on a statistical manifold endowed with the $\gamma$-Stein metric. Spectral analysis further suggests that this geometry induces an implicit Sobolev regularization, effectively damping high-frequency oscillations in void regions. Empirically, $\gamma$-FM significantly improves vector field smoothness and sampling efficiency on high-dimensional latent datasets, while demonstrating intrinsic robustness to outliers.

---

## Fundamental limits for weighted empirical approximations of tilted distributions
- **Authors**: Sarvesh Ravichandran Iyer, Himadri Mandal, Dhruman Gupta, Rushil Gupta, Agniv Bandhyopadhyay, Achal Bassamboo, Varun Gupta, Sandeep Juneja
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: math.ST
- **Link**: [https://arxiv.org/abs/2512.23979](https://arxiv.org/abs/2512.23979)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23979v1 Announce Type: cross 
Abstract: Consider the task of generating samples from a tilted distribution of a random vector whose underlying distribution is unknown, but samples from it are available. This finds applications in fields such as finance and climate science, and in rare event simulation. In this article, we discuss the asymptotic efficiency of a self-normalized importance sampler of the tilted distribution. We provide a sharp characterization of its accuracy, given the number of samples and the degree of tilt. Our findings reveal a surprising dichotomy: while the number of samples needed to accurately tilt a bounded random vector increases polynomially in the tilt amount, it increases at a super polynomial rate for unbounded distributions.

---

## RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress
- **Authors**: Ruixuan Huang, Qingyue Wang, Hantao Huang, Yudong Gao, Dong Chen, Shuai Wang, Wei Wang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CR
- **Link**: [https://arxiv.org/abs/2512.23995](https://arxiv.org/abs/2512.23995)
- **Matched Keywords**: MoE

**Abstract**:
arXiv:2512.23995v1 Announce Type: cross 
Abstract: Mixture-of-Experts architectures have become the standard for scaling large language models due to their superior parameter efficiency. To accommodate the growing number of experts in practice, modern inference systems commonly adopt expert parallelism to distribute experts across devices. However, the absence of explicit load balancing constraints during inference allows adversarial inputs to trigger severe routing concentration. We demonstrate that out-of-distribution prompts can manipulate the routing strategy such that all tokens are consistently routed to the same set of top-$k$ experts, which creates computational bottlenecks on certain devices while forcing others to idle. This converts an efficiency mechanism into a denial-of-service attack vector, leading to violations of service-level agreements for time to first token. We propose RepetitionCurse, a low-cost black-box strategy to exploit this vulnerability. By identifying a universal flaw in MoE router behavior, RepetitionCurse constructs adversarial prompts using simple repetitive token patterns in a model-agnostic manner. On widely deployed MoE models like Mixtral-8x7B, our method increases end-to-end inference latency by 3.063x, degrading service availability significantly.

---

## Score-based sampling without diffusions: Guidance from a simple and modular scheme
- **Authors**: M. J. Wainwright
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: math.ST
- **Link**: [https://arxiv.org/abs/2512.24152](https://arxiv.org/abs/2512.24152)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24152v1 Announce Type: cross 
Abstract: Sampling based on score diffusions has led to striking empirical results, and has attracted considerable attention from various research communities. It depends on availability of (approximate) Stein score functions for various levels of additive noise. We describe and analyze a modular scheme that reduces score-based sampling to solving a short sequence of ``nice'' sampling problems, for which high-accuracy samplers are known. We show how to design forward trajectories such that both (a) the terminal distribution, and (b) each of the backward conditional distribution is defined by a strongly log concave (SLC) distribution. This modular reduction allows us to exploit \emph{any} SLC sampling algorithm in order to traverse the backwards path, and we establish novel guarantees with short proofs for both uni-modal and multi-modal densities. The use of high-accuracy routines yields $\varepsilon$-accurate answers, in either KL or Wasserstein distances, with polynomial dependence on $\log(1/\varepsilon)$ and $\sqrt{d}$ dependence on the dimension.

---

## Joint Selection for Large-Scale Pre-Training Data via Policy Gradient-based Mask Learning
- **Authors**: Ziqing Fan, Yuqiao Xian, Yan Sun, Li Shen
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24265](https://arxiv.org/abs/2512.24265)
- **Matched Keywords**: MoE, MPI

**Abstract**:
arXiv:2512.24265v1 Announce Type: new 
Abstract: A fine-grained data recipe is crucial for pre-training large language models, as it can significantly enhance training efficiency and model performance. One important ingredient in the recipe is to select samples based on scores produced by defined rules, LLM judgment, or statistical information in embeddings, which can be roughly categorized into quality and diversity metrics. Due to the high computational cost when applied to trillion-scale token pre-training datasets such as FineWeb and DCLM, these two or more types of metrics are rarely considered jointly in a single selection process. However, in our empirical study, selecting samples based on quality metrics exhibit severe diminishing returns during long-term pre-training, while selecting on diversity metrics removes too many valuable high-quality samples, both of which limit pre-trained LLMs' capabilities. Therefore, we introduce DATAMASK, a novel and efficient joint learning framework designed for large-scale pre-training data selection that can simultaneously optimize multiple types of metrics in a unified process, with this study focusing specifically on quality and diversity metrics. DATAMASK approaches the selection process as a mask learning problem, involving iterative sampling of data masks, computation of policy gradients based on predefined objectives with sampled masks, and updating of mask sampling logits. Through policy gradient-based optimization and various acceleration enhancements, it significantly reduces selection time by 98.9% compared to greedy algorithm, enabling our study to explore joint learning within trillion-scale tokens. With DATAMASK, we select a subset of about 10% from the 15 trillion-token FineWeb dataset, termed FineWeb-Mask. Evaluated across 12 diverse tasks, we achieves significant improvements of 3.2% on a 1.5B dense model and 1.9% on a 7B MoE model.

---

## Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features
- **Authors**: Theodore MacMillan, Nicholas T. Ouellette
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: physics.ao-ph
- **Link**: [https://arxiv.org/abs/2512.24440](https://arxiv.org/abs/2512.24440)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24440v1 Announce Type: cross 
Abstract: Large data-driven physics models like DeepMind's weather model GraphCast have empirically succeeded in parameterizing time operators for complex dynamical systems with an accuracy reaching or in some cases exceeding that of traditional physics-based solvers. Unfortunately, how these data-driven models perform computations is largely unknown and whether their internal representations are interpretable or physically consistent is an open question. Here, we adapt tools from interpretability research in Large Language Models to analyze intermediate computational layers in GraphCast, leveraging sparse autoencoders to discover interpretable features in the neuron space of the model. We uncover distinct features on a wide range of length and time scales that correspond to tropical cyclones, atmospheric rivers, diurnal and seasonal behavior, large-scale precipitation patterns, specific geographical coding, and sea-ice extent, among others. We further demonstrate how the precise abstraction of these features can be probed via interventions on the prediction steps of the model. As a case study, we sparsely modify a feature corresponding to tropical cyclones in GraphCast and observe interpretable and physically consistent modifications to evolving hurricanes. Such methods offer a window into the black-box behavior of data-driven physics models and are a step towards realizing their potential as trustworthy predictors and scientifically valuable tools for discovery.

---

## Robust Bayesian Dynamic Programming for On-policy Risk-sensitive Reinforcement Learning
- **Authors**: Shanyu Han, Yangbo He, Yang Liu
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: q-fin.RM
- **Link**: [https://arxiv.org/abs/2512.24580](https://arxiv.org/abs/2512.24580)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24580v1 Announce Type: cross 
Abstract: We propose a novel framework for risk-sensitive reinforcement learning (RSRL) that incorporates robustness against transition uncertainty. We define two distinct yet coupled risk measures: an inner risk measure addressing state and cost randomness and an outer risk measure capturing transition dynamics uncertainty. Our framework unifies and generalizes most existing RL frameworks by permitting general coherent risk measures for both inner and outer risk measures. Within this framework, we construct a risk-sensitive robust Markov decision process (RSRMDP), derive its Bellman equation, and provide error analysis under a given posterior distribution. We further develop a Bayesian Dynamic Programming (Bayesian DP) algorithm that alternates between posterior updates and value iteration. The approach employs an estimator for the risk-based Bellman operator that combines Monte Carlo sampling with convex optimization, for which we prove strong consistency guarantees. Furthermore, we demonstrate that the algorithm converges to a near-optimal policy in the training environment and analyze both the sample complexity and the computational complexity under the Dirichlet posterior and CVaR. Finally, we validate our approach through two numerical experiments. The results exhibit excellent convergence properties while providing intuitive demonstrations of its advantages in both risk-sensitivity and robustness. Empirically, we further demonstrate the advantages of the proposed algorithm through an application on option hedging.

---

## mHC: Manifold-Constrained Hyper-Connections
- **Authors**: Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880)
- **Matched Keywords**: Manifold, MPI

**Abstract**:
arXiv:2512.24880v1 Announce Type: new 
Abstract: Recently, studies exemplified by Hyper-Connections (HC) have extended the ubiquitous residual connection paradigm established over the past decade by expanding the residual stream width and diversifying connectivity patterns. While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead. To address these challenges, we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold to restore the identity mapping property, while incorporating rigorous infrastructure optimization to ensure efficiency. Empirical experiments demonstrate that mHC is effective for training at scale, offering tangible performance improvements and superior scalability. We anticipate that mHC, as a flexible and practical extension of HC, will contribute to a deeper understanding of topological architecture design and suggest promising directions for the evolution of foundational models.

---

## Are First-Order Diffusion Samplers Really Slower? A Fast Forward-Value Approach
- **Authors**: Yuchen Jiao, Na Li, Changxiao Cai, Gen Li
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: stat.ML
- **Link**: [https://arxiv.org/abs/2512.24927](https://arxiv.org/abs/2512.24927)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24927v1 Announce Type: new 
Abstract: Higher-order ODE solvers have become a standard tool for accelerating diffusion probabilistic model (DPM) sampling, motivating the widespread view that first-order methods are inherently slower and that increasing discretization order is the primary path to faster generation. This paper challenges this belief and revisits acceleration from a complementary angle: beyond solver order, the placement of DPM evaluations along the reverse-time dynamics can substantially affect sampling accuracy in the low-neural function evaluation (NFE) regime.
  We propose a novel training-free, first-order sampler whose leading discretization error has the opposite sign to that of DDIM. Algorithmically, the method approximates the forward-value evaluation via a cheap one-step lookahead predictor. We provide theoretical guarantees showing that the resulting sampler provably approximates the ideal forward-value trajectory while retaining first-order convergence. Empirically, across standard image generation benchmarks (CIFAR-10, ImageNet, FFHQ, and LSUN), the proposed sampler consistently improves sample quality under the same NFE budget and can be competitive with, and sometimes outperform, state-of-the-art higher-order samplers. Overall, the results suggest that the placement of DPM evaluations provides an additional and largely independent design angle for accelerating diffusion sampling.

---

## Evaluating the Reasoning Abilities of LLMs on Underrepresented Mathematics Competition Problems
- **Authors**: Samuel Golladay, Majid Bani-Yaghoub
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.AI
- **Link**: [https://arxiv.org/abs/2512.24505](https://arxiv.org/abs/2512.24505)
- **Matched Keywords**: DeepSeek

**Abstract**:
arXiv:2512.24505v1 Announce Type: new 
Abstract: Understanding the limitations of Large Language Models, or LLMs, in mathematical reasoning has been the focus of several recent studies. However, the majority of these studies use the same datasets for benchmarking, which limits the generalizability of their findings and may not fully capture the diverse challenges present in mathematical tasks. The purpose of the present study is to analyze the performance of LLMs on underrepresented mathematics competition problems. We prompted three leading LLMs, namely GPT-4o-mini, Gemini-2.0-Flash, and DeepSeek-V3, with the Missouri Collegiate Mathematics Competition problems in the areas of Calculus, Analytic Geometry, and Discrete Mathematics. The LLMs responses were then compared to the known correct solutions in order to determine the accuracy of the LLM for each problem domain. We also analyzed the LLMs reasoning to explore patterns in errors across problem types and models. DeepSeek-V3 has the best performance in all three categories of Calculus, Analytic Geometry, and Discrete Mathematics, both in reasoning and correct final answers. All three LLMs exhibited notably weak performance in Geometry. The majority of errors made by DeepSeek-V3 were attributed to computational and logical mistakes, whereas GPT-4o-mini frequently exhibited logical and approach-related errors. Gemini, on the other hand, tended to struggle with incomplete reasoning and drawing rushed conclusions. In conclusion, evaluating LLMs on underrepresented mathematics competition datasets can provide deeper insights into their distinct error patterns and highlight ongoing challenges in structured reasoning, particularly within the domain of Geometry.

---

## Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem
- **Authors**: Weixun Wang, XiaoXiao Xu, Wanhe An, Fangwen Dai, Wei Gao, Yancheng He, Ju Huang, Qiang Ji, Hanqi Jin, Xiaoyang Li, Yang Li, Zhongwen Li, Shirong Lin, Jiashun Liu, Zenan Liu, Tao Luo, Dilxat Muhtar, Yuanbin Qu, Jiaqiang Shi, Qinghui Sun, Yingshui Tan, Hao Tang, Runze Wang, Yi Wang, Zhaoguo Wang, Yanan Wu, Shaopan Xiong, Binchen Xu, Xander Xu, Yuchi Xu, Qipeng Zhang, Xixia Zhang, Haizhou Zhao, Jie Zhao, Shuaibing Zhao, Baihui Zheng, Jianhui Zheng, Suhang Zheng, Yanni Zhu, Mengze Cai, Kerui Cao, Xitong Chen, Yue Dai, Lifan Du, Tao Feng, Tao He, Jin Hu, Yijie Hu, Ziyu Jiang, Cheng Li, Xiang Li, Jing Liang, Chonghuan Liu, ZhenDong Liu, Haodong Mi, Yanhu Mo, Junjia Ni, Shixin Pei, Jingyu Shen, XiaoShuai Song, Cecilia Wang, Chaofan Wang, Kangyu Wang, Pei Wang, Tao Wang, Wei Wang, Ke Xiao, Mingyu Xu, Tiange Xu, Nan Ya, Siran Yang, Jianan Ye, Yaxing Zang, Duo Zhang, Junbo Zhang, Boren Zheng, Wanxi Deng, Ling Pan, Lin Qu, Wenbo Su, Jiamang Wang, Wei Wang, Hu Wei, Minggang Wu, Cheng Yu, Bing Zhao, Zhicheng Zheng, Bo Zheng
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.AI
- **Link**: [https://arxiv.org/abs/2512.24873](https://arxiv.org/abs/2512.24873)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24873v1 Announce Type: cross 
Abstract: Agentic crafting requires LLMs to operate in real-world environments over multiple turns by taking actions, observing outcomes, and iteratively refining artifacts. Despite its importance, the open-source community lacks a principled, end-to-end ecosystem to streamline agent development. We introduce the Agentic Learning Ecosystem (ALE), a foundational infrastructure that optimizes the production pipeline for agent LLMs. ALE consists of three components: ROLL, a post-training framework for weight optimization; ROCK, a sandbox environment manager for trajectory generation; and iFlow CLI, an agent framework for efficient context engineering. We release ROME (ROME is Obviously an Agentic Model), an open-source agent grounded by ALE and trained on over one million trajectories. Our approach includes data composition protocols for synthesizing complex behaviors and a novel policy optimization algorithm, Interaction-based Policy Alignment (IPA), which assigns credit over semantic interaction chunks rather than individual tokens to improve long-horizon training stability. Empirically, we evaluate ROME within a structured setting and introduce Terminal Bench Pro, a benchmark with improved scale and contamination control. ROME demonstrates strong performance across benchmarks like SWE-bench Verified and Terminal Bench, proving the effectiveness of the ALE infrastructure.

---

## State-of-the-art Small Language Coder Model: Mify-Coder
- **Authors**: Abhinav Parmar, Abhisek Panigrahi, Abhishek Kumar Dwivedi, Abhishek Bhattacharya, Adarsh Ramachandra, Aditya Choudhary, Aditya Garg, Aditya Raj, Alankrit Bhatt, Alpesh Yadav, Anant Vishnu, Ananthu Pillai, Ankush Kumar, Aryan Patnaik, Aswatha Narayanan S, Avanish Raj Singh, Bhavya Shree Gadda, Brijesh Pankajbhai Kachhadiya, Buggala Jahnavi, Chidurala Nithin Krishna, Chintan Shah, Chunduru Akshaya, Debarshi Banerjee, Debrup Dey, Deepa R., Deepika B G, Faiz ur Rahman, Gagan Gayari, Gudhi Jagadeesh Kumar Naidu, Gursimar Singh, Harshal Tyagi, Harshini K, James Mani Vathalloor, Jayarama Nettar, Jayashree Gajjam, Joe Walter Sugil George, Kamalakara Sri Krishna Tadepalli, Kamalkumar Rathinasamy, Karan Chaurasia, Karthikeyan S, Kashish Arora, Kaushal Desai, Khushboo Buwade, Kiran Manjrekar, Malikireddy Venkata Sai Likhitha, Manjunath A, Mitali Mahavir Bedmutha, Mohammed Rafee Tarafdar, Nikhil Tiwari, Nikitha K Gigi, Pavan Ravikumar, Pendyala Swarnanjali, Piyush Anand, Prakash Chandrasekar, Prasanna Bhalchandra Gawade, Prasanth Sivan, Preeti Khurana, Priyanshi Babbar, Rajab Ali Mondal, Rajesh Kumar Vissapragada, Rajeshwari Ganesan, Rajeswari Koppisetti, Ramjee R., Ramkumar Thiruppathisamy, Rani G. S., S Reka, Samarth Gupta, Sandeep Reddy Kothakota, Sarathy K, Sathyanarayana Sampath Kumar, Saurabh Kumar, Shashank Khasare, Shenbaga Devi Venkatesh Kumar, Shiva Rama Krishna Parvatham, Shoeb Shaikh, Shrishanmathi A, Shubham Pathak, Sree Samhita Koppaka, Sreenivasa Raghavan K S, Sreeram Venkatasubramanian, Suprabha Desai Bojja, Swetha R, Syed Ahmed, Chinmai Harshitha Thota, Tushar Yadav, Veeravelly Kusumitha, V V S S Prasanth Patnaik, Vidya Sri Sesetti, Vijayakeerthi K, Vikram Raj Bakshi, Vinay K K, Vinoth Kumar Loganathan, Vipin Tiwari, Vivek Kumar Shrivastav, V Venkata Sri Datta Charan, Wasim Akhtar Khan
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.SE
- **Link**: [https://arxiv.org/abs/2512.23747](https://arxiv.org/abs/2512.23747)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23747v1 Announce Type: cross 
Abstract: We present Mify-Coder, a 2.5B-parameter code model trained on 4.2T tokens using a compute-optimal strategy built on the Mify-2.5B foundation model. Mify-Coder achieves comparable accuracy and safety while significantly outperforming much larger baseline models on standard coding and function-calling benchmarks, demonstrating that compact models can match frontier-grade models in code generation and agent-driven workflows. Our training pipeline combines high-quality curated sources with synthetic data generated through agentically designed prompts, refined iteratively using enterprise-grade evaluation datasets. LLM-based quality filtering further enhances data density, enabling frugal yet effective training. Through disciplined exploration of CPT-SFT objectives, data mixtures, and sampling dynamics, we deliver frontier-grade code intelligence within a single continuous training trajectory. Empirical evidence shows that principled data and compute discipline allow smaller models to achieve competitive accuracy, efficiency, and safety compliance. Quantized variants of Mify-Coder enable deployment on standard desktop environments without requiring specialized hardware.

---

## Audited Skill-Graph Self-Improvement for Agentic LLMs via Verifiable Rewards, Experience Synthesis, and Continual Memory
- **Authors**: Ken Huang, Jerry Huang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CR
- **Link**: [https://arxiv.org/abs/2512.23760](https://arxiv.org/abs/2512.23760)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23760v1 Announce Type: cross 
Abstract: Reinforcement learning is increasingly used to transform large language models into agentic systems that act over long horizons, invoke tools, and manage memory under partial observability. While recent work has demonstrated performance gains through tool learning, verifiable rewards, and continual training, deployed self-improving agents raise unresolved security and governance challenges: optimization pressure can incentivize reward hacking, behavioral drift is difficult to audit or reproduce, and improvements are often entangled in opaque parameter updates rather than reusable, verifiable artifacts.
  This paper proposes Audited Skill-Graph Self-Improvement (ASG-SI), a framework that treats self-improvement as iterative compilation of an agent into a growing, auditable skill graph. Each candidate improvement is extracted from successful trajectories, normalized into a skill with an explicit interface, and promoted only after passing verifier-backed replay and contract checks. Rewards are decomposed into reconstructible components derived from replayable evidence, enabling independent audit of promotion decisions and learning signals. ASG-SI further integrates experience synthesis for scalable stress testing and continual memory control to preserve long-horizon performance under bounded context.
  We present a complete system architecture, threat model, and security analysis, and provide a fully runnable reference implementation that demonstrates verifier-backed reward construction, skill compilation, audit logging, and measurable improvement under continual task streams. ASG-SI reframes agentic self-improvement as accumulation of verifiable, reusable capabilities, offering a practical path toward reproducible evaluation and operational governance of self-improving AI agents.

---

## From Correctness to Collaboration: Toward a Human-Centered Framework for Evaluating AI Agent Behavior in Software Engineering
- **Authors**: Tao Dong, Harini Sampath, Ja Young Lee, Sherry Y. Shi, Andrew Macvean
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.SE
- **Link**: [https://arxiv.org/abs/2512.23844](https://arxiv.org/abs/2512.23844)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.23844v1 Announce Type: cross 
Abstract: As Large Language Models (LLMs) evolve from code generators into collaborative partners for software engineers, our methods for evaluation are lagging. Current benchmarks, focused on code correctness, fail to capture the nuanced, interactive behaviors essential for successful human-AI partnership. To bridge this evaluation gap, this paper makes two core contributions. First, we present a foundational taxonomy of desirable agent behaviors for enterprise software engineering, derived from an analysis of 91 sets of user-defined agent rules. This taxonomy defines four key expectations of agent behavior: Adhere to Standards and Processes, Ensure Code Quality and Reliability, Solving Problems Effectively, and Collaborating with the User.
  Second, recognizing that these expectations are not static, we introduce the Context-Adaptive Behavior (CAB) Framework. This emerging framework reveals how behavioral expectations shift along two empirically-derived axes: the Time Horizon (from immediate needs to future ideals), established through interviews with 15 expert engineers, and the Type of Work (from enterprise production to rapid prototyping, for example), identified through a prompt analysis of a prototyping agent. Together, these contributions offer a human-centered foundation for designing and evaluating the next generation of AI agents, moving the field's focus from the correctness of generated code toward the dynamics of true collaborative intelligence.

---

## Developing controlled natural language for formal specification patterns using AI assistants
- **Authors**: Natalia Garanina, Vladimir Zyubin, Igor Anureev
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.SE
- **Link**: [https://arxiv.org/abs/2512.24159](https://arxiv.org/abs/2512.24159)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24159v1 Announce Type: cross 
Abstract: Using an AI assistant, we developed a method for systematically constructing controlled natural language for requirements based on formal specification patterns containing logical attributes. The method involves three stages: 1) compiling a generalized natural language requirement pattern that utilizes all attributes of the formal specification template; 2) generating, using the AI assistant, a corpus of natural language requirement patterns, reduced by partially evaluating attributes (the developed prompt utilizes the generalized template, attribute definitions, and specific formal semantics of the requirement patterns); and 3) formalizing the syntax of the controlled natural language based on an analysis of the grammatical structure of the resulting patterns. The method has been tested for event-driven temporal requirements.

---

## Generative Video Compression: Towards 0.01% Compression Rate for Video Transmission
- **Authors**: Xiangyu Chen, Jixiang Luo, Jingyu Xu, Fangqiu Yi, Chi Zhang, Xuelong Li
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: eess.IV
- **Link**: [https://arxiv.org/abs/2512.24300](https://arxiv.org/abs/2512.24300)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24300v1 Announce Type: cross 
Abstract: Whether a video can be compressed at an extreme compression rate as low as 0.01%? To this end, we achieve the compression rate as 0.02% at some cases by introducing Generative Video Compression (GVC), a new framework that redefines the limits of video compression by leveraging modern generative video models to achieve extreme compression rates while preserving a perception-centric, task-oriented communication paradigm, corresponding to Level C of the Shannon-Weaver model. Besides, How we trade computation for compression rate or bandwidth? GVC answers this question by shifting the burden from transmission to inference: it encodes video into extremely compact representations and delegates content reconstruction to the receiver, where powerful generative priors synthesize high-quality video from minimal transmitted information. Is GVC practical and deployable? To ensure practical deployment, we propose a compression-computation trade-off strategy, enabling fast inference on consume-grade GPUs. Within the AI Flow framework, GVC opens new possibility for video communication in bandwidth- and resource-constrained environments such as emergency rescue, remote surveillance, and mobile edge computing. Through empirical validation, we demonstrate that GVC offers a viable path toward a new effective, efficient, scalable, and practical video communication paradigm.

---

## Generative AI-enhanced Sector-based Investment Portfolio Construction
- **Authors**: Alina Voronina, Oleksandr Romanko, Ruiwen Cao, Roy H. Kwon, Rafael Mendoza-Arriaga
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: q-fin.PM
- **Link**: [https://arxiv.org/abs/2512.24526](https://arxiv.org/abs/2512.24526)
- **Matched Keywords**: DeepSeek

**Abstract**:
arXiv:2512.24526v1 Announce Type: cross 
Abstract: This paper investigates how Large Language Models (LLMs) from leading providers (OpenAI, Google, Anthropic, DeepSeek, and xAI) can be applied to quantitative sector-based portfolio construction. We use LLMs to identify investable universes of stocks within S&amp;P 500 sector indices and evaluate how their selections perform when combined with classical portfolio optimization methods. Each model was prompted to select and weight 20 stocks per sector, and the resulting portfolios were compared with their respective sector indices across two distinct out-of-sample periods: a stable market phase (January-March 2025) and a volatile phase (April-June 2025).
  Our results reveal a strong temporal dependence in LLM portfolio performance. During stable market conditions, LLM-weighted portfolios frequently outperformed sector indices on both cumulative return and risk-adjusted (Sharpe ratio) measures. However, during the volatile period, many LLM portfolios underperformed, suggesting that current models may struggle to adapt to regime shifts or high-volatility environments underrepresented in their training data. Importantly, when LLM-based stock selection is combined with traditional optimization techniques, portfolio outcomes improve in both performance and consistency.
  This study contributes one of the first multi-model, cross-provider evaluations of generative AI algorithms in investment management. It highlights that while LLMs can effectively complement quantitative finance by enhancing stock selection and interpretability, their reliability remains market-dependent. The findings underscore the potential of hybrid AI-quantitative frameworks, integrating LLM reasoning with established optimization techniques, to produce more robust and adaptive investment strategies.

---

## Localized Calibrated Uncertainty in Code Language Models
- **Authors**: David Gros, Prem Devanbu
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.SE
- **Link**: [https://arxiv.org/abs/2512.24560](https://arxiv.org/abs/2512.24560)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24560v1 Announce Type: cross 
Abstract: Large Language models (LLMs) can generate complicated source code from natural language prompts. However, LLMs can generate output that deviates from what the user wants, requiring supervision and editing. To support this process, we offer techniques to localize where generations might be misaligned from user intent. We first create a dataset of "Minimal Intent Aligning Patches" of repaired LLM generated programs. Each program uses test cases to verify correctness. After creating a dataset of programs, we measure how well various techniques can assign a well-calibrated probability to indicate which parts of code will be edited in a minimal patch (i.e., give a probability that corresponds with empirical odds it is edited). We compare white-box probing (where we propose a technique for efficient arbitrary-span querying), against black-box reflective and self-consistency based approaches. We find probes with a small supervisor model can achieve low calibration error and Brier Skill Score of approx 0.2 estimating edited lines on code generated by models many orders of magnitude larger. We discuss the generalizability of the techniques, and the connections to AI oversight and control, finding a probe trained only on code shows some signs of generalizing to natural language errors if new probability scaling is allowed.

---

## SynRAG: A Large Language Model Framework for Executable Query Generation in Heterogeneous SIEM System
- **Authors**: Md Hasan Saju, Austin Page, Akramul Azim, Jeff Gardiner, Farzaneh Abazari, Frank Eargle
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CR
- **Link**: [https://arxiv.org/abs/2512.24571](https://arxiv.org/abs/2512.24571)
- **Matched Keywords**: DeepSeek

**Abstract**:
arXiv:2512.24571v1 Announce Type: cross 
Abstract: Security Information and Event Management (SIEM) systems are essential for large enterprises to monitor their IT infrastructure by ingesting and analyzing millions of logs and events daily. Security Operations Center (SOC) analysts are tasked with monitoring and analyzing this vast data to identify potential threats and take preventive actions to protect enterprise assets. However, the diversity among SIEM platforms, such as Palo Alto Networks Qradar, Google SecOps, Splunk, Microsoft Sentinel and the Elastic Stack, poses significant challenges. As these systems differ in attributes, architecture, and query languages, making it difficult for analysts to effectively monitor multiple platforms without undergoing extensive training or forcing enterprises to expand their workforce. To address this issue, we introduce SynRAG, a unified framework that automatically generates threat detection or incident investigation queries for multiple SIEM platforms from a platform-agnostic specification. SynRAG can generate platformspecific queries from a single high-level specification written by analysts. Without SynRAG, analysts would need to manually write separate queries for each SIEM platform, since query languages vary significantly across systems. This framework enables seamless threat detection and incident investigation across heterogeneous SIEM environments, reducing the need for specialized training and manual query translation. We evaluate SynRAG against state-of-the-art language models, including GPT, Llama, DeepSeek, Gemma, and Claude, using Qradar and SecOps as representative SIEM systems. Our results demonstrate that SynRAG generates significantly better queries for crossSIEM threat detection and incident investigation compared to the state-of-the-art base models.

---

## HY-MT1.5 Technical Report
- **Authors**: Mao Zheng, Zheng Li, Tao Chen, Mingyang Song, Di Wang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24092](https://arxiv.org/abs/2512.24092)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24092v1 Announce Type: new 
Abstract: In this report, we introduce our latest translation models, HY-MT1.5-1.8B and HY-MT1.5-7B, a new family of machine translation models developed through a holistic training framework tailored for high-performance translation. Our methodology orchestrates a multi-stage pipeline that integrates general and MT-oriented pre-training, supervised fine-tuning, on-policy distillation, and reinforcement learning. HY-MT1.5-1.8B, the 1.8B-parameter model demonstrates remarkable parameter efficiency, comprehensively outperforming significantly larger open-source baselines (e.g., Tower-Plus-72B, Qwen3-32B) and mainstream commercial APIs (e.g., Microsoft Translator, Doubao Translator) in standard Chinese-foreign and English-foreign tasks. It achieves approximately 90% of the performance of ultra-large proprietary models such as Gemini-3.0-Pro, while marginally trailing Gemini-3.0-Pro on WMT25 and Mandarin-minority language benchmarks, it maintains a substantial lead over other competing models. Furthermore, HY-MT1.5-7B establishes a new state-of-the-art for its size class, achieving 95% of Gemini-3.0-Pro's performance on Flores-200 and surpassing it on the challenging WMT25 and Mandarin-minority language test sets. Beyond standard translation, the HY-MT1.5 series supports advanced constraints, including terminology intervention, context-aware translation, and format preservation. Extensive empirical evaluations confirm that both models offer highly competitive, robust solutions for general and specialized translation tasks within their respective parameter scales.

---

## Training Report of TeleChat3-MoE
- **Authors**: Xinzhang Liu, Chao Wang, Zhihao Yang, Zhuo Jiang, Xuncheng Zhao, Haoran Wang, Lei Li, Dongdong He, Luobin Liu, Kaizhe Yuan, Han Gao, Zihan Wang, Yitong Yao, Sishi Xiong, Wenmin Deng, Haowei He, Kaidong Yu, Yu Zhao, Ruiyu Fang, Yuhao Jiang, Yingyan Li, Xiaohui Hu, Xi Yu, Jingqi Li, Yanwei Liu, Qingli Li, Xinyu Shi, Junhao Niu, Chengnuo Huang, Yao Xiao, Ruiwen Wang, Fengkai Li, Luwen Pu, Kaipeng Jia, Fubei Yao, Yuyao Huang, Xuewei He, Zhuoru Jiang, Ruiting Song, Rui Xue, Qiyi Xie, Jie Zhang, Zilu Huang, Zhaoxi Zhang, Zhilong Lu, Yanhan Zhang, Yin Zhang, Yanlei Xue, Zhu Yuan, Teng Su, Xin Jiang, Shuangyong Song, Yongxiang Li, Xuelong Li
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24157](https://arxiv.org/abs/2512.24157)
- **Matched Keywords**: MoE

**Abstract**:
arXiv:2512.24157v1 Announce Type: new 
Abstract: TeleChat3-MoE is the latest series of TeleChat large language models, featuring a Mixture-of-Experts (MoE) architecture with parameter counts ranging from 105 billion to over one trillion,trained end-to-end on Ascend NPU cluster. This technical report mainly presents the underlying training infrastructure that enables reliable and efficient scaling to frontier model sizes. We detail systematic methodologies for operator-level and end-to-end numerical accuracy verification, ensuring consistency across hardware platforms and distributed parallelism strategies. Furthermore, we introduce a suite of performance optimizations, including interleaved pipeline scheduling, attention-aware data scheduling for long-sequence training,hierarchical and overlapped communication for expert parallelism, and DVM-based operator fusion. A systematic parallelization framework, leveraging analytical estimation and integer linear programming, is also proposed to optimize multi-dimensional parallelism configurations. Additionally, we present methodological approaches to cluster-level optimizations, addressing host- and device-bound bottlenecks during large-scale training tasks. These infrastructure advancements yield significant throughput improvements and near-linear scaling on clusters comprising thousands of devices, providing a robust foundation for large-scale language model development on hardware ecosystems.

---

## QianfanHuijin Technical Report: A Novel Multi-Stage Training Paradigm for Finance Industrial LLMs
- **Authors**: Shupeng Li, Weipeng Lu, Linyun Liu, Chen Lin, Shaofei Li, Zhendong Tan, Hanjun Zhong, Yucheng Zeng, Chenghao Zhu, Mengyue Liu, Daxiang Dong, Jianmin Wu, Yunting Xiao, Annan Li, Danyu Liu, Jingnan Zhang, Licen Liu, Dawei Yin, Dou Shen
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24314](https://arxiv.org/abs/2512.24314)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24314v1 Announce Type: new 
Abstract: Domain-specific enhancement of Large Language Models (LLMs) within the financial context has long been a focal point of industrial application. While previous models such as BloombergGPT and Baichuan-Finance primarily focused on knowledge enhancement, the deepening complexity of financial services has driven a growing demand for models that possess not only domain knowledge but also robust financial reasoning and agentic capabilities. In this paper, we present QianfanHuijin, a financial domain LLM, and propose a generalizable multi-stage training paradigm for industrial model enhancement.
  Our approach begins with Continual Pre-training (CPT) on financial corpora to consolidate the knowledge base. This is followed by a fine-grained Post-training pipeline designed with increasing specificity: starting with Financial SFT, progressing to Finance Reasoning RL and Finance Agentic RL, and culminating in General RL aligned with real-world business scenarios. Empirical results demonstrate that QianfanHuijin achieves superior performance across various authoritative financial benchmarks. Furthermore, ablation studies confirm that the targeted Reasoning RL and Agentic RL stages yield significant gains in their respective capabilities. These findings validate our motivation and suggest that this fine-grained, progressive post-training methodology is poised to become a mainstream paradigm for various industrial-enhanced LLMs.

---

## MUSIC: MUlti-Step Instruction Contrast for Multi-Turn Reward Models
- **Authors**: Wenzhe Li, Shujian Zhang, Wenxuan Zhou, John Lambert, Chi Jin, Andrew Hard, Rajiv Mathews, Lun Wang
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24693](https://arxiv.org/abs/2512.24693)
- **Matched Keywords**: MPI

**Abstract**:
arXiv:2512.24693v1 Announce Type: new 
Abstract: Evaluating the quality of multi-turn conversations is crucial for developing capable Large Language Models (LLMs), yet remains a significant challenge, often requiring costly human evaluation. Multi-turn reward models (RMs) offer a scalable alternative and can provide valuable signals for guiding LLM training. While recent work has advanced multi-turn \textit{training} techniques, effective automated \textit{evaluation} specifically for multi-turn interactions lags behind. We observe that standard preference datasets, typically contrasting responses based only on the final conversational turn, provide insufficient signal to capture the nuances of multi-turn interactions. Instead, we find that incorporating contrasts spanning \textit{multiple} turns is critical for building robust multi-turn RMs. Motivated by this finding, we propose \textbf{MU}lti-\textbf{S}tep \textbf{I}nstruction \textbf{C}ontrast (MUSIC), an unsupervised data augmentation strategy that synthesizes contrastive conversation pairs exhibiting differences across multiple turns. Leveraging MUSIC on the Skywork preference dataset, we train a multi-turn RM based on the Gemma-2-9B-Instruct model. Empirical results demonstrate that our MUSIC-augmented RM outperforms baseline methods, achieving higher alignment with judgments from advanced proprietary LLM judges on multi-turn conversations, crucially, without compromising performance on standard single-turn RM benchmarks.

---

## Compute-Accuracy Pareto Frontiers for Open-Source Reasoning Large Language Models
- **Authors**: \'Akos Prucs, M\'arton Csutora, M\'aty\'as Antal, M\'ark Marosi
- **Date**: Thu, 01 Jan 2026 00:00:00 -0500
- **Category**: cs.CL
- **Link**: [https://arxiv.org/abs/2512.24776](https://arxiv.org/abs/2512.24776)
- **Matched Keywords**: Mixture of Experts, MoE

**Abstract**:
arXiv:2512.24776v1 Announce Type: new 
Abstract: Large Language Models (LLMs) are demonstrating rapid improvements on complex reasoning benchmarks, particularly when allowed to utilize intermediate reasoning steps before converging on a final solution. However, current literature often overlooks the significant computational burden associated with generating long reasoning sequences. For industrial applications, model selection depends not only on raw accuracy but also on resource constraints and inference costs. In this work, we conduct a test-time-compute aware evaluation of both contemporary and older open-source LLMs, mapping their Pareto frontiers across math- and reasoning-intensive benchmarks. Our findings identify the Mixture of Experts (MoE) architecture as a strong candidate to balance performance and efficiency in our evaluation setting. Furthermore, we trace the trajectory of Pareto efficiency over time to derive an emergent trend regarding accuracy gain per unit of compute. Finally, we demonstrate that there is a saturation point for inference-time compute. Beyond a certain threshold, accuracy gains diminish, indicating that while extended reasoning capabilities are beneficial, they cannot overcome intrinsic model limitations regarding specific complexities.

---

