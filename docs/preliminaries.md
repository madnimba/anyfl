# Preliminaries

This chapter lays the formal groundwork for the entire book. It expands —
considerably beyond the necessarily terse two-page "Preliminaries" section of
the NeurIPS submission (`neurips.pdf`, §2) — into a full, self-contained
treatment of (i) Vertical Federated Learning (VFL) as a formal learning
protocol, (ii) Byzantine attacks as a general adversarial class and their
specific (and structurally different) realization in VFL, and (iii) the
complete threat model — every assumption, every capability, every protocol
guarantee — that underlies both the *Consistent Cross-View Poisoning* (CCVS)
attack and the *Reference-Guided Attribution and Reconstruction* (RGAR)
defense described in [methodology.md](methodology.md).

**Notational consistency.** Every symbol introduced here is used identically
in [methodology.md](methodology.md) and matches the NeurIPS draft's notation
(`neurips.pdf`, Eqs. 1–8 and Table 1) — this chapter is the canonical glossary
the rest of the book refers back to. Where the methodology chapter
*instantiates* a symbol for a specific dataset or stage (e.g. $\mathbf{p}^A_c$,
$\rho_A$, $w_i$), this chapter defines the *general* form it specializes from.

---

## P.1 Notation Reference

| Symbol | Meaning |
|---|---|
| $K$ | Number of participating clients (parties) |
| $c \in \{1,\dots,K\}$ | A client index |
| $N$ | Number of row-aligned training samples (the shared entity set) |
| $i \in [N] = \{1,\dots,N\}$ | A sample (entity row) index |
| $d_c$ | Raw feature dimensionality held by client $c$ |
| $x_i^{(c)} \in \mathbb{R}^{d_c}$ | Client $c$'s private raw feature view of sample $i$ |
| $x_i = \{x_i^{(1)}, \dots, x_i^{(K)}\}$ | The full (distributed, never centrally assembled) feature vector of sample $i$ |
| $X^{(c)} \in \mathbb{R}^{N\times d_c}$ | Client $c$'s full local feature matrix |
| $\theta_c$ | Client $c$'s bottom-model parameters |
| $f_c(\cdot\,;\theta_c): \mathbb{R}^{d_c} \to \mathbb{R}^{p_c}$ | Client $c$'s bottom encoder ("local model") |
| $h_i^{(c)} = f_c(x_i^{(c)};\theta_c) \in \mathbb{R}^{p_c}$ | Client $c$'s embedding ("smashed data") for sample $i$ |
| $p_c$ | Embedding dimensionality produced by client $c$ |
| $\Vert_{c=1}^{K}$ | Concatenation operator over the $K$ party embeddings |
| $\theta_g$ | Server (active-party) head-model parameters |
| $g(\cdot\,;\theta_g): \mathbb{R}^{\sum_c p_c} \to \mathcal{Y}$ | The server's head model (fusion + classifier) |
| $\hat{y}_i$ | The model's prediction for sample $i$ |
| $y_i \in \mathcal{Y}$ | The true label of sample $i$ — held **only** by the server |
| $\mathbf{y} = (y_1,\dots,y_N)$ | The full label vector |
| $\mathcal{Y}$ | The label space (e.g. $\{0,\dots,9\}$ for MNIST, $\{0,1\}$ for BANK) |
| $\mathcal{L}(\cdot,\cdot)$ | Per-sample loss function (cross-entropy throughout this work) |
| $\Theta = \{\theta_1,\dots,\theta_K,\theta_g\}$ | The full set of trainable parameters across all parties |
| $\nabla_{h_i^{(c)}}\mathcal{L}$ | Gradient of the loss w.r.t. client $c$'s embedding — the *only* quantity the server returns to client $c$ |
| $\mathcal{B} \subset \{1,\dots,K\}$ | The (fixed) subset of Byzantine / malicious clients |
| $\tilde{x}_i^{(c)}$ | A *crafted* (adversarially modified) input for client $c \in \mathcal{B}$ |
| $\psi(\cdot)$ | An embedding-level adversarial transformation |
| $\tilde{h}_i^{(c)}$ | A corrupted embedding submitted by a malicious client |
| $\tilde{H}_i$ | The mixed (partially corrupted) joint representation the server actually trains on |
| Party A | The (single, w.l.o.g. index 0) malicious client in our $K=2$ studies |
| Party B | The benign / passive client |
| $X^{(A)} \in \mathbb{R}^{N \times d_A}$ | Party A's full raw feature matrix |
| $\widetilde{X}^{(A)}$ | Party A's *rearranged/poisoned* feature matrix |
| $j(i) \in [N]$ | The donor index chosen to replace victim row $i$ |
| $\mathcal{D}_L$ | Party A's small stratified auxiliary labeled set |
| $\mathcal{I}_L \subset [N]$ | The index set of $\mathcal{D}_L$ |
| $\varepsilon \in [0.01,0.05]$ | Auxiliary label fraction, $|\mathcal{I}_L| = \lfloor \varepsilon N \rfloor$ |
| $\mathcal{R} \subset [N]$ | The server's small stratified, immutable reference index set (RGAR) |
| $r_{\text{ref}} \in [0.05,0.15]$ | Reference-set fraction, $|\mathcal{R}| = \lfloor r_{\text{ref}} N \rfloor$ |
| $\phi: x^{(A)} \mapsto z \in \mathbb{R}^p$ | Party A's Phase-I representation-learning embedding map |
| $\hat{c}_i \in \{1,\dots,K_c\}$ | Phase-I pseudo-cluster assignment for sample $i$ |
| $\hat{q}_i \in [0,1]$ | Phase-I pseudo-label / cluster-membership confidence |
| $K_c$ | Number of Phase-I pseudo-clusters |
| $\mathcal{G} = \{C_1,\dots,C_{K_c}\}$ | The Phase-I cluster partition |
| $\mu_s$ | L2-normalized centroid of cluster $C_s$ |
| $\pi \in \mathfrak{D}_{K_c}$ | A derangement (fixed-point-free permutation) of $\{1,\dots,K_c\}$ |
| $\pi^\star$ | The maximum-distance derangement chosen by the attacker |
| $D_{st}$ | Cosine *distance* between cluster centroids $\mu_s,\mu_t$ |
| $g^\star$ | The server's fitted head model under poisoning |
| $P_{\text{clean}}$ | The clean (correctly-paired) test/inference distribution |
| $\mathbf{p}^A_c, \mathbf{p}^B_c, \mathbf{p}^{\text{jnt}}_c$ | RGAR per-class reference prototypes (Party A / Party B / joint) |
| $\rho_A, \rho_B \in [0,1]$ | RGAR per-party trust weights |
| $w_i \in [0,1]$ | RGAR per-sample blend weight toward the repaired surrogate |
| $g_\theta(\cdot,\cdot)$ | RGAR's Honest-View Reconstructor |

---

## P.2 Vertical Federated Learning: A Formal Treatment

### P.2.1 Motivation and setting

Federated Learning (FL) was introduced to let multiple organizations
collaboratively train a shared model without centralizing raw data, lowering
the regulatory and operational barriers to cross-organization data sharing
(McMahan et al., 2023). FL bifurcates along *how* the data is distributed:

- **Horizontal FL (HFL)**: participants hold disjoint *samples* drawn from a
  *common feature space* (e.g. many hospitals each holding patient records
  with the same schema). The server aggregates locally trained model updates
  — e.g. via FedAvg.
- **Vertical FL (VFL)**: participants hold disjoint *feature blocks* over a
  *shared entity/sample index* (Vepakomma et al., 2018). The canonical example
  is a bank and a telecom company that serve overlapping sets of customers and
  each hold complementary attributes about the *same* people — neither alone
  has the full feature picture, but jointly they could build a much stronger
  predictive model (e.g. for credit scoring).

This book studies **VFL**. VFL is increasingly used in high-value cross-silo
settings — credit scoring, risk modeling, healthcare analytics (Yang et al.,
2023) — precisely *because* no single organization can legally or
operationally hold all relevant attributes about a shared population. This
also means VFL deployments tend to be **few-party** (often exactly two
organizations), which — as we formalize in §P.4.1 — is structurally
significant for both attack severity and defense design.

### P.2.2 Entities, alignment, and the privacy boundary

**Definition (VFL instance).** A VFL instance consists of:

1. $K$ **clients** (also called *parties*), indexed $c \in \{1,\dots,K\}$.
2. A shared **entity index** $[N] = \{1,\dots,N\}$: every client holds data
   for the *same* $N$ underlying entities (e.g. the same $N$ customers), in
   the *same* row order — this **row alignment** is established once,
   out-of-band, via a privacy-preserving entity-resolution protocol (private
   set intersection) before any training begins, and is assumed fixed and
   correct throughout.
3. For each entity $i \in [N]$ and client $c$, a **private raw feature view**
   $x_i^{(c)} \in \mathbb{R}^{d_c}$. The full (conceptual, never materialized)
   feature vector is the disjoint union
   $x_i = \{x_i^{(1)}, \dots, x_i^{(K)}\}$ — *no party, including the server,
   ever observes more than its own slice $X^{(c)}$*.
4. A **label vector** $\mathbf{y} = (y_1,\dots,y_N)$, $y_i \in \mathcal{Y}$,
   held *exclusively* by one designated party — the **server** (also called
   the *active party*). All other parties are **passive** and never see
   $\mathbf{y}$.

This structure is what makes VFL fundamentally different from HFL at the
*representation* level (§P.2.7): the parties do not compute comparable
quantities over a shared input space — they compute *complementary, disjoint*
views of the same entities.

### P.2.3 Bottom encoders, the head model, and the forward pass

Each client $c$ trains a **bottom model** (also "local model" or "encoder")

$$f_c(\cdot\,;\theta_c): \mathbb{R}^{d_c} \to \mathbb{R}^{p_c}$$

that maps its private raw view to a lower-dimensional **embedding** (also
called "smashed data" or "intermediate representation"):

$$h_i^{(c)} = f_c(x_i^{(c)}; \theta_c).$$

The server holds a **head model** (also "top model" or "fusion model")

$$g(\cdot\,;\theta_g): \mathbb{R}^{\sum_{c=1}^{K} p_c} \to \mathcal{Y}$$

that consumes the **concatenation** of all $K$ embeddings and produces the
prediction:

$$\hat{y}_i = g\Big(\big\Vert_{c=1}^{K} h_i^{(c)} \,;\ \theta_g\Big). \tag{P.1}$$

Concatenation is the canonical, simplest fusion operator (Cheng et al., 2021)
and is what every model architecture in this book's experiments uses
(`KPartyLegacyFlattenVFL`, `KPartyEmbeddingFusion`, the tabular MLP family —
see [methodology.md §1.3](methodology.md#13-vfl-models-used-in-the-attack-phase-architecture-choices-and-why));
more general fusion functions (attention, gating) are possible but not studied
here.

### P.2.4 The joint training objective

The system is trained **end-to-end**, jointly minimizing the empirical risk
over *all* parameters:

$$\Theta^\star = \arg\min_{\Theta}\ \frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}(\hat{y}_i, y_i),
\qquad
\Theta = \{\theta_1,\dots,\theta_K,\theta_g\}. \tag{P.2}$$

In all experiments in this book, $\mathcal{L}$ is the cross-entropy loss and
optimization is by mini-batch SGD-family optimizers (Adam / AdamW — see
[methodology.md §1.3.1](methodology.md#131-training-hyperparameters-identical-for-clean-baseline-attack-and-defense-runs--this-is-what-makes-the-comparisons-valid)
for the exact per-dataset schedules).

### P.2.5 The communication protocol (what crosses party boundaries)

Training proceeds in synchronized rounds. In each round, for a sampled
mini-batch $\mathcal{I}_b \subset [N]$:

1. **Forward pass (bottom → top).** Every client $c$ computes
   $h_i^{(c)} = f_c(x_i^{(c)};\theta_c)$ for $i \in \mathcal{I}_b$ and
   transmits these embeddings to the server. *Raw features $x_i^{(c)}$ never
   leave client $c$.*
2. **Server forward + loss.** The server concatenates the received embeddings
   (Eq. P.1), computes $\hat{y}_i$, and evaluates $\mathcal{L}(\hat{y}_i,y_i)$
   — *labels never leave the server*.
3. **Backward pass (top → bottom).** The server backpropagates and sends each
   client $c$ **only** the gradient of the loss with respect to *that
   client's own embeddings*, $\nabla_{h_i^{(c)}}\mathcal{L}$ — never the full
   computation graph, never another party's gradients, never the labels.
4. **Local update.** Each client locally completes the backward pass through
   its own bottom model, $\nabla_{\theta_c}\mathcal{L} = \nabla_{h^{(c)}}\mathcal{L}\cdot \nabla_{\theta_c} f_c$, and updates $\theta_c$. The
   server updates $\theta_g$ from its own local gradient.

**This is the entire privacy contract of VFL**: the *only* objects that ever
cross a party boundary are forward embeddings $h_i^{(c)}$ (client → server)
and backward embedding-gradients $\nabla_{h_i^{(c)}}\mathcal{L}$ (server →
client). Raw features, raw labels, model weights, and full gradients with
respect to other parties' quantities are never exchanged. This is also,
critically, the contract that an adversarial party can *exploit* (its
transmissions are never directly auditable against another party's raw data —
see §P.3) and that a defense must operate *within* (the server can only ever
reason about embeddings and labels — see §P.4.8).

### P.2.6 Structural assumptions baked into the protocol

For completeness, we make explicit the structural assumptions a "vanilla" VFL
protocol — and, by extension, the version studied in this book — relies on:

| # | Assumption | Role |
|---|---|---|
| A1 | **Row alignment is established correctly and remains fixed.** Entity resolution (e.g. PSI-based ID matching) happens once, out of band, before training. | Without it, "sample $i$" would not refer to the same real-world entity across parties — VFL would be ill-defined. |
| A2 | **Labels reside only at the server / active party**, and are never transmitted to passive parties. | Defines the label-confidentiality boundary that both the protocol and our threat model preserve. |
| A3 | **Communication is point-to-point and (at minimum) integrity-protected** between each client and the server (e.g. authenticated/MAC'd channels, or hardware-backed TEEs in production deployments). | Rules out *transport-layer* tampering (a man-in-the-middle altering a benign party's transmissions) as an attack vector — the threats we study originate *inside* a party, not on the wire. |
| A4 | **Synchronous mini-batch training** with all parties online for every round. | Standard simplification; asynchronous VFL is an active but separate research area not studied here. |
| A5 | **Embedding dimensionality $p_c$ and architecture $f_c$ are fixed for the duration of training** (no dynamic re-architecting). | Keeps the threat surface (what can be corrupted, and how a defense models "normal" embeddings) well-defined and stationary enough to be learnable from a reference set. |

---

### P.2.7 VFL vs. HFL: a structural comparison (why this distinction drives the entire book)

| Aspect | Horizontal FL | Vertical FL |
|---|---|---|
| What's partitioned | **Samples** (rows) | **Features** (columns) |
| What clients hold | Disjoint subsets of *rows* over a *common* feature schema | *Different* feature spaces over the *same* rows |
| What's exchanged | (Aggregatable) **model updates** $\Delta\theta$ over a *common* parameter space | **Embeddings** and **embedding-gradients** over *heterogeneous*, generally non-comparable representation spaces |
| Server's role | Aggregates updates (e.g. FedAvg) | Hosts labels and the head/fusion model — an active learning participant |
| Notion of "outlier" | A client update statistically far from the consensus of (homogeneous) peer updates — *meaningful*, because all updates live in the same space | An embedding from one party compared to another party's embedding — *structurally meaningless*, because the spaces encode entirely different feature semantics |
| Byzantine defense primitive | Robust aggregation / outlier filtering across *interchangeable* client contributions (Krum, trimmed mean, median) — relies on **update homogeneity**: benign clients optimize over the same space and so naturally cluster (Shi et al., 2022; So et al., 2020) | Cannot directly transfer: there is no "consensus among peers" to compare against — each party's embedding is the *unique* encoding of *its own, non-overlapping* feature slice |

This comparison crystallizes the central methodological tension this book
addresses: **HFL Byzantine defenses are built on an assumption — update
homogeneity — that VFL violates by construction.** §P.3.5 formalizes why this
means HFL-style defenses (and even many VFL-adapted defenses) miss the failure
mode CCVS exploits, and [methodology.md §2.0/§2.4](methodology.md#20-why-detect-and-drop-is-the-wrong-primitive-motivating-rgars-design)
shows it empirically (the SOTA baselines collapse or recover only by accident).

### P.2.8 Why $K=2$ is the canonical and most security-critical regime

Real-world VFL deployments are overwhelmingly **few-party** — typically
*exactly two* organizations (a bank + a telecom, a hospital + an insurer)
because each additional party multiplies the legal, contractual, and
operational coordination cost of cross-silo data sharing. This is not an
experimental simplification; it is *the* dominant deployment topology (Yang
et al., 2023; Vepakomma et al., 2018).

It is also the **adversarially worst case**. With $K$ parties each
contributing roughly $1/K$ of the joint representation, a single malicious
party controls $\approx 1/K$ of the information the head model sees. At
$K=2$, that share is **one half** — there is no "honest majority" of
co-equal feature contributors for the server to lean on, unlike in HFL where
the server aggregates over (typically) many clients. Our own ablation over
$K \in \{2,4,8,10\}$ ([methodology.md §1.6.4](methodology.md#164-ablation-number-of-parties-k-post-attack-accuracy--optimal_topk))
empirically confirms this scaling: attack damage falls from 42–73 percentage
points at $K=2$ to under 7 points at $K=10$ — i.e. **attacker leverage decays
roughly as $1/K$**. This is why this book's headline experiments fix $K=2$:
it is simultaneously the most realistic *and* the most security-critical
configuration.

---

## P.3 Byzantine Attacks: General Definition and Their Realization in VFL

### P.3.1 The general Byzantine adversary

**Definition (Byzantine adversary).** In a distributed learning protocol with
$K$ participants, a **Byzantine adversary** is a fixed subset
$\mathcal{B} \subset \{1,\dots,K\}$ of participants who may deviate from the
protocol *arbitrarily* — they may send any value, at any time, in any way that
the protocol's message format permits — with the goal of corrupting the
learned global model $g^\star$. This is in contrast to a *semi-honest
("honest-but-curious")* adversary, who follows the protocol faithfully but
tries to *infer* private information from what it legitimately observes.

Byzantine attacks have been extensively studied in HFL (Fang et al., 2020),
where a malicious client typically manipulates its locally computed *model
update* $\Delta\theta_c$ before submitting it for aggregation — e.g. scaling
it, negating it, replacing it with noise, or colluding with other malicious
clients to bias the aggregate. The *objective* is normally **untargeted**:
degrade the global model's overall accuracy, as opposed to causing
misclassification only on specific trigger inputs (the hallmark of a
*backdoor* — see §P.3.4).

### P.3.2 Two corruption channels available to a VFL Byzantine client

Crucially, a VFL client does **not** submit model updates to an aggregator —
it submits **forward embeddings** computed from its **own private input**
(§P.2.5). This collapses the HFL "manipulate-the-update" channel into two
*structurally distinct* corruption channels available at the VFL forward step:

**(i) Input-level corruption** — craft the raw input fed to one's own honest
encoder:

$$\tilde{h}_i^{(c)} = f_c(\tilde{x}_i^{(c)}; \theta_c), \qquad \tilde{x}_i^{(c)} \text{ is a crafted/rearranged input}$$

**(ii) Embedding-level corruption** — apply an arbitrary transformation
directly to the (honestly computed) embedding before transmission:

$$\tilde{h}_i^{(c)} = \psi\big(f_c(x_i^{(c)};\theta_c)\big), \qquad \psi(\cdot) \text{ an arbitrary embedding-level map}$$

Formally, for $c \in \mathcal{B}$:

$$\tilde{h}_i^{(c)} = f_c(\tilde{x}_i^{(c)};\theta_c)
\quad\text{or}\quad
\tilde{h}_i^{(c)} = \psi\big(f_c(x_i^{(c)};\theta_c)\big). \tag{P.3}$$

For benign clients $c \notin \mathcal{B}$, the clean embedding
$h_i^{(c)} = f_c(x_i^{(c)};\theta_c)$ is used as normal. The server — *which
has no way to distinguish the two cases, or to know an attack is occurring at
all* — assembles the **mixed representation**

$$\tilde{H}_i = \big\Vert_{c=1}^{K}
\begin{cases}
\tilde{h}_i^{(c)}, & c \in \mathcal{B} \\
h_i^{(c)}, & c \notin \mathcal{B}
\end{cases}
\qquad \text{and computes} \qquad
\hat{y}_i = g(\tilde{H}_i;\theta_g). \tag{P.4}$$

**CCVS occupies channel (i) exclusively** (§P.3.6): it never applies an
arbitrary embedding transformation $\psi$; it only **permutes which genuine,
previously-observed raw rows the attacker's *own* honest encoder $f_A$ is
asked to encode**. This is a deliberate scoping choice — explained fully in
§P.4.2 — that makes the attack maximally *stealthy at the embedding level*
(every transmitted $\tilde{h}_i^{(A)}$ is a perfectly genuine output of the
honest encoder $f_A$ on a perfectly genuine row of $X^{(A)}$ — there is no
"crafted" or out-of-distribution embedding for a defense to detect via
embedding-space anomaly alone).

### P.3.3 Untargeted vs. targeted: the threat taxonomy

The Byzantine/poisoning literature on FL (and ML security broadly) is
typically organized along two orthogonal axes:

| Axis | Targeted | Untargeted |
|---|---|---|
| **Goal** | Cause specific misclassifications (e.g. "inputs bearing trigger $\tau$ → label $t$") while **preserving** overall clean accuracy (so the corruption stays hidden) | **Degrade overall model utility** — the adversary doesn't care *which* predictions are wrong, only that *many* are |
| **When the damage manifests** | At inference time, *only* on trigger-bearing inputs (the model is otherwise "fine") | Either during training (parameters corrupted) or at inference (broad accuracy collapse) |
| **Canonical example in VFL** | Backdoor attacks (Naseri et al., 2023), label-inference attacks (Fu et al., 2022), reconstruction attacks (Gu and Bai, 2023) — these dominate the existing VFL security literature | Byzantine availability attacks — extensively studied in HFL (Fang et al., 2020) but **largely unexplored in VFL** prior to this work |
| **Why it's hard to detect** | The model looks healthy on all "normal" traffic — only an adversary-chosen trigger reveals the flaw | The damage is diffuse and *global* — there's no single "smoking gun" input; the entire learned cross-party association is subtly wrong |

| Axis | Inference-time | Training-time |
|---|---|---|
| **Mechanism** | Corrupt or query the *already-trained* model (e.g. submit adversarial embeddings at test time, run gradient-based label-inference) | Corrupt the *data the model is fit on*, so the **learned parameters themselves** are wrong |
| **How to fix it (after detection)** | Often patchable without retraining — e.g. filter the offending queries, add an input sanitizer | Generally requires **expensive retraining from scratch** and prolonged loss of service (He et al., 2024; Gu and Bai, 2023) — the corruption is baked into $\Theta^\star$ |

**CCVS sits at the intersection that the existing literature has left open**:
it is **untargeted** (degrades overall accuracy, no trigger, no target label)
*and* **training-time** (corrupts $\Theta^\star$ itself, not merely the
deployed model's behavior on crafted queries). This combination is what makes
it both *practically dangerous* (recovery is expensive) and *scientifically
novel* in the VFL context — the gap this book's attack methodology
([methodology.md §1](methodology.md#1-attack-methodology--consistent-cross-view-poisoning-ccvs))
fills.

### P.3.4 Why HFL Byzantine defenses do not directly transfer to VFL (formal argument)

HFL Byzantine defenses — robust aggregation, outlier filtering, similarity
gating (e.g. Krum (Blanchard et al., 2017), trimmed mean, FLTrust-style cosine
filtering (Cao et al., 2022)) — rest on a load-bearing structural premise:

> **Update homogeneity**: in HFL, every benign client computes an update
> $\Delta\theta_c$ that is an (approximately unbiased, noisy) estimate of the
> *same* underlying gradient $\nabla_\theta \mathcal{L}(\theta; \mathcal{D})$,
> because all clients optimize *the same model over the same feature space*.
> Hence benign updates **naturally cluster**, and a Byzantine update —
> computed from a different objective or fabricated outright — is, with high
> probability, a **statistical outlier** relative to that cluster (Shi et al.,
> 2022; So et al., 2020).

This premise **fails by construction in VFL**:

$$\underbrace{h_i^{(1)} \in \mathbb{R}^{p_1}}_{\text{encodes } x_i^{(1)}}, \quad
\underbrace{h_i^{(2)} \in \mathbb{R}^{p_2}}_{\text{encodes } x_i^{(2)}}, \quad
\dots, \quad x_i^{(c)} \cap x_i^{(c')} = \emptyset \ \ \forall c \neq c'$$

Each party's embedding is the **unique encoding of a non-overlapping feature
slice**. There is no shared objective, no common representation space, and
hence **no expectation that benign embeddings cluster together** — "an
outlier among parties" is not merely rare in VFL, it is *structurally
undefined*: the server fuses *heterogeneous* representations rather than
*aggregating interchangeable* ones. Worse, this heterogeneity is already known
to **undermine** HFL-style similarity/outlier defenses when ported to VFL (Xu
et al., 2024; Oonishi and Nakai, 2025) — but, prior to this work, the
*offensive* implication of that fact (that a malicious party can submit
statistically *plausible* features while corrupting the *global cross-view
association* the head model learns) had been largely overlooked.

Existing VFL-adapted defenses therefore fall back to monitoring **local**
signals — embedding-distribution statistics, gradient norms, temporal
embedding drift, or per-sample reconstruction error. We make a sharper
distinction explicit, because it is exactly the gap CCVS exploits and RGAR is
designed to close:

> These local signals are well-suited to catching **batch-level or
> party-level statistical abnormalities** (an embedding that looks
> "wrong" in isolation). They are **not** designed to catch an attack that
> produces embeddings that are **locally perfectly plausible** — because they
> are genuine outputs of the honest encoder on genuine inputs — while
> systematically corrupting the **joint cross-view association** the head
> model fits. The defect is not *in* any single party's transmissions; it is
> *in the relationship between* the two parties' transmissions, conditioned on
> the label. Detecting it requires comparing **pairs**, calibrated against a
> **joint** notion of "what an honest pair looks like" — which is precisely
> what RGAR's reference prototypes $\mathbf{p}^{\text{jnt}}_c$ and joint
> cosine score $s^{\text{jnt}}$ provide
> ([methodology.md Eq. 10](methodology.md#stage-b--online-per-sample-suspicion-and-party-evidence-scoring)).

### P.3.5 Existing VFL Byzantine-adjacent studies, and what remains open

A handful of recent works do study Byzantine-*like* behavior in VFL — but, as
a class, they import HFL's framing of Byzantine behavior as **local anomaly in
a transmitted value or client contribution**: arbitrary corrupted values,
message tampering, label-flipping, bit-flipping, sign-flipping (Yuan et al.,
2022; Xu et al., 2024; Oonishi and Nakai, 2025). These are useful *stress
tests* for robustness, but they do not model an adversary that is
**strategic about the semantic relationship it fabricates between views** —
which is the failure mode that actually exploits VFL's defining structural
property (learning from *fused cross-party representations* rather than
*aggregating comparable updates*). This is precisely the overlooked offensive
surface CCVS targets and formalizes (§P.3.6, and in full mechanistic detail in
[methodology.md §1.2](methodology.md#12-stage-2-phase-ii-class-consistent-cluster-swap-poisoning)).

### P.3.6 Where CCVS sits in this taxonomy (a precise characterization)

Putting the above together, **Consistent Cross-View Poisoning (CCVS)** is
precisely characterized as:

- **Byzantine** (§P.3.1): a fixed adversarial party $\mathcal{B} = \{A\}$
  deviating arbitrarily from "submit your true encoded view."
- **Channel (i): input-level only** (§P.3.2): $\tilde{h}_i^{(A)} = f_A(\tilde{x}_i^{(A)})$
  with $\tilde{x}_i^{(A)} = x_{j(i)}^{(A)}$ — a genuine row, permuted; *never*
  $\psi(\cdot)$ applied to an embedding, *never* a synthetic feature vector.
- **Untargeted** (§P.3.3, top axis): the goal is global clean-accuracy
  collapse (Eq. P.6 below), with no trigger pattern and no target label.
- **Training-time** (§P.3.3, bottom axis): the corruption is baked into
  $\Theta^\star = \{\theta_1,\dots,\theta_K,\theta_g\}$ via Eq. P.5 — fixing it
  requires retraining, not query filtering.
- **Locally plausible / globally wrong** (§P.3.4): every transmitted
  $\tilde{h}_i^{(A)}$ is statistically indistinguishable, *in isolation*, from
  a clean embedding — the corruption lives entirely in the
  *label-conditioned cross-view relationship* $(\tilde{h}^{(A)}, h^{(B)} \mid y)$,
  which only a joint, reference-calibrated comparison (RGAR's Stage B,
  [methodology.md §2.2](methodology.md#22-stage-by-stage-description-with-full-equations))
  can expose.

---

## P.4 Threat Model

This section is the single authoritative specification of every assumption,
capability boundary, and protocol guarantee that defines the security setting
studied throughout this book. It governs **both** the attack
([methodology.md §1.0](methodology.md#10-setting-and-threat-model)) and the
defense ([methodology.md §2.1](methodology.md#21-reference-set-assumption-and-reconstructed-objects)),
and every numerical result reported should be interpreted *relative to* this
specification.

### P.4.1 Party roles, cardinality, and the security-critical regime

| Role | Cardinality / identity | Notes |
|---|---|---|
| Total parties | $K \geq 2$; **headline studies fix $K=2$** | $K=2$ is both the dominant real deployment topology and the adversarial worst case (formal argument in §P.2.8) |
| Malicious party | Exactly one, $\mathcal{B} = \{A\}$, conventionally client index $0$ ("Party A") | A single-attacker model — the **minimal** non-trivial Byzantine setting, and the realistic one for cross-silo VFL where collusion among independent regulated institutions is a substantially stronger (and less plausible) assumption |
| Benign party | Party B (and, for $K>2$ ablations, parties $B_1,\dots,B_{K-1}$) | Trains *exactly* as it would in an unattacked deployment — no special behavior, no awareness of the attack, no participation in any defense mechanism beyond the standard protocol |
| Server / active party | Single, holds $\mathbf{y}$ and $\theta_g$ | Doubles as the *defender* when RGAR is deployed (§P.4.8) — the defense requires **no cooperation from, or special trust placed in, any client** |
| Attacker selection | Empirically chosen as the *most informative* client via a leave-one-out informativeness probe (`_probe_most_informative_client`, [methodology.md §1.4.1](methodology.md#141-attacker-selection-probe-which-client-should-be-the-attacker)) | Models the realistic worst case: *if* an institution is compromised, the most damaging one to lose control of is the one whose data the joint model relies on most |

### P.4.2 Adversary capabilities — what Party A *can* do

Party A may, **at will and throughout training**:

1. **Permute its own raw input rows.** It may submit, for any sample index
   $i$, the genuine raw feature vector belonging to a *different* sample
   $j(i)$ from its own dataset:
   $$\tilde{x}_i^{(A)} = x_{j(i)}^{(A)}, \qquad j(i) \in [N].$$
2. **Choose $j(i)$ adaptively**, using any function of information available
   to it (its own feature matrix $X^{(A)}$, its inferred cluster structure,
   its auxiliary labels — see §P.4.5). The donor-selection mechanism
   ([methodology.md §1.2](methodology.md#12-stage-2-phase-ii-class-consistent-cluster-swap-poisoning))
   is precisely this choice, optimized adversarially.
3. **Persist the chosen permutation across all training epochs** — this
   *consistency* (as opposed to per-epoch or per-batch re-randomization) is
   the single most important capability the attack exploits (formalized as
   the mutual-information argument, [methodology.md Eq. (6)](methodology.md#122-why-cluster-level-consistent-swapping-and-not-per-sample-random-noise)).
4. Use a small **auxiliary labeled set** $\mathcal{D}_L$ (§P.4.5) for
   calibrating its clustering and (for BANK) its donor-selection classifier.

What Party A explicitly **does not** do — and this is as important to the
threat model as what it *can* do — is craft synthetic feature vectors,
manipulate gradients, flip labels, or embed test-time triggers (enumerated
fully in §P.4.4). **Every transmitted feature is a bit-exact, previously
genuine row of $X^{(A)}$.** This scoping is deliberate: it defines the
*minimal* capability set sufficient to mount a devastating attack, which is
the strongest possible argument that the threat is realistic and the gap in
existing defenses is consequential — an adversary need not be sophisticated
enough to synthesize data or tamper with the learning algorithm; *merely
permuting its own genuine rows, consistently, is enough.*

### P.4.3 Adversary knowledge — what Party A *knows*

| Item | Available to Party A? |
|---|---|
| Its own raw feature view $X^{(A)} \in \mathbb{R}^{N \times d_A}$ | **Yes** — by definition of being a party |
| Its own bottom model $f_A(\cdot;\theta_A)$ and its parameters/gradients | **Yes** — it trains $f_A$ itself |
| Phase-I cluster assignments $\{\hat{c}_i\}$ and confidences $\{\hat{q}_i\}$ derived *from its own view* | **Yes** — these are computed entirely locally, with no information from any other party |
| A small stratified auxiliary labeled set $\mathcal{D}_L = \{(x_i^{(A)}, y_i)\}_{i \in \mathcal{I}_L}$, $|\mathcal{I}_L| = \lfloor \varepsilon N \rfloor$, $\varepsilon \in [0.01, 0.05]$, with $\geq 1$ sample per class | **Yes** — see §P.4.5 for the formal definition and realism justification |
| The fact that it is conducting an attack, the existence of $\mathcal{B}$, $K$, etc. | **Yes** (trivially — it is the attacker) |
| Whether/which defense mechanism the server runs | Conservatively assumed **unknown but irrelevant** — CCVS is constructed without reference to any specific defense, and the experiments evaluate it as a fixed, defense-oblivious strategy against multiple defenses post hoc |

### P.4.4 Adversary non-knowledge — hard limits on what Party A *cannot* do or see

| Item | Available to Party A? |
|---|---|
| Party B's (or any other benign party's) raw features $X^{(B)}$ | **No** |
| Party B's bottom-model architecture or weights $\theta_B$ | **No** |
| The full label vector $\mathbf{y}$ (beyond the $\varepsilon$-fraction in $\mathcal{D}_L$) | **No** |
| Any message on the B↔server channel (forward embeddings $h_i^{(B)}$, or backward gradients $\nabla_{h^{(B)}}\mathcal{L}$ destined for B) | **No** — the channel is assumed authenticated/integrity-protected (Assumption A3, §P.2.6); this rules out on-the-wire eavesdropping or tampering as an attack vector |
| The server's head-model architecture/weights $\theta_g$, beyond what it can infer from its own received gradients $\nabla_{h^{(A)}}\mathcal{L}$ | **No** direct access; only the standard protocol-sanctioned gradient signal |
| The composition or existence of the server's reference set $\mathcal{R}$ (under the *non*-conservative reading) | Formally **assumed known** anyway — see the conservative stance in §P.4.8 |

Equally important — these are **structural commitments of the attack design**,
not limitations imposed by an external referee:

- **Sample alignment is never broken.** Row $i$ still pairs
  $(\tilde{x}_i^{(A)}, x_i^{(B)}, y_i)$ at the server — CCVS *only* changes
  *which* feature content occupies a given row's Party-A slot; it never
  desynchronizes the entity index. (Were alignment broken, the attack would
  degenerate into trivial data corruption that any consistency check would
  catch instantly.)
- **The test set is never poisoned.** `swap.protect_test: true` is enforced
  in every run — Party A submits **clean, correctly aligned** features at
  inference time. CCVS is a pure **training-time / availability** attack: the
  damage is to $\Theta^\star$, manifesting as collapsed *clean*-test accuracy,
  not a backdoor trigger ([methodology.md §1.2.9](methodology.md#129-stealth-diagnostics-per-strategy-stealthjson)
  spells out exactly why this distinction is the crux of interpreting the
  results).

### P.4.5 The auxiliary labeled set $\mathcal{D}_L$: definition and realism

**Formal definition.**

$$\mathcal{D}_L = \big\{(x_i^{(A)}, y_i)\big\}_{i \in \mathcal{I}_L},
\qquad |\mathcal{I}_L| = \lfloor \varepsilon N \rfloor,
\qquad \varepsilon \in [0.01, 0.05], \tag{P.5}$$

stratified so that **at least one labeled example per class** is present.
($\mathcal{D}_L$ is used identically for Phase-I clustering calibration
*and*, where applicable — UCI-BANK's `class_flip` — for donor-class
prediction; see [methodology.md §1.2.8](methodology.md#128-uci-bank-special-case-breaking-the-structural-ceiling-with-class_flip).
No *additional* label budget is ever drawn for the latter.)

**Why this is strictly weaker than — and therefore more realistic than — full
label access**, and why $1$–$5\%$ is a defensible real-world budget (three
independent channels, each separately sufficient to motivate the assumption):

1. **Gradient-based label-inference is already a demonstrated VFL attack
   surface.** The literature on label leakage in split learning / VFL (Fu et
   al., 2022; He et al., 2024) shows that a participating party can often
   recover labels (or strong proxies for them) for *some* fraction of samples
   purely from the gradient signal it legitimately receives during normal
   training — i.e. $\mathcal{D}_L$ may require **no extra-protocol access at
   all**, just patient observation of the standard backward messages.
2. **Contractual outcome-sharing is common between collaborating
   institutions.** Real cross-silo VFL deployments often involve parties that
   have legitimate, audited, partial visibility into outcomes for shared
   customers — e.g. a bank may legally know whether a shared customer later
   defaulted on a loan, independent of the VFL collaboration itself.
3. **Partially public records exist for many tabular domains** — bankruptcy
   filings, public health statistics, census/labor outcomes — that a party
   could cross-reference against its own customer index to recover labels for
   a small subset without violating any protocol guarantee.

The ablation over $\varepsilon$
([methodology.md §1.6.5](methodology.md#165-ablation-auxiliary-label-fraction-ε-post-attack-accuracy--k2))
demonstrates the threat remains serious down to $\varepsilon = 1\%$ — i.e.
the *practically achievable* lower end of these channels is already
sufficient to mount tens-of-percentage-points damage.

### P.4.6 The formal adversarial objective

**Stage 1 — the poisoned fitting problem.** Party A replaces its training
features with a rearranged set $\widetilde{X}^{(A)}$, where
$\tilde{x}_i^{(A)} = x_{j(i)}^{(A)}$ for some donor function
$j: [N] \to [N]$. The server, **oblivious to the poisoning** (it has no reason
to suspect anything — every transmitted value is a genuine embedding of a
genuine row), fits the empirical-risk minimizer over the corrupted data:

$$g^\star = \arg\min_{g}\ \frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\big(g(\tilde{h}_i^{(A)} \,\Vert\, h_i^{(B)}),\, y_i\big),
\qquad \tilde{h}_i^{(A)} = f_A(\tilde{x}_i^{(A)}). \tag{P.6}$$

**Stage 2 — the attacker's true objective: maximize *clean*-test loss.** The
adversary does not care about the loss it induces on the *poisoned training
distribution* — that is merely the mechanism. Its actual goal is to choose the
permutation $\widetilde{X}^{(A)}$ (equivalently, the donor function $j(\cdot)$)
that **maximizes the resulting loss on the clean, correctly-paired test
distribution**:

$$\max_{\widetilde{X}^{(A)}}\
\mathbb{E}_{P_{\text{clean}}}\Big[
\mathcal{L}\big(g^\star(f_A(x^{(A)}) \,\Vert\, f_B(x^{(B)})),\, y\big)
\Big]. \tag{P.7}$$

**The conceptual crux** (worth dwelling on, since it is the entire mechanism
of the attack and the reason it is so hard to detect): Eq. (P.6) and Eq. (P.7)
are evaluated on **different distributions** — the server *fits* $g^\star$ on
the *poisoned* joint distribution $(\widetilde{X}^{(A)}, X^{(B)}, \mathbf{y})$,
but is *evaluated* (and deployed) on the *clean* joint distribution
$(X^{(A)}, X^{(B)}, \mathbf{y})$. The adversary's leverage comes entirely from
this **train/deploy distribution mismatch that it controls**: by choosing
$j(\cdot)$ to be *consistent* (same mapping every epoch — §P.3.6 point 3) and
*class-inverting* (cluster-level semantic mismatch — [methodology.md Eqs.
(6)–(7)](methodology.md#123-optimal-cluster-to-cluster-mapping-maximum-distance-derangement)),
it forces $g^\star$ to internalize a **stable, statistically plausible, but
fundamentally false** cross-view association — one that fits the poisoned
training data *well* (low training loss — nothing looks wrong from the
inside) but is *actively misleading* once the genuine pairing is restored at
test time. This is precisely why the attack is **untargeted** (the false
association corrupts predictions broadly, not on a chosen trigger set) and
**training-time** (the falsehood is baked into $\theta_g, \theta_A$
themselves, not a deployable artifact that can be filtered post hoc).

### P.4.7 Protocol-level assumptions retained from §P.2.6 (restated for the security analysis)

The threat model **inherits, and does not weaken**, the structural protocol
assumptions of §P.2.6 — in particular:

- **A1 (alignment)** is preserved by the attack itself (§P.4.4) — the
  adversary's permutation never desynchronizes entity indices.
- **A2 (label confidentiality)** is preserved — labels never leave the server,
  and Party A's only label exposure is the small, bounded $\mathcal{D}_L$.
- **A3 (channel integrity)** rules out wire-level tampering — every threat we
  study originates from *within* a party's own data-submission choice, never
  from intercepting or altering another party's transmissions. This is what
  makes the attack a genuinely **novel availability threat** rather than a
  repackaging of classical man-in-the-middle tampering: *all corruption is
  authored locally, by a party acting entirely within its protocol rights to
  choose what data to encode and submit.*

### P.4.8 RGAR's reference-set assumption $\mathcal{R}$

The defense requires one additional, carefully bounded assumption — made
explicit here because it is the *sole* extra trust placed in the protocol
beyond the standard VFL contract:

**Formal definition.**

$$\mathcal{R} \subset [N], \qquad
|\mathcal{R}| = \lfloor r_{\text{ref}} N \rfloor, \quad r_{\text{ref}} \in [0.05, 0.10]\text{–}[0.05,0.15],
\qquad
|\mathcal{R} \cap \{i : y_i = c\}| \geq 1\ \ \forall c \in \mathcal{Y}. \tag{P.8}$$

(The methodology chapter's per-dataset overlays use $r_{\text{ref}}$ values
within $[0.085, 0.16]$ — see
[methodology.md §2.3.4](methodology.md#234-rgar-effective-hyperparameters-by-dataset-post-merge-summary)
— all consistent with this bounded range.)

**The protocol it requires (one-time, before training begins):**

1. Party A submits $\{x_i^{(A)}\}_{i \in \mathcal{R}}$ to the server **once**,
   via an **authenticated channel** (the same channel-integrity guarantee as
   A3, §P.2.6) — *honestly*, i.e. these are the true raw features for these
   rows.
2. The server stores an **immutable copy**. This is the critical property:
   once stored, this reference copy can never be altered by any subsequent
   action of any party — it is the server's fixed, trusted "ground truth"
   anchor for what an honest Party-A view looks like, *for these specific
   rows*.
3. **No labels are disclosed to Party A** as part of this exchange — the
   reference protocol adds *zero* label exposure beyond $\mathcal{D}_L$
   (§P.4.5), preserving A2.

**Operational realism.** This single pre-training exchange is operationally
comparable to the **entity-alignment step (PSI / ID matching) that production
VFL systems already mandate** before any joint training can occur (Yu et al.,
2024) — i.e. it does not introduce a *qualitatively new* kind of
cross-organization exchange, only a small, bounded, one-time, audited instance
of a kind of exchange the deployment already requires.

**The conservative "attacker knows $\mathcal{R}$" stance — and why it costs
the defender nothing.** We adopt the *pessimistic* assumption that Party A
knows exactly which rows constitute $\mathcal{R}$ (e.g. it could in principle
infer this from which of its submissions were requested via the special
authenticated channel). This assumption can only ever *help* the attacker —
and yet it confers **no actual advantage**, for a simple, airtight reason:
*even knowing $\mathcal{R}$, Party A cannot retroactively alter the server's
already-stored, immutable reference copy.* The rows in $\mathcal{R}$ were
submitted honestly, stored once, and are frozen thereafter — Party A's
knowledge of *which* rows they are changes nothing about *what value* the
server holds for them. We therefore analyze and report results under this
worst-case assumption throughout, and RGAR remains fully effective
(detection rates of 97–100% — [methodology.md §2.5.1](methodology.md#251-defense-comparison-clean-test-accuracy--detection-rate-in-parentheses)).

This is precisely the **calibration anchor** for every subsequent RGAR stage:
the per-class prototypes $\mathbf{p}^A_c, \mathbf{p}^B_c, \mathbf{p}^{\text{jnt}}_c$
and diagonal variances $\Sigma^A_c, \Sigma^B_c$ (Stage A), the Honest-View
Reconstructor $g_\theta$ (also Stage A), and ultimately the entire online
scoring/attribution/repair pipeline (Stages B–E) are all *fit from, and
periodically re-validated against,* this one small, immutable, honest anchor —
detailed in full in [methodology.md §2.1–2.2](methodology.md#21-reference-set-assumption-and-reconstructed-objects).

### P.4.9 Consolidated assumptions ledger

A single-glance reference enumerating **every** assumption made anywhere in
the attack-defense study, for reproducibility review and critique:

| ID | Assumption | Where it matters | Conservative direction |
|---|---|---|---|
| A1 | Row alignment fixed & correct (one-time PSI before training) | Defines what "sample $i$" means; both attack and defense rely on it | Standard VFL precondition — not specific to this study |
| A2 | Labels held only at server; never transmitted | Bounds adversary's label knowledge to $\mathcal{D}_L$ | Strict — even RGAR's reference exchange discloses no labels |
| A3 | Authenticated/integrity-protected channels (or TEEs) | Rules out wire-level tampering as an attack vector for *both* B↔server and the one-time $\mathcal{R}$ exchange | Standard production assumption (Yu et al., 2024) |
| A4 | Synchronous, all-parties-online training | Keeps the protocol model tractable and standard | Not security-relevant; a modeling simplification |
| A5 | Fixed architecture/embedding dims for training duration | Lets a reference-fitted trust model remain calibrated | Mild — violated only by adversarial dynamic re-architecting, out of scope |
| T1 | $\mathcal{B} = \{A\}$, single attacker, most-informative party | Defines the malicious-party scope | Conservative toward the *defender* (single, strong attacker is the realistic worst case for cross-silo collusion-resistant settings) |
| T2 | Attacker permutes only its **own genuine rows**; never crafts synthetic features, never touches gradients/labels, never breaks alignment, never poisons the test set | Defines exactly what CCVS is and is not | Conservative toward the *defender* — a more powerful (synthetic-feature) adversary is plausible but *not* needed to cause catastrophic damage, which is the strongest argument for taking this minimal threat seriously |
| T3 | Auxiliary label budget $\varepsilon \in [0.01, 0.05]$, stratified, $\geq 1$/class | Bounds attacker's label knowledge; calibrates Phase-I clustering and (BANK) donor selection | Justified by three independent real channels (§P.4.5); ablation shows the threat persists at the low end ($\varepsilon{=}1\%$) |
| T4 | Attacker is defense-oblivious (CCVS designed without reference to RGAR or any specific defense) | Ensures the evaluation is a fair, non-adaptive comparison | Conservative toward the *attacker* — an adaptive, defense-aware adversary could plausibly do *more* damage; not studied here |
| D1 | Reference set $\mathcal{R}$, $r_{\text{ref}} \in [0.05, 0.16]$, stratified, submitted honestly once, stored immutably, no labels disclosed | The sole extra trust RGAR requires beyond the base protocol | Bounded, one-time, operationally comparable to mandatory entity-alignment exchange (Yu et al., 2024) |
| D2 | Attacker is conservatively assumed to **know** $\mathcal{R}$'s composition | Worst-case analysis of RGAR's robustness | Strictly favors the *attacker* — yet RGAR remains effective (§P.4.8), which is the strongest possible robustness argument |
| D3 | Defender (server) has no special trust in, or cooperation from, any client; RGAR runs entirely server-side | Keeps the defense deployable without renegotiating the trust model of the underlying VFL system | Conservative toward the *defender* — RGAR could plausibly do more with client cooperation, but does not require it |

This ledger is the master checklist against which any extension, attack
variant, or defense modification proposed later in this book should be
evaluated: **does it respect every assumption above, and if it relaxes one,
which numbered item, and in which direction?**
