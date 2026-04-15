# Optimizer States as an Interpretable Window into LLM Learning Dynamics

## The Core Claim

Adaptive optimizer states — specifically Adam's first moment (m) and second moment (v) — encode far richer information about a neural network's learning process than is currently exploited. These states are not merely optimization bookkeeping. They are a compressed, per-parameter history of what the model has learned, what it is struggling with, and what it has seen before. We formalize this observation into three interpretable signal families — **surprise**, **stability**, and **repetition** — and demonstrate that they are sufficiently informative to drive a fully automatic knowledge routing system during multi-teacher distillation of large language models.

---

## The Problem: Training Is a Black Box

When we train an LLM, we observe a single scalar — the loss — going down over time. Behind that scalar, a vastly more complex process is unfolding:

- Some layers are learning rapidly while others are stagnant.
- Some modules are receiving conflicting gradients from different data domains.
- Some knowledge patterns are recurring and consolidating, while others are one-off anomalies.
- Some updates are reinforcing prior learning, while others are actively undoing it.

We know none of this from the loss curve. Existing interpretability methods focus on **inference time** — attention maps, probing classifiers, activation patching. Almost no work systematically reads the **training process itself** for interpretable signals.

Yet the training process already maintains a rich, per-parameter record of its own history. Every adaptive optimizer does this by design. Adam, the most widely used optimizer for LLMs, maintains two exponential moving averages for every parameter in the model:

- **m (first moment)**: a smoothed average of recent gradient directions — effectively a memory of "which way has this parameter been pushed recently?"
- **v (second moment)**: a smoothed average of recent gradient magnitudes — effectively a memory of "how much activity has this parameter seen recently?"

These states are updated at every training step. They are already computed. They are already stored in memory. And they are almost entirely ignored for anything beyond computing the parameter update.

We argue this is a missed opportunity.

---

## The Key Insight: Reading the Optimizer as a Sensor

Consider what the optimizer states actually encode:

### What m tells us

The first moment m_{t,i} for parameter i is an EMA of its gradients. If gradients consistently push in the same direction, m grows large in that direction. If gradients flip back and forth (conflicting signals), m stays near zero despite high activity. Therefore:

- **Large |m|**: consistent learning pressure — the model is receiving a coherent signal.
- **Small |m| with large v**: lots of activity but no consistent direction — noise or conflict.
- **The direction of m**: the recent "preferred update direction" for this parameter.

### What v tells us

The second moment v_{t,i} is an EMA of squared gradients. It tracks the scale of gradient activity regardless of direction. Therefore:

- **Large v**: this parameter has been receiving strong gradients — it is "active."
- **Small v**: this parameter is quiet — either already well-trained or irrelevant to recent data.
- **v relative to m²**: if v ≈ m², all activity is coherent signal. If v >> m², most activity is noise.

### The statistical identity

This relationship is not a loose analogy. From basic statistics:

    E[g²] = (E[g])² + Var(g)

Since m ≈ E[g] and v ≈ E[g²], we can decompose v into signal and noise:

    v ≈ m² + Var(g)

This means the ratio m²/v directly estimates "what fraction of the gradient activity is consistent signal versus random fluctuation" — a per-parameter signal-to-noise ratio, computed for free by the optimizer.

---

## Three Signal Families

We formalize three families of interpretable signals that can be derived from optimizer states and gradient statistics, computed per module (attention block or FFN block per layer):

### 1. Surprise: "Is this input teaching the model something unexpected?"

**Definition (Adam-based):**

    S_t(j) = (1/d_j) · Σ_i  g²_{t,i} / (v_{t-1,i} + ε)

This measures how much the current gradient exceeds what Adam expects for each parameter coordinate. It is a scale-aware novelty detector:

- If a typically quiet parameter suddenly receives a large gradient, that coordinate contributes high surprise.
- If a typically volatile parameter receives a large gradient, that is normal — low surprise.

**What it reveals about the model:**
Surprise heatmaps across layers and training steps expose which modules are encountering genuinely novel information versus processing routine inputs. We can literally see:
- Layer 8 FFN shows high surprise when medical text appears after a long stretch of code.
- Attention heads in deeper layers show persistent surprise on reasoning-heavy examples.
- Early layers quickly stop being surprised — they learn surface features fast.

This is **per-module novelty detection for free**, without any external probe.

### 2. Stability: "Is this module learning consistently or thrashing?"

We define three complementary stability measures:

**Directional consistency (instant):**

    C_t(j) = cos(g_t(j), m_{t-1}(j))

Current gradient alignment with optimizer memory. High C means the new gradient reinforces recent learning. Negative C means active conflict — the current input wants to undo what recent inputs taught.

**Adam ratio (historical):**

    Stab_t(j) = (1/d_j) · Σ_i  m²_{t-1,i} / (v_{t-1,i} + ε)

The signal-to-noise ratio from the statistical identity above. High means sustained consistent learning. Low means directionless activity.

**Magnitude volatility (windowed):**

    V_t(j) = (EMA[r²] - EMA[r]²) / (EMA[r]² + ε)

How much the gradient magnitude fluctuates. Low means steady learning pressure. High means erratic updates — possibly from task switching, teacher conflicts, or noisy data.

**What they reveal about the model:**
Stability signals expose the internal dynamics of knowledge consolidation:
- Modules that achieve high C and low V early in training are learning "easy" skills — surface syntax, common patterns.
- Modules that remain unstable for long periods are where genuine learning challenges lie — complex reasoning, rare constructions, domain-specific knowledge.
- When training on multi-domain data, stability drops are visible at domain boundaries — the optimizer literally records the moment of distributional shift.
- Multi-teacher disagreement manifests as low C_t: different teachers push the same module in different directions.

### 3. Repetition: "Has this pattern appeared before, and is the model learning from it?"

**Momentum-based (optimizer-only):**

    R_t(mom) = EMA of max(0, cos(g_t, m_{t-1}))

If gradients keep aligning with optimizer memory, the same type of learning pressure is recurring. Persistent agreement = recurring pattern. This is the cheapest signal — derived purely from quantities Adam already maintains.

**Bucket surprise decay (learning progress):**

    R_t(hash) = [n(h) / (n(h) + k)] · σ(S̄_{t-Δ}(h) - S̄_t(h))

If surprise decreases within a semantic cluster over time, the model is successfully learning that pattern. Repetition is not just "this appeared before" but "this appeared before AND the model is getting better at it." The distinction matters: a frequently occurring but never-resolved conflict should not register as useful repetition.

**What they reveal about the model:**
Repetition signals track the arc of skill acquisition:
- A pattern that appears with high surprise and then shows decreasing surprise over repeated encounters — the model is learning it.
- A pattern that appears repeatedly but surprise never drops — the model cannot learn it (capacity limitation, conflicting supervision, or fundamentally hard).
- The rate of surprise decay for different domains reveals which knowledge types are "easy" versus "hard" for the architecture.

---

## Demonstration: Knowledge Routing as Evidence

To demonstrate that these signals are genuinely informative — not just correlated noise — we build a system that uses them as the sole basis for non-trivial decisions during training. If the signals contain real information about learning dynamics, a controller that reads only these signals should make intelligent choices that measurably improve outcomes.

### The Tri-Store Architecture (R/F/P)

We equip a student LLM with three knowledge stores:

- **R-store (Retrieval)**: An external embedding buffer that holds examples the model encountered but should not commit to weight updates. No parameters are modified. The knowledge is stored externally for potential future retrieval.

- **F-store (Fast)**: Low-rank adapters (LoRA) attached to each attention and FFN block. These can be updated quickly, are reversible (can be reset), and represent tentative knowledge — "I've seen this a few times but I'm not sure yet."

- **P-store (Permanent)**: The base model weights. Updates here are expensive to reverse and represent consolidated knowledge — "I've seen this repeatedly, I've been learning it consistently, and it's stable."

### Signal-Driven Routing Policy

For each training sample, after the loss is computed and gradients flow, the controller reads the optimizer-derived signals and decides, per module:

**Route to R** when surprise is high but repetition is low:
> "This is novel — I haven't seen it before. Don't change the weights over a one-off. Just remember it."

**Route to F** when repetition is moderate but stability is not yet proven:
> "I've seen this before and there's learning pressure, but the gradients are still messy. Learn it quickly in a reversible store."

**Route to P** when repetition is high AND directional stability is high AND magnitude volatility is low:
> "This pattern keeps recurring, the gradients consistently agree on the direction, and the update magnitudes are steady. This is stable knowledge. Consolidate it permanently."

The entire routing decision is derived from quantities the optimizer already computes. No auxiliary models, no external classifiers, no additional forward passes.

### Why This Validates the Signals

If the routing works — if R/F/P decisions based purely on optimizer signals lead to better distillation quality, less catastrophic forgetting, and more efficient parameter usage — then the signals must contain genuine information about learning dynamics. The routing system becomes **experimental evidence** for the broader claim about optimizer states as interpretable windows into training.

---

## What We Expect to See (Hypotheses)

### H1: Learning Phase Transitions

Over the course of training, the distribution of R/F/P actions should shift:
- **Early training**: Mostly R and F. Everything is novel. The model is exploring.
- **Mid training**: F-dominant. Patterns are recurring, the model is actively acquiring skills.
- **Late training**: Increasing P actions. Knowledge stabilizes and consolidates.

This would correspond to known learning dynamics (rapid early learning → slower consolidation) but observed through a completely new lens — optimizer states rather than loss curves.

### H2: Layer-Wise Specialization

Different layers should show different signal profiles:
- Early layers: Low surprise after initial training (surface features learned quickly), high stability, early consolidation to P.
- Middle layers: Mixed signals, domain-sensitive surprise, slower stabilization.
- Late layers: Task-specific surprise patterns, potentially persistent F-store usage for specialized knowledge.

### H3: Multi-Teacher Conflict Detection

When two teachers disagree on how a module should update (e.g., a coding teacher and a medical teacher want opposite changes to the same FFN), this should appear as:
- High surprise (large unexpected gradients).
- Low directional stability (negative C_t — gradient opposes momentum).
- The policy should route to R or F, protecting base weights from conflicting updates.

### H4: Surprise Decay as Learning Progress

For each knowledge domain, surprise should follow a characteristic curve:
- Initial encounters: High surprise.
- Repeated encounters: Decreasing surprise (model adapts).
- Convergence: Low stable surprise (knowledge acquired).

The rate and shape of this curve may differ across domains and architectures, revealing fundamental properties of how LLMs acquire different types of knowledge.

---

## Broader Implications

### For Interpretability

Optimizer states offer a complementary view of model internals. While inference-time interpretability asks "what does the model know?", optimizer-state interpretability asks "how is the model learning?" — a question that is arguably more actionable for practitioners.

### For Training Efficiency

If we can detect in real time which modules are surprised, which are stable, and which are receiving redundant signals, we can make smarter decisions about:
- Where to allocate compute (skip updates for modules that aren't learning).
- When to stop training on a domain (surprise has converged).
- How to schedule curriculum (present data that targets high-surprise modules).

### For Continual Learning

The R/F/P framework directly addresses catastrophic forgetting by separating knowledge into stores with different update dynamics. The stability signals act as automatic forgetting protection — unstable knowledge stays in reversible F-store until it is safe to consolidate.

### For Multi-Agent / Multi-Source Learning

The teacher routing and conflict detection mechanisms generalize beyond distillation. Any setting where multiple knowledge sources provide potentially conflicting supervision (federated learning, mixture of experts, ensemble distillation) could benefit from optimizer-state-based routing.

---

## Positioning Relative to Existing Work

| Area | Existing Approach | Our Contribution |
|---|---|---|
| Knowledge distillation | Fixed teacher assignment, averaged teacher outputs | Per-sample routing with optimizer-informed conflict detection |
| Interpretability | Inference-time probes, attention analysis | Training-time signal extraction from optimizer states |
| Training dynamics | Loss curves, gradient norms | Per-module signal decomposition (surprise, stability, repetition) |
| Continual learning | EWC, PackNet, progressive nets | Automatic R/F/P routing without task boundaries |
| LoRA / adapters | Static adapter training | Dynamic sparse adapter writes driven by learning signals |
| Curriculum learning | External difficulty scoring | Optimizer-derived novelty/difficulty signals |

---

## One-Paragraph Summary

We show that the internal states of adaptive optimizers — quantities already computed and stored during standard LLM training — encode rich, interpretable information about per-module learning dynamics. We formalize three signal families (surprise, stability, repetition) derived from Adam's moment estimates and demonstrate their utility by building a knowledge routing system for multi-teacher distillation that uses these signals as its sole decision basis. The system automatically routes incoming knowledge to retrieval memory (novel one-offs), fast adapters (recurring but unstable patterns), or permanent base weights (consolidated stable knowledge), with all routing decisions made by reading the optimizer rather than any external classifier. This work opens a new perspective on neural network interpretability — understanding models not by inspecting their outputs or activations, but by reading the optimization process that shaped them.
