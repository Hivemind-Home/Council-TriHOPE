#### We will describe each component separately into the upcoming sections. But I am here presenting snapshot of our idea.

End-to-end pipeline: Multi-Teacher Distillation + Tri-Store Control (R/F/P)

- 0) Setup

- • Student: Transformer with 𝐿layers.
- • Modules: 𝑗 = (𝑙,𝑏,𝑠)where 𝑏 ∈ {attn,ffn}, 𝑠 ∈ {𝑃, 𝐹}(Base vs LoRA).
- • Fast store 𝐹: LoRA params inside attn/ffn.
- • Permanent store 𝑃: base weights.
- • Retrieval store 𝑅: external memory (examples/embeddings/teacher outputs).


- 1) Per-sample teacher routing (multi-teacher selection) For each sample 𝑥𝑡:

- 1. Compute lightweight routing features (optionally: teacher confidence/entropy, domain tag, retrieval cues).
- 2. Choose a teacher 𝑇𝑡(hard route) or mixture weights (soft route).


Output: selected teacher distribution 𝑝𝑇𝑡(⋅∣ 𝑥𝑡).

- 2) Distillation loss (the training objective)

Compute student distribution 𝑝𝜃(⋅∣ 𝑥𝑡). Loss:

𝐿𝑡(𝜃) = 𝜆𝐾𝐷𝐾𝐿(𝑝𝑇𝑡(⋅∣ 𝑥𝑡) ∥ 𝑝𝜃(⋅∣ 𝑥𝑡)) + 𝜆𝐶𝐸𝐶𝐸(𝑦𝑡,𝑝𝜃(⋅∣ 𝑥𝑡)) + 𝜆𝑟𝑒𝑔𝐿𝑟𝑒𝑔𝑡

(CE optional if labels exist; reg can be trust-region KL-to-old-student, anti-forgetting, etc.)

- 3) Compute module-level gradients and Adam states For each module 𝑗:


- • Gradient: 𝑔𝑡(𝑗) = ∇𝜃(𝑗)𝐿𝑡(𝜃𝑡)
- • Adam memory: 𝑚𝑡−1(𝑗) ,𝑣𝑡−1(𝑗)
- • Norm: 𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2


###### 4) Compute controller signals (per module)

- (A) Surprise (Adam-based)

𝑆𝑡(𝑗) =

1 𝑑𝑗

∑

(𝑔𝑡,𝑖(𝑗))2

𝑣𝑡−1,𝑖(𝑗) + 𝜀 𝑖

- (B) Stability (direction + magnitude)

- • Direction: 𝐶𝑡(𝑗) = cos⁡(𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) )
- • Windowed volatility: 𝑉𝑡(𝑗) = 𝑏𝑡


(𝑗)−(𝑎𝑡(𝑗))2 (𝑎𝑡(𝑗))2+𝜀

where 𝑎𝑡,𝑏𝑡are EMAs of 𝑟𝑡and 𝑟𝑡2

- (C) Repetition (choose default + optional fusion)


- • Default practical: bucket surprise decay 𝑅𝑡(

ℎ𝑎𝑠ℎ,𝑗)

- • Optional: momentum recurrence 𝑅𝑡(

𝑚𝑜𝑚,𝑗)

- • Optional: retrieval hit rate 𝑅𝑡(

𝑟𝑒𝑡,𝑗)

- • Fused:


𝑚𝑜𝑚,𝑗) + 𝜆2𝑅𝑡(

ℎ𝑎𝑠ℎ,𝑗) + 𝜆3𝑅𝑡(

𝑟𝑒𝑡,𝑗)

𝑅𝑡(𝑗) = 𝜆1𝑅𝑡(

###### 5) Coarse module selection (where to act)

Select affected modules (top-M by 𝑟𝑡(𝑗)or threshold):

𝒮𝑡 = TopModules(𝑟𝑡(𝑗))

Only modules in 𝒮𝑡get an action.

###### 6) R/F/P decision policy (store choice)

For each selected module 𝑗 ∈ 𝒮𝑡, choose: R-store

If 𝑆𝑡(𝑗)high but 𝑅𝑡(𝑗)low

→ “novel but one-off” F-store (LoRA fast write)

If 𝑅𝑡(𝑗)medium/high but stability not proven (e.g., 𝐶𝑡(𝑗)not high or 𝑉𝑡(𝑗)high)

→ “repeats but still messy” P-store (consolidate)

If 𝑅𝑡(𝑗)high and stability strong (𝐶𝑡(𝑗) high and 𝑉𝑡(𝑗)low, sustained)

→ “repeats and stable”

###### 7) Execute the write (how the model updates)If action = R

- • Do no weight update for that module.
- • Store to retrieval:


o embedding 𝑒𝑡, bucket id, teacher id, teacher soft targets, metadata. If action = F

- • Freeze base 𝑃, update only LoRA 𝐹parameters in that module.
- • Apply Top-K within LoRA (fine placement):


###### o best: Top-K LoRA rank components

- o or Top-K rows/channels


- • Apply masked AdamW update on LoRA.


If action = P Two clean options:

- 1. Consolidation step (recommended): distill/merge LoRA knowledge into base weights periodically, then reset LoRA for that module.
- 2. Direct base update: allow small base updates with strong regularization (trust region), used only when stable.


- 8) Periodic consolidation (F → P) On a schedule (or when repetition+stability persist):

• For modules flagged “ready for P”:

- o train base weights to match the student-with-LoRA behavior (distillation)
- o then merge/reset LoRA This converts short-term learned patterns into permanent knowledge safely.


- 9) Evaluation plan (what we measure)


- • Distillation quality (task accuracy / perplexity / KD loss)
- • Forgetting / stability (performance on old buckets/domains)
- • Memory efficiency (how often we write to F/P, LoRA sparsity via Top-K)
- • Retrieval reliance (R usage rate, retrieval hit rate)
- • Conflict handling (multi-teacher disagreement scenarios)


## Let’s start

The training loop (with 5 teachers → 1 student)

- Step 0: Setup


- • We have teachers: 𝑇1,𝑇2,𝑇3,𝑇4,𝑇5
- • One student with params 𝜃
- • We also have R/F/P stores (retrieval store, fast weights/adapters, permanent weights)


For each batch / sample 𝒙𝒕

- 1) Router picks the teacher (per sample or per batch) For the current input 𝑥𝑡, the router chooses:

𝑇∗(𝑥𝑡) ∈ {𝑇1,…,𝑇5}

This choice can be based on heuristics or our signals (later we can make it learned).

- 2) Compute the loss (KD + CE)

- • Teacher produces distribution: 𝒑𝑻∗(⋅∣ 𝒙𝒕)
- • Student produces distribution: 𝒑𝜽(⋅∣ 𝒙𝒕)


Loss:

𝐿𝑡(𝜃) = 𝜆𝐾𝐷 𝐾𝐿(𝑝𝑇∗ ∥ 𝑝𝜃) + 𝜆𝐶𝐸 𝐶𝐸(𝑦𝑡,𝑝𝜃) + (𝜆𝑟𝑒𝑔𝐿𝑟𝑒𝑔)

- 3) Compute gradient 𝑔𝑡 = ∇𝜃𝐿𝑡(𝜃𝑡)

This gradient is the “how should the student change” signal.

- 4) Compute our control signals from gradient/optimizer/history This is where our surprise / stability / repetition signals come in.


We compute things like:

- • surprise 𝑆𝑡(how big/unexpected the update is)
- • stability/interference 𝐼𝑡(does it conflict with existing knowledge)
- • repetition 𝑅𝑡(is this pattern recurring)


###### 5) Routing decision: where to write this knowledge (R/F/P)Based on those signals, we decide the write target:

- • R-store (retrieval) if it looks rare / one-off / not worth weight changes
- • F-store (fast) if it’s useful but not yet stable (learn quickly, low risk)
- • P-store (permanent) if it’s repeated + stable + generalizable


Important clarification:

- • We’re not “routing the student” somewhere.
- • We’re routing the update / learning: where should this knowledge be stored?


###### 6) Apply the update accordinglyDepending on the routing result:

- • R: store the example/embedding/teacher outputs in retrieval memory (and maybe do little/no weight update)
- • F: update fast adapters/LoRA/fast weights
- • P: update base (permanent) weights, or consolidate F → P


###### 7) Next batch / next stepRepeat for 𝑥𝑡+1

###### This is the first step

- 1) Step index 𝒕 At training step 𝑡, we process one item of data:

𝑥𝑡

This can be:

- • a single example (one sample), or
- • a microbatch (small batch used for one optimizer step)


Why “sample (or microbatch)”? Because in practice, many systems compute gradients on a microbatch, but mathematically both are treated the same way: one training unit at step 𝑡.

- 2) Loss at step 𝒕 We define the loss for the current step as:


ℒ𝑡(𝜃):= ℒ(𝜃;𝑥𝑡)

###### What this means

- • ℒ(𝜃;𝑥𝑡)is the model’s loss on current data 𝑥𝑡
- • 𝜃are the parameters (weights) of the module we care about
- • ℒ𝑡is just shorthand for “the loss function at time 𝑡”


So instead of writing ℒ(𝜃;𝑥𝑡)every time, we write: ℒ𝑡(𝜃)

This is notation convenience. “ℒ𝑡(𝜃) is a placeholder for whatever training loss we use at step 𝑡.” Since we training is KD-based, the loss is usually made of a few pieces.

###### 1) KD loss to the selected teacher

If the router picks teacher 𝑇for sample 𝑥𝑡, then the teacher gives a probability distribution over outputs:

𝑝𝑇(⋅∣ 𝑥𝑡)

We student model (with parameters 𝜃) also gives a distribution: 𝑝𝜃(⋅∣ 𝑥𝑡)

The KD term encourages the student to match the teacher distribution, often with KL divergence:

ℒ𝑡KD(𝜃) = KL ⁣(𝑝𝑇(⋅∣ 𝑥𝑡) ∥ 𝑝𝜃(⋅∣ 𝑥𝑡))

###### Meaning

- • Teacher says: “these outputs are likely”
- • Student should imitate that full distribution (not just the top label)


This transfers richer information than hard labels.

###### 2) KD + CE (cross-entropy with ground truth) -> We can move with this loss.

Sometimes we don’t want the student to only imitate the teacher. We also want it to stay anchored to the true label 𝑦𝑡. So we add a supervised term:

ℒ𝑡CE(𝜃) = CE(𝑦𝑡,𝑝𝜃(⋅∣ 𝑥𝑡))

Then combine them:

ℒ𝑡(𝜃) = 𝜆KD ℒ𝑡KD(𝜃) + 𝜆CE ℒ𝑡CE(𝜃)

###### Meaning of the 𝝀's

- • 𝜆KD: how much we trust teacher imitation
- • 𝜆CE: how much we enforce ground-truth supervision


They are just weighting coefficients.

###### 3) KD + regularization

We may also add terms that stabilize training or protect memory (especially relevant for we R/F/P idea), e.g.:

- • weight decay
- • consistency penalty
- • anti-forgetting penalty
- • trust-region / KL-to-old-student penalty
- • sparsity penalty on adapters, etc.


Then the full loss becomes something like: ℒ𝑡(𝜃) = 𝜆KD ℒ𝑡KD(𝜃) + 𝜆CE ℒ𝑡CE(𝜃) + 𝜆reg ℒ𝑡reg(𝜃)

###### 4) Why I said “conceptually”Because at the setup stage, we only need:

ℒ𝑡(𝜃)

as a differentiable scalar loss, so we can compute the gradient 𝑔𝑡 = ∇𝜃ℒ𝑡(𝜃𝑡)

The routing math (surprise/stability/repetition) works regardless of the exact decomposition of the loss.

So the exact loss form can be specified later in we method section.

###### 5) Important subtle point (for our method)

Changing the loss composition changes the gradient 𝑔𝑡, and therefore changes our routing signals.

For example:

- • More KD weight 𝜆KD→ gradients reflect teacher behavior more strongly
- • More CE weight 𝜆CE→ gradients reflect label correction more strongly
- • Strong regularization → gradients may look smaller / more stable


So the controller is reading the gradient induced by we training objective, not raw data directly.

That’s actually a strength: it routes based on what the model is actually trying to learn at that step.

- 3) Gradient on the target module We define:


𝑔𝑡 = ∇𝜃ℒ𝑡(𝜃𝑡) ∈ ℝ𝑑

This is the most important line. Break it down

- • 𝜃𝑡: current parameter values at step 𝑡
- • ℒ𝑡(𝜃𝑡): current loss evaluated at current weights
- • ∇𝜃ℒ𝑡(𝜃𝑡): gradient of loss w.r.t. those parameters


So 𝑔𝑡tells we: If I change the parameters slightly right now, in which direction does the loss decrease fastest? This is the exact training signal the optimizer sees.

Why 𝒈𝒕 ∈ ℝ𝒅? Because the target module has 𝑑parameters (or flattened parameter dimension 𝑑). If we target module is, say, an adapter layer with 50k params, then:

• 𝑑 = 50,000

• 𝑔𝑡is a 50,000-dimensional vector Each coordinate 𝑔𝑡,𝑖is the gradient for parameter 𝑖.

###### What we must do for distillation training

• We always compute the normal training gradients so the student can update (that’s standard backprop).

What we do for our method (signals/routing)

We compute extra gradient stats per module/parameter-group (or at least read them) to decide:

- • write to R vs F vs P
- • whether a pattern is stable/repeating/etc.


So it’s like:

- • Training update: uses gradients for the parameters we’re updating (standard).
- • Our controller signals: look at gradients per group (F-group, P-group, trunk, adapters…) to make better decisions.


- 4) “Target module (or parameter group)” — what does that mean? This is very important for our method. We do not necessarily compute one signal for the entire model. Instead, we choose a target module/group, e.g.:


- • F-store parameters only
- • P-store parameters only
- • student trunk
- • a router-related block
- • adapter stack


For this paper, we want modules that are:

- • architecturally meaningful (so the controller’s decisions make sense),
- • small enough that different parts can behave differently,
- • not too many (so routing is stable + cheap).


###### Core idea

A “module” is still a part of the Transformer (attention or FFN in a layer). LoRA just adds a fast sub-parameter set inside each module.

So we define two levels:

- 1. Architecture module (where the knowledge belongs): Attention vs FFN per layer
- 2. Parameter type inside that module: Base (P) vs LoRA (F)


Assume the student has 𝐿layers. For each layer ℓ:

###### A) Attention block (layer 𝓵)

• Base (P) module

𝑀ℓ,𝑃attn = {𝑊𝑞,𝑊𝑘,𝑊𝑣,𝑊𝑜 (base weights)}

• LoRA (F) module 𝑀ℓ,𝐹attn = {Δ𝑊𝑞,Δ𝑊𝑘,Δ𝑊𝑣,Δ𝑊𝑜 (LoRA params)}

###### B) FFN / MLP block (layer 𝓵)

- • Base (P) module 𝑀ℓ,𝑃ffn = {𝑊𝑢𝑝,𝑊𝑔𝑎𝑡𝑒,𝑊𝑑𝑜𝑤𝑛 (base weights)}
- • LoRA (F) module 𝑀ℓ,𝐹ffn = {Δ𝑊𝑢𝑝,Δ𝑊𝑔𝑎𝑡𝑒,Δ𝑊𝑑𝑜𝑤𝑛 (LoRA params)}


That gives we:

- • Architecture modules: 2𝐿(attn + ffn per layer)
- • If we split by P vs F: 4𝐿“parameter modules”


Even simpler (if we want modules) Instead of splitting P and F as separate modules, we can write: “Modules are attention/FFN blocks per layer. We implement F-store as LoRA parameters attached to each module, while P-store corresponds to the base parameters.” Then we keep routing stats per (ℓ,attn)and (ℓ,ffn), but updates go to LoRA when store=F. This is often the cleanest writing.

Then 𝑔𝑡is the gradient restricted to that group. So more explicitly, we could write:

𝑔𝑡(𝑗) = ∇𝜃(𝑗)ℒ𝑡(𝜃𝑡)

for module/group 𝑗. This is usually better because gradient scales differ wildly across modules.

###### 5) Per-parameter vs per-module signalsWe wrote:

- • per-parameter (vector form), or
- • per-module (scalar summary)


This means there are two levels of granularity for the controller.

###### A) Per-parameter (vector form)

Use full gradient vector 𝑔𝑡 ∈ ℝ𝑑, and compute signals coordinate-wise. Example:

- • surprise per parameter
- • stability per parameter using 𝑚𝑡,𝑖,𝑣𝑡,𝑖


Pros:

• very precise

Cons:

• expensive / noisy / hard to route with

###### B) Per-module (scalar summary)

Compress the gradient vector into one scalar that summarizes “how strong the update is” for that module.

This is what we want for routing decisions.

5.1) How we do the right combination per-parameter and per-module signal?

So the flow is:

- 1. Per-module routing: choose which module(s) (𝑙,attn)or (𝑙,ffn)to write to, and whether it’s R vs F vs P.
- 2. Per-parameter routing (Top-K): inside that chosen module, choose which parameters actually get updated.


###### Ideal solution: two-level controller with LoRA + Top-K Core idea

A module is an architectural component: Attention or FFN in a layer. Inside each module, we have two parameter types:

- • Base weights = 𝑃(permanent knowledge)
- • LoRA weights = 𝐹(fast, editable knowledge)


So the controller makes two decisions:

1. Where does knowledge belong? (which modules: attn vs ffn, which layers) 2. How should we write? (LoRA only, and optionally sparse Top-K within LoRA)

- 1) Module definitions (our structure, cleaned & formal) Assume student has 𝐿layers. For each layer 𝑙:


- A) Attention module in layer 𝒍 Permanent (base) parameters

𝑀𝑙,𝑃attn = {𝑊𝑞,𝑊𝑘,𝑊𝑣,𝑊𝑜}

Fast (LoRA) parameters

𝑀𝑙,𝐹attn = {Δ𝑊𝑞,Δ𝑊𝑘,Δ𝑊𝑣,Δ𝑊𝑜}

- B) FFN module in layer 𝒍 Permanent (base) parameters


𝑀𝑙,𝑃ffn = {𝑊up,𝑊gate, 𝑊down}

###### Fast (LoRA) parameters

𝑀𝑙,𝐹ffn = {Δ𝑊up,Δ𝑊gate,Δ𝑊down}

Interpretation:

- • 𝑀𝑃: stable long-term knowledge store
- • 𝑀𝐹: “scratchpad” memory that can change quickly


- 2) Per-module routing (coarse decision) For each training step 𝑡, compute gradients per module:

𝑔𝑡,𝑙,𝑏 = ∇𝜃𝑀

𝑙,𝑏

𝐿𝑡(𝜃𝑡),𝑏 ∈ {attn,ffn}

Compute a scalar “how much does this sample want to change this module?”: 𝑢𝑡,𝑙,𝑏 =∥ 𝑔𝑡,𝑙,𝑏 ∥2

Now choose which modules are “active” this step: Pick Top-M modules by 𝑢𝑡,𝑙,𝑏(e.g., Top 2–6 across all layers), OR threshold by percentile. This gives we interpretable routing like:

• “This sample mostly hits FFN in layers 8–12”

Then our R/F/P controller (using our repetition/stability/surprise) decides for each active module:

- • Write to R (don’t update weights, store as retrieval memory)
- • Write to F (update LoRA inside that module)
- • Promote to P (consolidate later)


- 3) Fine placement inside a selected module (Top-K) Now the key part we asked: once we choose: “Write to 𝐹in module (𝑙,𝑏)” we still don’t need to update all LoRA params in that module.


- 3.1 Option A: Top-K rows / channels, not individual weights


Doing Top-K across every single weight is expensive and noisy. Best “ideal research” compromise:

###### For each LoRA matrix 𝚫𝑾, compute row scores

Let Δ𝑊 ∈ ℝ𝑑out×𝑑in. Compute per-row gradient norm:

𝑠𝑟 =∥ ∇Δ𝑊[𝑟,:]𝐿𝑡 ∥2

Select Top-K rows:

ℛ = TopKIndices(𝑠,𝐾rows)

Mask update:

- • update only rows in ℛ
- • freeze other rows


This is sparse, stable, and efficient. Why rows/channels Top-K is better than per-weight Top-K

- • less noisy (row norms are smoother than single weights)
- • faster to compute and apply
- • maps to real “features/neuron directions” in the model


- 3.2 Option B: Top-K LoRA rank components LoRA usually parameterizes:


Δ𝑊 = 𝐵𝐴,𝐴 ∈ ℝ𝑟×𝑑𝑖𝑛, 𝐵 ℝ𝑑𝑜𝑢𝑡×𝑟

Each rank component 𝑘 ∈ [1..𝑟]corresponds to one “direction” of adaptation. Compute gradient norm per component:

𝑠𝑘 =∥ ∇𝐵[:,𝑘]𝐿𝑡 ∥2 +∥ ∇𝐴[𝑘,:]𝐿𝑡 ∥2

Select Top-K components:

𝒦 = TopKIndices(𝑠,𝐾rank)

Update only those rank components.

This is arguably the “best” for LoRA Because it sparsifies at the natural LoRA unit (rank direction), not random weights.

### So to sum up,

###### 1) Compute the raw gradient (always first)Given our loss 𝐿𝑡(𝜃), compute:

𝑔𝑡 = ∇𝜃𝐿𝑡(𝜃𝑡)

This is the full gradient over all parameters.

###### 2) Per-module decision → module mask

Let modules be 𝑀𝑗(e.g., (𝑙,attn), (𝑙,ffn)). Our controller selects a subset of modules to update:

𝒮𝑡 ⊆ {1,…,𝐽}

Create a module mask 𝑚𝑚𝑜𝑑that is 1 for parameters inside selected modules, else 0:

1 if parameter 𝑖 ∈ ⋃𝑗∈𝒮 𝑀𝑗

(𝑚𝑚𝑜𝑑)𝑖 = {

𝑡

0 otherwise

Apply it:

𝑔𝑡(𝑚𝑜𝑑) = 𝑚𝑚𝑜𝑑 ⊙ 𝑔𝑡

Meaning: only gradients inside chosen modules remain.

###### 3) Store decision (R / F / P) → which parameters are allowed at allIf store = RNo parameter update:

𝑔𝑡𝑒𝑓𝑓 = 0

(we store the example/embedding/teacher output in retrieval instead) If store = F (LoRA) Only LoRA parameters are trainable. Base weights are frozen. So we apply a parameter-type mask 𝑚𝐹(1 on LoRA params, 0 on base):

𝑔𝑡(𝐹) = (𝑚𝐹) ⊙ 𝑔𝑡(𝑚𝑜𝑑)

If store = P We allow base parameters (possibly with a small LR / stronger regularization):

𝑔𝑡(𝑃) = (𝑚𝑃) ⊙ 𝑔𝑡(𝑚𝑜𝑑)

(Usually 𝑚𝑃selects base weights; LoRA may be frozen or merged.)

###### 4) Per-parameter Top-K inside the selected module(s)Now do Top-K within the allowed parameter set (typically within LoRA of those modules).Create a Top-K mask 𝑚𝑡𝑜𝑝𝑘:

𝑚𝑡𝑜𝑝𝑘 = TopKMask(∣ 𝑔𝑡(𝐹) ∣,𝐾)

Then:

𝑔𝑡𝑒𝑓𝑓 = 𝑚𝑡𝑜𝑝𝑘 ⊙ 𝑔𝑡(𝐹)

This 𝑔𝑡𝑒𝑓𝑓is the final effective gradient we actually use for the optimizer step.

###### Putting it all together (one line) For F-store (LoRA) with Top-K:

𝑔𝑡𝑒𝑓𝑓 = 𝑚𝑡𝑜𝑝𝑘

⊙ 𝑚𝐹

⊙ 𝑚𝑚𝑜𝑑

⊙ ∇𝜃𝐿𝑡(𝜃𝑡)

⏟ per-parameter

⏟ LoRA-only

⏟ per-module

⏟ raw gradient

For R-store: 𝑔𝑡𝑒𝑓𝑓 = 0 For P-store: replace 𝑚𝐹with 𝑚𝑃(and often no Top-K or a different Top-K).

###### After we apply our routing decisions (per-module + per-parameter/Top-K), we get an effective gradient 𝒈𝒕𝒆𝒇𝒇. Then we can compute the gradient norm from that.

- 6) Gradient norm as a module-level magnitude signal For a given parameter group (module) 𝑗, define the module-restricted gradient:


𝑔𝑡(𝑗):= ∇𝜃(𝑗)𝐿(𝜃𝑡;𝑥𝑡)

and its L2 norm:

𝑟𝑡(𝑗):=∥ 𝑔𝑡(𝑗) ∥2

Expanded:

𝑑𝑗

2

𝑟𝑡(𝑗) = √∑(𝑔𝑡,𝑖(𝑗))

𝑖=1

What it measures. 𝑟𝑡(𝑗)measures the overall strength of the learning signal on module 𝑗at step 𝑡.

- • Large 𝑟𝑡(𝑗): the sample 𝑥𝑡pushes this module strongly
- • Small 𝑟𝑡(𝑗): little update pressure on this module In our framework, 𝑟𝑡(𝑗)is the basic magnitude signal used to build:
- • surprise scores
- • stability/volatility proxies


- • routing decisions (R/F/P)


###### 7) Why the norm is useful for routing

Our routing decision (R/F/P) is a module-level control action. We do not want a highdimensional control decision over individual parameters. Instead we want coarse statements like:

- • “This step is highly surprising for module 𝑗”
- • “This step looks stable/consistent for module 𝑗”
- • “This update is weak/noisy”


The gradient norm provides a compact scalar that is:

- • easy to track with EMA
- • easy to threshold
- • optimizer-agnostic
- • cheap to compute


So we introduce 𝑟𝑡(𝑗)early as the simplest module-level summary.

###### 8) Important subtlety: the norm does not capture direction

The norm 𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2captures magnitude only, not direction. For example, 𝑔and −𝑔have the same norm. Therefore:

- • a large norm can correspond to a stable update or a conflicting one
- • the norm alone cannot detect directional disagreement This is why we later include directional and variance-based signals, e.g.:
- • cosine alignment with momentum / Adam first moment 𝑚𝑡−1(𝑗)
- • variance/stability proxies using Adam second moment 𝑣𝑡−1(𝑗) So 𝑟𝑡(𝑗)is a first summary, not the whole story.


- 9) Cleaner indexing for our Attention/FFN + Base/LoRA design We index modules by:


- • layer index 𝑙
- • block type 𝑏 ∈ {attn,ffn}
- • parameter subset 𝑠 ∈ {𝑃,𝐹}, where 𝑃= base (permanent) weights, 𝐹= LoRA/adapters (fast) weights


Define:

𝑙,𝑏,𝑠):= ∇

𝑙,𝑏,𝑠):=∥ 𝑔𝑡(

𝑙,𝑏,𝑠) ∥2

𝑔𝑡(

𝜃(𝑙,𝑏,𝑠)𝐿(𝜃𝑡;𝑥𝑡),𝑟𝑡(

Interpretation:

- • large 𝑟𝑡(

𝑙,𝑏,𝐹): sample strongly pushes fast memory (LoRA) in that block

- • large 𝑟𝑡(


𝑙,𝑏,𝑃): sample strongly pushes permanent base weights in that block

Optional (very useful) distinction: pressure vs actual write In practice we may compute norms at two stages:

- • Pre-mask pressure (what the sample wants to change):

𝑟𝑡,(pre

𝑙,𝑏):=∥ 𝑔𝑡(

𝑙,𝑏) ∥2

- • Post-mask write (what we actually allow after routing + Top-K):


𝑙,𝑏,𝑠):=∥ 𝑔̃𝑡(

𝑙,𝑏,𝑠) ∥2

𝑟𝑡,(post

This separates “update pressure” from “written update,” which is helpful for debugging and analysis.

## Surprise Signal

What we want to measure We want a number that answers:

“For this module 𝑗, is the gradient from the new example unusually large compared to what this module has been seeing recently?”

Why gradient?

Because the gradient is the learning pressure this example applies to the model (specifically: to module 𝑗):

- • Small / typical gradient → the model already knows how to handle it
- • Large gradient → the example is pushing hard to change the model
- • Very large relative to recent history → “surprising” / novel / difficult


So mathematically:

Surprise = current gradient magnitude (on module 𝑗) compared to a running baseline (for module 𝑗).

That means, For one module 𝒋(like “Layer 10 FFN”)

On each training step 𝑡, that module produces a gradient vector 𝑔𝑡(𝑗). We turn it into a single number (how strong the push is):

𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2

That’s the current gradient magnitude for module 𝑗.

“Running baseline” = what’s normal for that module recently We keep a moving average (EMA) of past magnitudes:

𝜇𝑡−1(𝑗) ≈ typical size of 𝑟(𝑗) recently

(and often a spread/variability 𝛿𝑡−1(𝑗) too).

###### But first some basics about EMA

EMA means Exponential Moving Average. It’s a way to keep a running average of a quantity (like gradient norm) that gives more weight to recent values and less weight to old ones.

Formula For a sequence 𝑥𝑡, the EMA is:

𝑚𝑡 = (1 − 𝛼)𝑚𝑡−1 + 𝛼𝑥𝑡

where:

- • 𝑚𝑡= current EMA (smoothed value)
- • 𝑥𝑡= current observation
- • 𝛼 ∈ (0,1)= smoothing factor Intuition
- • If 𝛼is small (e.g. 0.01), EMA changes slowly → smoother, more stable
- • If 𝛼is large (e.g. 0.3), EMA reacts quickly → less smooth So EMA is basically: “A memory of recent history, with exponential decay.”


Why “exponential”? Because older values get weighted by powers of (1−𝛼):

𝑚𝑡 = 𝛼𝑥𝑡 + 𝛼(1 − 𝛼)𝑥𝑡−1 + 𝛼(1 − 𝛼)2𝑥𝑡−2 + ⋯

So the influence of old values decays exponentially.

Tiny example Suppose 𝛼 = 0.2, initial 𝑚0 = 10, and new values are:

- • 𝑥1 = 20
- • 𝑥2 = 30


Then:

𝑚1 = 0.8(10) + 0.2(20) = 8 + 4 = 12 𝑚2 = 0.8(12) + 0.2(30) = 9.6 + 6 = 15.6

So EMA moves toward new values gradually (not abruptly). So , again back to the main theory:

- 1.1 EMA baseline surprise (scalar norm version, module-level) This is the simplest and best for intuition.


- Step A: Turn the module gradient into a scalar For module 𝑗, define the module gradient vector:

𝑔𝑡(𝑗) ∈ ℝ𝑑𝑗

Reduce it to a scalar by taking its L2 norm:

𝑟𝑡(𝑗):=∥ 𝑔𝑡(𝑗) ∥2

So 𝑟𝑡(𝑗)is “how big the update signal is”for module 𝑗at step 𝑡.

- Step B: Track what is “normal” using EMA (per module) Maintain a running average of the module gradient norm:


𝜇𝑡(𝑗) = (1 − 𝛼𝜇)𝜇𝑡−1(𝑗) + 𝛼𝜇𝑟𝑡(𝑗)

Meaning:

- • 𝜇𝑡(𝑗)estimates the typical gradient size for module 𝑗
- • 𝛼𝜇controls responsiveness: o small 𝛼𝜇→ smoother, slower o large 𝛼𝜇→ reacts quickly


###### Step C: Track spread / variability (robust dispersion, per module)We need not only the average but also how much gradients fluctuate.Define an EMA of absolute deviations:

𝛿𝑡(𝑗) = (1 − 𝛼𝛿)𝛿𝑡−1(𝑗) + 𝛼𝛿 ∣ 𝑟𝑡(𝑗) − 𝜇𝑡(𝑗) ∣

This is an EMA of absolute deviations from the running mean. Why absolute deviation? Because it’s more robust than squared deviation (less sensitive to outliers). Interpretation:

- • small 𝛿𝑡(𝑗)→ norms are tightly clustered (stable scale)
- • large 𝛿𝑡(𝑗)→ norms are naturally noisy/variable


###### Step D: Define surprise as normalized deviation (z-like score, per module)Now compare current norm to baseline (causal version):

𝑟𝑡(𝑗) − 𝜇𝑡−1(𝑗) 𝛿𝑡−1(𝑗) + 𝜀

𝑆𝑡(𝑗):=

Important detail (causality): use 𝜇𝑡−1(𝑗) , 𝛿𝑡−1(𝑗) instead of 𝜇𝑡(𝑗), 𝛿𝑡(𝑗)so we compare the sample against history before seeing it (otherwise the baseline is “diluted” by the current point).

###### Interpretation of 𝑺𝒕(𝒋)

- • 𝑆𝑡(𝑗) ≫ 0: much larger than expected → high surprise
- • 𝑆𝑡(𝑗) ≈ 0: near normal → typical


- • 𝑆𝑡(𝑗) < 0: smaller than expected → easy / already learned


Tiny example If recent norms are around 2.0 with small variation:

• 𝜇𝑡−1(𝑗) = 2.0 • 𝛿𝑡−1(𝑗) = 0.2

If 𝑟𝑡(𝑗) = 3.0:

3.0 − 2.0 0.2 + 𝜀

𝑆𝑡(𝑗) ≈

≈ 5

Very high surprise. If 𝑟𝑡(𝑗) = 2.05, then 𝑆𝑡(𝑗) ≈ 0.25: ordinary. Why this works for routing This score is a proxy for:

- • “How much new information is trying to enter the model (module 𝑗)?”
- • “How strongly is this sample challenging the current memory?”


So high surprise is a good trigger for:

- • F-store (if repeating)
- • R-store (if rare)


Intuition with a tiny example Suppose recent gradient norms are around 2.0, with small variation:

- • 𝜇𝑡−1 = 2.0
- • 𝛿𝑡−1 = 0.2


Now a new sample gives 𝑟𝑡 = 3.0. Then:

3.0 − 2.0 0.2 + 𝜀

𝑆𝑡 =

≈ 5

That’s a very high surprise score. This sample is pushing much harder than normal.

If instead 𝑟𝑡 = 2.05, then 𝑆𝑡 ≈ 0.25: ordinary.

- 1.2 Adam-based surprise (preconditioned version, module-level) -> We will use this This is the more principled version when using Adam/AdamW. The key improvement:


Instead of treating all parameters equally, normalize each gradient coordinate by what Adam already expects for that coordinate (scale-aware surprise).

- Step A: Adam’s memory states (per module / per parameter) Adam maintains:

𝑚𝑡 = 𝛽1𝑚𝑡−1 + (1 − 𝛽1)𝑔𝑡,𝑣𝑡 = 𝛽2𝑣𝑡−1 + (1 − 𝛽2)(𝑔𝑡 ⊙ 𝑔𝑡)

- • 𝑚𝑡: EMA of gradients (direction memory)
- • 𝑣𝑡: EMA of squared gradients (scale/variance memory)
- • ⊙: elementwise product


𝑣𝑡−1,𝑖tells the expected squared gradient size for parameter 𝑖.

- Step B: Normalize current gradient by expected scale Define the elementwise normalized (preconditioned) gradient:


𝑔𝑡 √𝑣𝑡−1 + 𝜀

𝑔̃𝑡:=

Per coordinate:

𝑔𝑡,𝑖 √𝑣𝑡−1,𝑖 + 𝜀

𝑔̃𝑡,𝑖 =

Why square root? Because 𝑣tracks a squared quantity, so √𝑣matches the units of 𝑔. This matches Adam’s own update logic.

###### What does 𝒈̃𝒕,𝒊tell we?

Think of 𝑔̃𝑡,𝑖as a “z-score-ish” value but using Adam’s variance memory.

- Case 1: ∣ 𝒈̃𝒕,𝒊 ∣≈ 𝟏 This means:

• 𝑔𝑡,𝑖is about the same size as what 𝑖 usually sees. So nothing special is happening on that parameter.

- Case 2: ∣ 𝒈̃𝒕,𝒊 ∣≫ 𝟏 This means:

• 𝑔𝑡,𝑖is much bigger than normal for that parameter. So this parameter is experiencing an unusually strong push — surprising.

- Case 3: ∣ 𝒈̃𝒕,𝒊 ∣≪ 𝟏 This means:


• 𝑔𝑡,𝑖is much smaller than its usual scale. So the sample isn’t really pushing this parameter much right now.

###### Step C: Aggregate into a scalar surprise score (module-level)For module 𝑗 with 𝑑𝑗parameters, define Adam-based surprise:Squared (“whitened energy”) version

𝑗,adam):=

1 𝑑𝑗

𝑆𝑡(

∥ 𝑔̃𝑡(𝑗) ∥22=

𝑑𝑗

2

(𝑔𝑡,𝑖(𝑗))

1 𝑑𝑗

∑

𝑣𝑡−1,𝑖(𝑗) + 𝜀

𝑖=1

“On average, how unusually large are the gradients in this module compared to what Adam expects?”

Why “whitened energy”? Because it measures gradient energy after dividing by expected per-coordinate scale (like a diagonal Mahalanobis energy). If gradients are normal, it stays near baseline; if many coordinates spike beyond expectation, it rises.

###### Alternative norm version

𝑗,adam-norm):=

1 √𝑑𝑗

𝑆𝑡(

∥ 𝑔̃𝑡(𝑗) ∥2

Relationship:

2

𝑗,adam) = (𝑆𝑡(

𝑗,adam-norm))

𝑆𝑡(

Squared emphasizes outliers more; norm is easier to interpret as average normalized magnitude.

###### Why Adam-based surprise is better than raw norm surprise

Raw EMA surprise uses only total norm 𝑟𝑡(𝑗), which can hide structure:

- • a big gradient in a usually volatile parameter may not be surprising
- • a moderate gradient in a usually quiet parameter may be very surprising


Adam-based surprise fixes this by comparing each coordinate to its own history, so it’s more adaptive, scale-invariant, and aligned with the optimizer’s own “expected gradient behavior.”

###### Practical notes (important for implementation) — module-level

- 1. Use 𝑣𝑡−1, not 𝑣𝑡, for causal surprise Same reason as before: compare current gradient against prior memory.
- 2. Compute per module / parameter group Do not compute one surprise score over the entire model unless we normalize by group, because modules have very different gradient scales. So for module 𝑗:


𝑆𝑡(𝑗) =

𝑑𝑗

2

(𝑔𝑡,𝑖(𝑗))

1 𝑑𝑗

∑

𝑣𝑡−1,𝑖(𝑗) + 𝜀

𝑖=1

and compute separate surprises like:

- • 𝑆𝑡(𝐹): fast store module(s)
- • 𝑆𝑡(𝑃): permanent store module(s)
- • 𝑆𝑡(trunk): shared trunk so the controller can decide better.


- 3. Bias correction (optional) Early in training, Adam moments are biased toward zero. We may use biascorrected 𝑣̂𝑡, but for routing signals many implementations can get away with raw 𝑣𝑡after warmup.
- 4. Clamp / extreme protection For routing stability, clip surprise scores:


𝑆𝑡(𝑗) ← clip(𝑆𝑡(𝑗),𝑆min,𝑆max)

to prevent one pathological batch from dominating decisions.

Why Adam-based surprise is “better” than raw EMA surprise {According to ChatGPT} The scalar EMA version uses only total norm 𝑟𝑡, which can hide structure:

- • A big gradient in a usually volatile parameter may not be surprising
- • A moderate gradient in a usually quiet parameter may be very surprising Adam-based surprise fixes this by comparing each coordinate to its own history. So it is:
- • more adaptive
- • more scale invariant
- • more aligned with the optimizer’s own notion of expected gradient behavior


That makes it especially suitable for our “optimizer-as-memory” idea.

###### Practical notes (important for implementation)

- 1) Use 𝒗𝒕−𝟏, not 𝒗𝒕, for causal surprise Same reason as before: compare current gradient against prior memory.
- 2) Compute per module / parameter group

Do not compute one surprise score over the entire model unless we normalize by group. Different modules have very different gradient scales.

So for module 𝑗:

𝑆𝑡(𝑗) =

1 𝑑𝑗

∑

(𝑔𝑡,𝑖(𝑗))2 𝑣𝑡−1,𝑖(𝑗) + 𝜀

𝑑𝑗

𝑖=1

- 3) Bias correction (optional)

If we are early in training, Adam moments are biased toward zero. We may optionally use bias-corrected 𝑣̂𝑡, but for routing signals, many implementations can get away with raw 𝑣𝑡after a warmup.

- 4) Clamp / extreme protection (why clip surprise?) What the line means


We have a surprise score 𝑆𝑡. Sometimes one batch/example can be “pathological” (weird, noisy, corrupted, or just extremely hard), producing a huge 𝑆𝑡.

So we do:

𝑆𝑡 ← clip(𝑆𝑡,𝑆min,𝑆max)

That just means:

- • if 𝑆𝑡is too small, set it to 𝑆min
- • if 𝑆𝑡is too large, set it to 𝑆max


Why it helps routing stability Our controller uses surprise to decide “store to R vs F vs P”. If one extreme batch produces a massive surprise spike, it could:

- • force the policy to overreact (e.g., always write to R or always write to F)


- • dominate EMA baselines and mess up the next many steps
- • cause unstable, jittery store decisions


So clipping is basically: “Don’t let one crazy example hijack the controller.” It’s a controller-safety trick.

### 2) Stability / Noise Signal — what are we measuring?

The core question When we see a gradient 𝑔𝑡, we want to know: “Is this gradient part of a consistent learning direction… or is it fighting/undoing what we’ve been learning recently?” Because:

- • Stable / consistent trend → safer to put into P (permanent)
- • Noisy / conflicting → safer to stage into F, or even R first Important point we already wrote (and it’s correct): Stability is not “gradient is small”. It’s “gradient direction is consistent over time.”


###### 2.1 Directional stability from momentum (cosine with memory)

- Step 1) What is 𝒎𝒕−𝟏? Momentum (or Adam’s first moment) is basically: a smoothed average of recent gradients So we can treat 𝑚𝑡−1as:


- • the model’s “recent preferred update direction”
- • a short-term memory of what training has been doing lately


- Now 𝑔𝑡is what this new sample wants right now. So comparing 𝑔𝑡to 𝑚𝑡−1tells we:
- • does the new sample agree with recent learning?
- • or does it fight it? That’s exactly what “stability vs conflict” means.


- Step 2) Why cosine similarity? Cosine similarity measures direction agreement without being confused by size.


⟨𝑔𝑡, 𝑚𝑡−1⟩ ∥ 𝑔𝑡 ∥2⁡⁡∥ 𝑚𝑡−1 ∥2+ 𝜀

𝐶𝑡 = cos⁡(𝑔𝑡,𝑚𝑡−1) =

Breakdown:

- • ⟨𝑔𝑡,𝑚𝑡−1⟩= dot product (agreement in direction)
- • divide by norms = removes magnitude effect
- • 𝜀= avoids divide-by-zero


Range is about [−1,1]. Why not dot product? Dot product mixes:

- • direction and
- • magnitude


But we specifically want: “Are we pushing the same way, regardless of how hard?” So cosine is the right tool.

###### Step 3: Interpretation (very important)

𝑪𝒕 ≈ 𝟏 Current gradient points in nearly the same direction as optimizer memory.

- • learning is consistent
- • this sample reinforces what recent samples oure already teaching
- • likely a stable pattern / recurring skill 𝑪𝒕 ≈ 𝟎 Current gradient is nearly orthogonal to the stored direction.
- • unrelated to recent trend
- • could be noise
- • could be a different subtask
- • signal is uncertain


𝑪𝒕 < 𝟎 Current gradient points against recent trend.

- • active conflict / reversal
- • this update may undo recent learning
- • high risk of instability or forgetting if written to P


Step 4) Turn alignment into a “badness” score Sometimes we want the controller to use a score where:

• higher = worse (more noise/conflict)

###### Option A: “anything not aligned is bad”

𝑁𝑡(𝑑𝑖𝑟) = 1 − 𝐶𝑡

So:

- • 𝐶𝑡 = 1 ⇒ 𝑁 = 0(best)
- • 𝐶𝑡 = 0 ⇒ 𝑁 = 1
- • 𝐶𝑡 = −1 ⇒ 𝑁 = 2(worst)


This treats:

- • orthogonal (0) as moderately bad
- • negative as very bad


Use this when we want a general stability measure (agreement is good; everything else is less good).

###### Option B: “only true conflict is bad”

𝑁𝑡(𝑑𝑖𝑟) = max⁡(0,−𝐶𝑡)

So:

- • if 𝐶𝑡 ≥ 0: 𝑁 = 0(no penalty)
- • if 𝐶𝑡 < 0: 𝑁 = −𝐶𝑡(penalize only conflict)


Use this when our goal is specifically: “Don’t let destructive updates reach P” This is often better for R/F/P because:

- • orthogonal might just mean “new skill” (not necessarily dangerous)
- • negative means “interference” (dangerous)


###### Practical R/F/P intuition (how the controller uses it)

- • High 𝐶𝑡+ high repetition → ok to promote F → P
- • Low 𝐶𝑡(near 0) → keep in F (learn a bit, observe if it stabilizes)
- • Negative 𝐶𝑡→ treat as conflict; prefer R (store) or F with protection; block P writes


###### One important implementation note (small but real)

If ∥ 𝑚𝑡−1 ∥is tiny early in training (or for a rarely-updated module), cosine becomes unstable. Typical fix:

• if ∥ 𝑚𝑡−1 ∥< warmup_thr, set 𝐶𝑡 = 0(unknown) or skip stability signal for that step.

###### 2.2 Variance-style stability from Adam moments

This one is more “statistical” and very elegant because it directly uses Adam’s internal memory.

###### Step 1: What Adam’s moments mean (intuition)Adam keeps two running memories for each parameter 𝑖:𝒎𝒕(first moment)

- • Think of 𝑚𝑡as: the average direction the gradients have been pushing recently.
- • If gradients often point in the same direction, 𝑚𝑡becomes large (in that direction).
- • If gradients keep flipping + and −, they cancel and 𝑚𝑡stays small. 𝒗𝒕(second moment)
- • Think of 𝑣𝑡as: how much gradient activity there has been. • It grows when gradients are often large (regardless of sign). • It measures magnitude/variability, not direction.


So:

- • 𝑚= direction consistency memory
- • 𝑣= how “active / volatile” the gradients are


𝟐

###### Step 2: The key ratio 𝝆𝒕,𝒊 = 𝒎𝒕−𝟏,𝒊

𝒗𝒕−𝟏,𝒊+𝜺

This ratio is a signal-to-noise idea. Why squaring 𝒎?

Because 𝑚is signed (+/−), but we want “how strong is the consistent direction,” not whether it’s positive or negative.

So 𝑚2measures “how strong the average direction is.” Why divide by 𝒗?

Because 𝑣 is like “total gradient energy” (how big gradients have been overall). So:

𝑚2 𝑣

means: “Out of all the gradient activity, how much is consistent direction, rather than random fluctuation?”

Step 2.5: The statistical identity behind it (why this is elegant) Classic Expectation formula

𝔼[𝑔2] = (𝔼[𝑔])2 + Var(𝑔)

Interpretation:

- • total squared energy = squared mean (consistent part) + variance (noise part) Adam’s moments roughly approximate:
- • 𝑚 ≈ 𝔼[𝑔] • 𝑣 ≈ 𝔼[𝑔2]


So:

𝑚2 𝑣

≈

(𝔼[𝑔])2 (𝔼[𝑔])2 + Var(𝑔)

Now we can see why it’s a stability score:

- • If variance is small (stable gradients), denominator ≈ numerator → ratio close to 1
- • If variance is huge (noisy gradients), numerator small compared to denominator → ratio near 0


That’s the entire idea.

###### Step 3: Interpreting 𝝆𝒕,𝒊High 𝝆𝒕,𝒊

- • gradients keep reinforcing the same direction
- • Adam’s “belief” about how this parameter should move is strong
- • this parameter is learning consistently This is a good sign for:
- • consolidation (F → P)
- • trusting this learning as not just noise Low 𝝆𝒕,𝒊
- • gradients are active but cancel out (flip directions)
- • lots of activity but no consistent direction
- • suggests conflict/interference or noisy data This means:
- • don’t consolidate yet
- • maybe store in R or keep in F until it stabilizes


###### Step 4: Why we average over a module

Our routing actions are module-level (“write to FFN layer 10”, “write to attention layer 7”), not per-weight.

So we aggregate:

Stab𝑡(𝑎𝑑𝑎𝑚) =

𝑑

𝑚𝑡−1,𝑖2 𝑣𝑡−1,𝑖 + 𝜀

1 𝑑

∑

𝑖=1

Interpretation:

- • high: module gradients are directionally consistent overall
- • low: module is noisy / conflicting overall


This gives one scalar stability score for that module.

- Step 5: Optional “noise score” version If our policy prefers “higher = worse/noisier”, define:


𝑁𝑡(𝑎𝑑𝑎𝑚) = 1 − Stab𝑡(𝑎𝑑𝑎𝑚)

So:

- • 𝑁high → noisy/unstable
- • 𝑁low → stable Why clip? Because in practice:
- • bias correction, numerical issues, or weird moment states can produce values slightly outside the clean 0–1 intuition. Clipping keeps the policy robust. Why this is powerful for our method (R/F/P) This metric is special because it doesn’t only look at the current gradient. It asks: “Over recent steps, has this module been learning in a consistent direction?” That’s exactly what we want for Nested Learning / “optimizer as memory”:
- • optimizer state = compressed history
- • stability ratio reads that history directly So in routing terms:
- • High surprise + high stability → safe to learn (F) and maybe consolidate (P) if repetition is high
- • High surprise + low stability → likely conflict/noise → avoid writing to P; maybe store in R or keep tentative in F


- • Low surprise → not much new info, minimal write


- 2.3 Windowed gradient variance (module-level, simple and strong) What this “windowed gradient variance” is trying to answer For a module 𝑗, we want to know: “Is the learning signal for this module steady over recent steps, or is it jumping around?” Because:


- • steady signal usually means the module is learning something consistent
- • volatile signal often means noise, task switching, or teacher conflicts And we want this without relying on Adam internals.


- Step 0: Make it module-level Instead of global 𝑟𝑡, define:

- • module gradient: 𝑔𝑡(𝑗)
- • module gradient norm:


𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2

Everything below should be per module 𝑗.

- Step 1: Track the mean and mean-square of 𝒓𝒕(𝒋) We keep two EMAs:


- (A) EMA of the norm (mean)

𝑎𝑡(𝑗) = (1 − 𝛼)𝑎𝑡−1(𝑗) + 𝛼𝑟𝑡(𝑗)

This is: “what is the typical gradient norm for module 𝑗recently?”

- (B) EMA of the squared norm (mean-square)


𝑏𝑡(𝑗) = (1 − 𝛼)𝑏𝑡−1(𝑗) + 𝛼(𝑟𝑡(𝑗))2

This is: “what is the typical squared norm recently?” Why track both? Because variance can be computed from mean and mean-square.

###### Step 2: Build the variance proxyIn statistics:

Var(𝑟) = 𝔼[𝑟2] − (𝔼[𝑟])2

Here:

- • 𝑏𝑡(𝑗) ≈ 𝔼[𝑟2]
- • 𝑎𝑡(𝑗) ≈ 𝔼[𝑟] So:


𝑏𝑡(𝑗) − (𝑎𝑡(𝑗))2

acts like a running variance estimate of the module’s gradient norm.

###### Step 3: Normalize it (why divide by 𝒂𝒕𝟐?)We define:

𝑏𝑡(𝑗) − (𝑎𝑡(𝑗))2 (𝑎𝑡(𝑗))2+𝜀

𝑉𝑡(𝑗):=

Why this is important Raw variance depends on scale. Example:

- • Big modules naturally have larger gradients → bigger variance
- • Small modules naturally have smaller gradients → smaller variance


- That would make the stability threshold unfair across modules. Dividing by (𝑎𝑡(𝑗))2makes it roughly: “How variable is it relative to its typical size?” This becomes comparable across:
- • different modules
- • different training stages (early vs late) We can think of it like a “relative volatility” measure. (Yes: it’s similar to coefficient of variation squared.)


- Step 4: Interpretation (what high/low means)


Low 𝑽𝒕(𝒋)

- • the gradient norm is stable over recent steps
- • learning pressure magnitude is consistent
- • good sign for stable learning and consolidation


###### High 𝑽𝒕(𝒋)

- • gradient norm jumps around
- • learning is volatile
- • could mean:


- o noisy batches
- o switching skills/topics
- o conflicting teacher signals
- o unstable optimization dynamics


###### Important limitation This only measures magnitude volatility, not direction.

Two bad/good cases can look identical in 𝑉𝑡(𝑗):

- 1. Gradients are same size but flip direction (conflict)
- 2. Gradients are same size and same direction (stable learning)


Both can produce low variance of the norm → low 𝑉𝑡(𝑗). So we need a directional signal too.

How our “three stability signals” fit together We listed three complementary signals. Here’s the intuitive meaning of each:

- 1) Cosine with momentum 𝑪𝒕(𝒋) 𝐶𝑡(𝑗) = cos⁡(𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) )

Meaning: “Right now, is the gradient pointing in the same direction as recent updates?”

- • high → consistent direction
- • low/negative → conflict / direction flips


- 2) Adam ratio stability Stab𝒕(

𝒋,𝒂𝒅𝒂𝒎)

Stab𝑡(

𝑗,𝑎𝑑𝑎𝑚) =

1 𝑑𝑗

∑

(𝑚𝑡−1,𝑖(𝑗) )2 𝑣𝑡−1,𝑖(𝑗) + 𝜀

𝑖

Meaning: “Over time, is there a consistent signal compared to noise/variance per parameter?”

- • high → gradients have been consistently meaningful
- • low → mostly noisy / inconsistent (This is like “signal energy / noise energy” using Adam’s memory.)


- 3) Windowed norm variance 𝑽𝒕(𝒋)


𝑏𝑡(𝑗) − (𝑎𝑡(𝑗))2 (𝑎𝑡(𝑗))2+𝜀

𝑉𝑡(𝑗) =

Meaning: “Is the strength of the update stable or erratic?” So yes — these are complementary:

- • 𝐶= direction now
- • Adam ratio = historical consistency vs noise
- • 𝑉= magnitude volatility


Recommended controller usage (our rule, explained) Compute per module 𝑗:

- • 𝐶𝑡(𝑗)(direction)
- • Stab𝑡(𝑗)(Adam-based, if available) • 𝑉𝑡(𝑗)(optimizer-agnostic volatility)


Then a very readable decision logic is: “Stable” (good for consolidation / P)

- • 𝐶𝑡(𝑗)high (direction agrees)
- • 𝑉𝑡(𝑗)low (magnitude steady)
- • optionally Stab𝑡(𝑗)high


###### “Noisy / conflicting” (prefer F or R)

- • 𝐶𝑡(𝑗)low or negative (direction conflict)
- • or 𝑉𝑡(𝑗)high (magnitude volatile)
- • or Stab𝑡(𝑗)low (weak signal vs noise)


This is exactly the clean justification we want for F vs P gating.

Subtle note about Adam bias correction (we’re right) Early on, Adam’s 𝑚𝑡,𝑣𝑡are biased toward 0. Bias correction:

𝑚𝑡 1 − 𝛽1𝑡

𝑣𝑡 1 − 𝛽2𝑡

𝑚̂𝑡 =

,𝑣̂𝑡 =

Using 𝑚̂,𝑣̂makes the definition “paper clean,” especially if we evaluate early-training behavior.

In practice we can also just:

- • warm up the controller thresholds
- • or ignore stability decisions until some steps pass


# 3) Repetition Signal

Goal: “Is this pattern recurring or just a one-off?” We want a score that answers:

- • Recurring pattern → likely worth learning (F → P)
- • One-off / rare event → better for retrieval (R)


The challenge is: A single gradient spike does not tell we repetition. So we build proxies that infer repetition from:

- • optimizer memory (momentum / Adam state)
- • gradient behavior over time
- • optionally retrieval/embedding matches


- 3.1 Repetition via Persistent Directional Agreement (Optimizer-only proxy) Intuition first


If similar examples keep appearing, they tend to push the model in a similar gradient direction.

That means:

- • current gradient 𝑔𝑡keeps aligning with the optimizer’s memory 𝑚𝑡−1
- • momentum grows in a stable direction
- • this is evidence of repetition If examples are one-off or conflicting:
- • alignment is inconsistent
- • momentum does not build
- • repetition score stays low


###### The formula

𝑚𝑜𝑚,𝑗) = (1 − 𝛼𝑅)𝑅𝑡−1(

𝑚𝑜𝑚,𝑗) + 𝛼𝑅max⁡ ⁣ (0,cos⁡(𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) ))

𝑅𝑡(

“Here 𝑗 indexes a module/parameter group (e.g., 𝑗 = (𝑙,𝑏,𝑠)).” Let’s define every term.

𝑚𝑜𝑚,𝑗)

𝑹𝒕(

Our repetition score at step 𝑡 based only on optimizer/momentum behavior.

- • High = recurring/consistent pattern
- • Low = isolated/noisy/conflicting pattern 𝜶𝑹 ∈ (𝟎,𝟏) EMA smoothing factor.
- • Small 𝛼𝑅: slow-moving, stable estimate
- • Large 𝛼𝑅: reacts quickly to recent changes This is just an exponential moving average (EMA).


𝐜𝐨𝐬⁡(𝒈𝒕,𝒎𝒕−𝟏) Cosine similarity between current gradient and previous momentum memory: Then⁡define⁡cosine⁡similarity⁡as:

⟨𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) ⟩ ∥ 𝑔𝑡(𝑗) ∥2 ∥ 𝑚𝑡−1(𝑗) ∥2+ 𝜀

cos⁡(𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) ) =

Interpretation:

- • +1: perfectly aligned (same direction)
- • 0: unrelated / orthogonal
- • −1: opposite directions (conflict)


Why 𝐦𝐚𝐱⁡(𝟎,⋅)? We clip negative cosine to zero:

max⁡(0,cos⁡(𝑔𝑡,𝑚𝑡−1))

This means:

- • aligned gradients contribute positively to repetition
- • conflicting gradients do not count as repetition (instead of subtracting) That makes repetition score easier to interpret:
- • it measures positive recurrence, not conflict (Conflict can be captured separately by our stability/interference signal.)


###### Why this works (conceptually)

Momentum 𝑚𝑡−1is a compressed summary of recent gradients. If current gradient keeps agreeing with it, then the same type of learning pressure is recurring.

So this score is really: “How persistently does the current learning signal reinforce the optimizer’s memory?” That’s a very NL-consistent notion.

Behavior examples

- Case A: Repeating pattern


- • Similar samples arrive repeatedly
- • gradients align over time
- • cosine stays positive and often high
- • 𝑅𝑡(mom)rises


interpreted as recurring learnable pattern

###### Case B: One-off rare fact

- • One big gradient spike
- • no similar follow-up gradients
- • later cosines are low/unrelated
- • 𝑅𝑡(mom)decays


interpreted as non-recurring / better for R-store

###### Case C: Conflicting mixed signals

- • gradients flip directions
- • cosine often negative or near zero
- • clipping suppresses contribution
- • 𝑅𝑡(mom)remains low


not a stable recurring pattern

###### Practical notes

- • Compute per module/group: 𝑅𝑡,𝑗(mom)
- • If 𝑚𝑡−1 = 0early in training, define cosine = 0 (or skip until warmup)
- • We can use Adam’s first moment 𝑚𝑡−1exactly the same way


###### 3.2 Repetition via “Surprise Decay” Across Similar Samples (Better practical signal)

This is stronger than 3.1 because it uses semantic grouping (bucket/hash) instead of only direction agreement.

Intuition first If a pattern repeats, what should happen?

- • First few times: model is surprised → high 𝑆𝑡
- • After repeated exposure: model adapts → surprise decreases


So repetition can be inferred from:

- 1. Frequency: how often this kind of sample appears
- 2. Learning progress: surprise is going down over time This is exactly what our formula captures.


###### Step 1: Group similar samples into a bucket 𝒉(𝒙𝒕)We define a mapping:

ℎ(𝑥𝑡) ∈ ℋ

where ℎ(𝑥𝑡)is a bucket ID (cluster/hash/category) for the current example. Examples:

- • prompt template hash
- • topic/domain hash
- • nearest-neighbor cluster ID
- • retrieval signature bucket This lets we track repetition by pattern, not just by raw sample identity.


###### Step 2: Track surprise per bucket with an EMAWe wrote:

𝑆ˉ𝑡(ℎ) = (1 − 𝛼ℎ)𝑆ˉ𝑡−1(ℎ) + 𝛼ℎ𝑆𝑡 ⋅ 𝟏[ℎ(𝑥𝑡) = ℎ].

###### What this means

For each bucket ℎ, maintain a running average of surprise values for samples belonging to that bucket.

- • 𝑆𝑡: current surprise score (from Section 1)
- • 𝟏[ℎ(𝑥𝑡) = ℎ]: indicator function:


- o = 1 if current sample belongs to bucket ℎ


- o = 0 otherwise


So only the active bucket gets updated by 𝑆𝑡.

###### Important subtlety (practical correction)

As written, buckets that are not visited decay every step due to (1 − 𝛼ℎ)𝑆ˉ𝑡−1(ℎ), which may or may not be what we want.

Two versions are common:

- Version A (global-time EMA) — our current form All bucket stats decay every step. Good if we want recency emphasis.
- Version B (event-time EMA, often better) Update only when bucket ℎis visited:


(1 − 𝛼ℎ)𝑆ˉ𝑡−1(ℎ) + 𝛼ℎ𝑆𝑡, ℎ(𝑥𝑡) = ℎ 𝑆ˉ𝑡−1(ℎ), otherwise.

𝑆ˉ𝑡(ℎ) = {

This avoids artificial decay when the bucket is absent. For our method, Version B is usually easier to interpret.

###### Step 3: Count frequency of the bucket𝑛𝑡(ℎ)

is the number of times bucket ℎhas appeared up to step 𝑡. This is our raw repetition evidence. But raw counts can be noisy for small 𝑛, so we use a smoothed frequency confidence term:

𝑛𝑡(ℎ) 𝑛𝑡(ℎ) + 𝑘

,𝑘 > 0.

###### Why this form?

It grows from 0 to 1 smoothly:

- • if 𝑛𝑡(ℎ) = 0, confidence = 0
- • if 𝑛𝑡(ℎ)is small, confidence is low
- • if 𝑛𝑡(ℎ) ≫ 𝑘, confidence → 1 So this term says: “How much do I trust that this pattern really repeats (not just random coincidence)?”


- Step 4: Check whether surprise is decreasing Choose a lookback Δ(like 50 or 200 steps). Compare:


𝑆ˉ𝑡−Δ(ℎ) − 𝑆ˉ𝑡(ℎ)

Then pass through sigmoid to get a clean 0–1 score:

trend𝑡(ℎ) = 𝜎(𝑆ˉ𝑡−Δ(ℎ) − 𝑆ˉ𝑡(ℎ)) where:

- • Δ > 0: lookback gap
- • 𝜎(⋅): sigmoid function Interpretation Compute the difference:


𝑆ˉ𝑡−Δ(ℎ) − 𝑆ˉ𝑡(ℎ)

- • Positive → surprise has decreased since (𝑡−Δ)
- • Near 0 → no change
- • Negative → surprise increased (pattern may be shifting) Then sigmoid squashes it to (0,1):
- • near 1 if surprise is clearly decaying
- • near 0.5 if no strong trend


- • near 0 if surprise is increasing


This term answers: “Is the model getting less surprised by this pattern over time?” That is a signature of repeated learnable structure.

###### Final repetition score (hash-based)

𝑛𝑡(ℎ) 𝑛𝑡(ℎ) + 𝑘 ⏟ frequency confidence

𝑅𝑡(hash):=

⋅ 𝜎 ⁣(𝑆ˉ𝑡−Δ(ℎ) − 𝑆ˉ𝑡(ℎ))

.

⏟ surprise decay trend

Why multiplication? We only want a high repetition score when both are true:

- • the pattern occurs often enough (frequency confidence high)
- • surprise is actually decreasing (learning progress) If either is missing, repetition should be lower.


Behavior examples

- Case A: Repeating learnable pattern

- • bucket appears many times → 𝑛𝑡(ℎ)high
- • surprise decreases over time → positive trend
- • both factors high


𝑅𝑡(hash)high

- Case B: Frequent but still unresolved/conflicting pattern


- • bucket frequency high
- • but surprise does not decrease (or increases)
- • trend factor low


repetition score moderated (it repeats, but is not cleanly learnable yet) This is useful — may indicate teacher conflict or interference.

###### Case C: Rare one-off pattern

- • 𝑛𝑡(ℎ)small
- • frequency confidence low score stays low even if one surprise fluctuation looks favorable


###### Practical implementation notes

- • Use ℎ = ℎ(𝑥𝑡)as shorthand in code/math for the active bucket • Choose Δbased on training timescale (e.g., 50, 100, 500 steps) • We may normalize 𝑆𝑡before bucket EMA so buckets are comparable
- • Keep a minimum count before trusting trend (e.g., require 𝑛𝑡(ℎ) ≥ 𝑛min)


- 3.3 Repetition via Retrieval Hit Rate (External-memory assisted) This is the most direct semantic repetition measure. Intuition first


If the current example is similar to many recent examples in embedding space, then the pattern is recurring.

This avoids relying only on gradient geometry.

- Step 1: Maintain a recent embedding buffer Let ℬ𝑡be a buffer of embeddings from recent samples. For current sample 𝑥𝑡, compute embedding:


𝑒𝑡 = 𝐸(𝑥𝑡),

where 𝐸(⋅)is an encoder / embedding function. This could be:

- • student encoder representation
- • teacher embedding
- • frozen sentence encoder
- • retrieval encoder


###### Step 2: Count near-neighbors above threshold

𝐻𝑡:= ∑ 𝟏[cos⁡(

𝑒𝑡, 𝑒) ≥ 𝜏].

𝑒∈ℬ𝑡

What each term means

- • cos⁡(𝑒𝑡,𝑒): cosine similarity between current embedding and buffer embedding
- • 𝜏: similarity threshold (e.g., 0.8, 0.9 depending on embedding space)
- • 𝟏[⋅]: indicator = 1 if similarity passes threshold
- • sum = total number of “similar recent examples”


So 𝐻𝑡is a hit count.

Why this is a repetition signal

- • Large 𝐻𝑡: many near-duplicates / semantically similar samples recently seen
- • Small 𝐻𝑡: current sample is novel or rare This captures recurrence more directly than optimizer-only proxies.


###### Step 3: Normalize to [0,1]

###### Why this function?

𝑅𝑡(ret):= 1 − exp⁡(−𝐻𝑡/𝜅).

It is a smooth saturating transform:

- • if 𝐻𝑡 = 0, score = 0
- • as 𝐻𝑡grows, score increases
- • eventually saturates near 1 𝜅 > 0controls how quickly it saturates:
- • small 𝜅: a few hits already count as high repetition
- • large 𝜅: need many hits This avoids unbounded counts and makes the score comparable to other signals. Behavior examples
- • 𝐻𝑡 = 0→ 𝑅𝑡(ret) = 0(novel/rare)
- • 𝐻𝑡 = 𝜅→ 𝑅𝑡(ret) = 1 − 𝑒−1 ≈ 0.632
- • 𝐻𝑡 = 3𝜅→ 𝑅𝑡(ret) ≈ 0.95 So the score quickly becomes “high” when strong repetition evidence accumulates.


How these three repetition scores differ

- 1) 𝑹𝒕(𝐦𝐨𝐦)— optimizer-only

- • cheapest
- • no extra memory/index
- • weak semantic resolution
- • good fallback baseline


- 2) 𝑹𝒕(𝐡𝐚𝐬𝐡)— surprise decay by bucket


- • stronger and more task-aware
- • captures “repeats and becomes learnable”
- • depends on quality of bucketing ℎ(⋅)


###### 3) 𝑹𝒕(𝐫𝐞𝐭)— embedding retrieval hit rate

- • strongest semantic recurrence signal
- • extra system cost (buffer/index + embeddings)
- • pairs very well with R-store logic


Recommended way to use them together (practical) Use a fused repetition score:

𝑅𝑡 = 𝜆1𝑅𝑡(mom) + 𝜆2𝑅𝑡(hash) + 𝜆3𝑅𝑡(ret),𝜆𝑖 ≥ 0, ∑𝜆𝑖

= 1.

𝑖

If no retrieval buffer is available, set 𝜆3 = 0. This gives us:

- • optimizer consistency
- • trend of learnability
- • semantic recurrence (if available)


###### One small improvement we may want in our writeup For 3.2, explicitly define ℎ:= ℎ(𝑥𝑡)and write:

𝑛𝑡(ℎ) 𝑛𝑡(ℎ) + 𝑘

𝑅𝑡(hash) =

⋅ 𝜎 ⁣(𝑆ˉ𝑡−Δ(ℎ) − 𝑆ˉ𝑡(ℎ)),ℎ = ℎ(𝑥𝑡).

That makes the notation cleaner and avoids ambiguity.

##### The full pipeline (measurement → policy)

- Step A — compute raw gradient (physics of learning) 𝑔𝑡 = ∇𝜃𝐿𝑡(𝜃𝑡)
- Step B — measure per module (sensors) For each module 𝑗:

𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2

plus direction & variance signals.

- Step C — policy decides store (R/F/P) Using those signals:

- • if rare/one-off → R
- • if repeating and learnable → F
- • if highly repeating and stable → P


- Step D — apply per-parameter mask (how to write) If store = F (LoRA), then Top-K decides which LoRA components update. So Top-K is “write precision,” not “where to store.”


How to decide what we “use” in the main method

- 3.1 Momentum agreement (optimizer-only)

- • Keep as a cheap baseline / fallback
- • Works even without bucketing or retrieval
- • But weaker semantics


- 3.2 Surprise decay in buckets (best practical default)


- • Best main choice if we can define a decent bucket ℎ(𝑥)
- • Captures “it repeats and becomes learnable”


- • Strong for our R/F/P story


###### 3.3 Retrieval hit rate (strongest but extra system cost)

- • Use if we already have embeddings + buffer/index
- • Great for identifying true semantic repeats
- • Adds compute + memory


Recommended setup for our case (simple + strong) Since we’re already doing module-level routing and we’ll likely have some bucketing: Main repetition signal (use in method): 3.2 (bucket surprise decay) plus optionally a small weight of 3.1 for robustness Ablation / optional extension: add 3.3 if we implement retrieval buffer

###### 5) Super clear mapping

Per-module signals = Where is the learning happening? Per-parameter signals = Within that place, which parts matter? R/F/P policy = Where should this knowledge be stored? Top-K / LoRA write rules = How do we store it safely and sparsely?

#### Overall Controller Flow (Summary, AdamW Setting)

This section summarizes how the surprise, repetition, and stability signals combine into a single module-level R/F/P routing policy when the student is trained with Adam/AdamW.

Notation (module-level) We index a module by

𝑗 = (𝑙,𝑏,𝑠),

where:

- • 𝑙: layer index
- • 𝑏 ∈ {attn,ffn}: architecture block type
- • 𝑠 ∈ {𝑃,𝐹}: parameter subset (base/permanent vs LoRA/fast) For each step 𝑡, we compute the module-restricted gradient:


𝑔𝑡(𝑗):= ∇𝜃(𝑗)𝐿(𝜃𝑡;𝑥𝑡).

###### Step 0: Compute Adam moments (available “for free”)

Adam/AdamW maintains per-parameter first and second moments. Restrict them to module 𝑗:

- • 𝑚𝑡−1(𝑗) : first moment (EMA of gradients)
- • 𝑣𝑡−1(𝑗) : second moment (EMA of squared gradients) These are reused by the controller as memory signals.


###### Step 1: Surprise (scale-aware novelty / pressure)Using Adam’s second moment, define a scale-normalized surprise for module 𝑗:

𝑆𝑡(𝑗):=

𝑑𝑗

2

(𝑔𝑡,𝑖(𝑗))

1 𝑑𝑗

∑

𝑣𝑡−1,𝑖(𝑗) + 𝜀

𝑖=1

Interpretation:

• high 𝑆𝑡(𝑗)means many coordinates are larger than their usual scale → novel /

###### difficult / high-pressure

• low 𝑆𝑡(𝑗)means gradients are within expected range → not surprising This is our primary “should we write?” signal.

- Step 2: Repetition (recurrence evidence)

Compute a repetition score 𝑅𝑡(𝑗) ∈ [0,1]that estimates whether the pattern behind the current sample is recurring rather than one-off. Practically, 𝑅𝑡(𝑗)may be computed by any of:

- • optimizer-only directional recurrence (momentum cosine EMA)
- • bucketed surprise-decay recurrence
- • retrieval hit-rate recurrence or a weighted combination. Interpretation:
- • high 𝑅𝑡(𝑗): recurring pattern → more worth learning into weights
- • low 𝑅𝑡(𝑗): rare/isolated → better for retrieval memory This is our main “R vs (F/P)?” signal.


- Step 3: Stability (is it safe to consolidate?) Stability is not one number; we use two complementary checks:


- 3.1 Directional consistency (instant)


𝐶𝑡(𝑗):= cos⁡(𝑔𝑡(𝑗),𝑚𝑡−1(𝑗) )

Interpretation:

- • high 𝐶𝑡(𝑗): current update aligns with recent trend → consistent direction
- • low/negative 𝐶𝑡(𝑗): conflict / direction flips


###### 3.2 Magnitude volatility (windowed, optimizer-agnostic)

Let 𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2. Maintain EMAs: 𝑎𝑡(𝑗) = (1 − 𝛼)𝑎𝑡−1(𝑗) + 𝛼𝑟𝑡(𝑗),𝑏𝑡(𝑗) = (1 − 𝛼)𝑏𝑡−1(𝑗) + 𝛼(𝑟𝑡(𝑗))2

Define normalized windowed variance:

𝑏𝑡(𝑗) − (𝑎𝑡(𝑗))2 (𝑎𝑡(𝑗))2+𝜀

𝑉𝑡(𝑗):=

Interpretation:

- • low 𝑉𝑡(𝑗): update strength is steady → stable regime
- • high 𝑉𝑡(𝑗): update strength is erratic → noisy/mixed regime


Why keep 𝑉𝑡(𝑗)even with Adam? Because it captures module-level volatility directly (mixture/teacher switching/noisy bursts) and serves as a robust safety check before making updates permanent.

- Step 4: The R/F/P routing policy (module-level action) For each module 𝑗, the controller outputs an action:


𝑎𝑡(𝑗) ∈ {R,F,P}.

A simple and strong rule-set is:

- A) Route to R (retrieval) — “surprising but not recurring” If:


- • 𝑆𝑡(𝑗)is high (novel pressure)
- • 𝑅𝑡(𝑗)is low (not repeating) Then:
- • do not update weights
- • store the example/embedding/teacher response in retrieval Interpretation: one-off novelty → don’t overwrite model weights.


###### B) Route to F (fast / LoRA) — “recurring but not yet stable”If:

- • 𝑆𝑡(𝑗)is high/moderate (there is learning pressure)
- • 𝑅𝑡(𝑗)is medium/high (repeats)
- • but stability is not proven, e.g.:


Then:

o 𝐶𝑡(𝑗)not consistently high, and/or o 𝑉𝑡(𝑗)is high

- • update fast parameters only (LoRA/adapters), i.e. 𝑠 = 𝐹
- • optionally apply Top-K masking inside LoRA for sparse writes


Interpretation: it repeats, but dynamics are messy → learn quickly in a reversible store.

###### C) Route to P (permanent) — “recurring and stable”If:

- • 𝑅𝑡(𝑗)is high (clearly repeating)
- • and stability is strong:


- o 𝐶𝑡(𝑗)high (direction consistent)


Then:

- o 𝑉𝑡(𝑗)low (magnitude stable) (plus optionally “stability has remained high for multiple windows”)


• consolidate into base weights 𝑠 = 𝑃, e.g.: o merge/distill fast LoRA knowledge into base o or directly allow base updates with strong regularization

Interpretation: repeated + stable structure → safe to make permanent.

Practical implementation detail: “pressure” vs “write” It is useful to distinguish:

- • pressure signals (computed from raw 𝑔𝑡(𝑗))
- • write amount (computed after masking/gating, e.g., LoRA-only + Top-K) This keeps the controller interpretable:
- • pressure explains why the policy chose an action
- • write amount tracks how much memory was actually written


One-line conceptual summary Surprise decides whether the sample carries unusual learning pressure, Repetition decides whether it is worth learning into weights versus retrieval, and Stability decides whether it is safe to consolidate into permanent memory (P) or keep it in fast memory (F).

