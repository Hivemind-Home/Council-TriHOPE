# Distillation Training Algorithm

Let the teacher set be

𝒯 = {𝑇1,𝑇2,…,𝑇𝐾}

and let the student model be 𝑆𝜃with parameters 𝜃. We use a shared embedding function

𝑓emb(𝑥) ∈ ℝ𝑑

to map each input sample into a common routing space. For each teacher 𝑇𝑘, we precompute a prototype vector

𝑁𝑘

1 𝑁𝑘

(𝑥𝑖(𝑘)),

𝑐𝑘 =

∑𝑓emb

𝑖=1

where {𝑥𝑖(𝑘)}𝑖=1𝑁𝑘 are seed examples associated with teacher 𝑇𝑘. The role of this block is only to answer: which teacher should supervise this sample?

This corresponds to the teacher-selection stage in our proposal, while the Theory 101 machinery begins only after the loss has produced gradients.

## Step 1. Sample embedding and teacher selection

For each training sample 𝑥𝑡, compute its shared routing embedding: ℎ𝑡 = 𝑓emb(𝑥𝑡)

Then score each teacher prototype by cosine similarity:

ℎ𝑡⊤𝑐𝑘 ∥ ℎ𝑡 ∥2∥ 𝑐𝑘 ∥2

𝑠𝑘(𝑥𝑡) = cos⁡(ℎ𝑡,𝑐𝑘) =

Select the supervising teacher by hard routing: 𝑘𝑡∗ = arg⁡ max⁡

𝑠𝑘(𝑥𝑡)

𝑘∈{1,…,𝐾}

and denote the selected teacher by

𝑇𝑡∗ = 𝑇𝑘𝑡∗.

This stage replaces the traditional fixed one-student one-teacher setup with per-sample teacher assignment. Our proposal already defines teacher routing as the first stage before the downstream memory/write routing.

Before jumping to the next section what is shared embedding here? The shared routing embedding is just a common vector representation of the input sample that we use to decide which teacher should handle that sample. In our setup, it means:

- • take the input 𝑥
- • pass it through one embedding function
- • get one vector ℎ(𝑥) ∈ ℝ𝑑
- • compare that vector against each teacher’s prototype
- • choose the closest teacher


So “shared” means: the same embedding space is used for all teachers. Not one latent space for coding teacher, another for math teacher, another for medical teacher. Instead, all samples are mapped into one common routing space, so comparison is possible. Formula

ℎ(𝑥) = 𝑓emb(𝑥)

where:

- • 𝑥= input sample
- • 𝑓emb= embedding model / encoder
- • ℎ(𝑥)= shared routing embedding


Then teacher selection becomes:

𝑠𝑘(𝑥) = cos⁡(ℎ(𝑥),𝑐𝑘)

where 𝑐𝑘is teacher 𝑘’s prototype.

Why “shared” is important Because if we use raw hidden states from different teachers:

- • their dimensions may differ
- • their internal geometry may differ
- • direct comparison is unreliable


So instead of comparing teacher latent spaces directly, we create one neutral comparison space.

That neutral space is the shared routing embedding space.

Very simple intuition Suppose we have 4 teachers:

- • coding
- • math
- • medical
- • general


Now a new sample comes in: “Write a Python function for binary search.”

The shared routing embedding converts this sentence into a vector. That vector will likely land closest to the coding teacher prototype.

Another sample: “What is the derivative of 𝑥2sin⁡ 𝑥?” Its embedding lands closer to the math teacher prototype. So the shared routing embedding is basically the space where samples “live” before teacher selection.

Now again resume:

## Step 2. Teacher and student predictive distributions

Let the selected teacher 𝑇𝑡∗produce logits 𝑧𝑇𝑡∗(𝑥𝑡), and let the student produce logits 𝑧𝜃(𝑥𝑡).

Using temperature 𝜏 > 0, define:

𝑧𝑇𝑡∗(𝑥𝑡) 𝜏

𝜏 (𝑦 ∣ 𝑥𝑡) = softmax ⁣ (

𝑝𝑇

)

∗

𝑡

𝑧𝜃(𝑥𝑡) 𝜏

𝑝𝜃𝜏(𝑦 ∣ 𝑥𝑡) = softmax ⁣ (

)

These two distributions are the only objects used in the distillation objective.

## Step 3. Knowledge distillation loss

The temperature-scaled KD loss for sample 𝑥𝑡is

ℒKD(𝑡) = 𝜏2 KL (𝑝𝑇

𝜏 (⋅∣ 𝑥𝑡) ∥ 𝑝𝜃𝜏(⋅∣ 𝑥𝑡))

∗

𝑡

or equivalently,

ℒKD(𝑡) = 𝜏2 ∑ 𝑝𝑇

𝜏 𝑣∈𝒱

∗

𝑡

𝜏 (𝑣 ∣ 𝑥𝑡) 𝑝𝜃𝜏(𝑣 ∣ 𝑥𝑡)

𝑝𝑇

∗

𝑡

(𝑣 ∣ 𝑥𝑡)log⁡

.

This matches the KD-based training objective used in our notes theory 101.

## Step 4. Optional supervised loss

If a gold target 𝑦𝑡exists, add cross-entropy: ℒCE(𝑡) = −log⁡𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡)

For sequence generation, this is the token-level sum over the target sequence.

Lets deep dive into this The KD loss makes the student copy the teacher’s probability distribution, but sometimes we also have the actual correct answer for the sample. In that case, we can add a normal supervised learning loss so the student is not only imitating the teacher, but is also being pulled toward the ground-truth target. This matches our draft where the total loss can include both KD and CE.

### 1. What the symbols meanWe wrote:

ℒ𝐶𝐸(𝑡) = −log⁡𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡)

Here:

- • 𝑥𝑡= the input sample at training step 𝑡
- • 𝑦𝑡= the true target label or true output for that sample
- • 𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡)= the probability that the student assigns to the correct answer
- • −log⁡(⋅)= negative log-likelihood


So this loss says:

- • if the student gives high probability to the correct answer, loss is small
- • if the student gives low probability to the correct answer, loss is large


- 2. Why add CE if we already have KD? Because teacher imitation is not always enough. KD tells the student: “match what the selected teacher believes.” CE tells the student: “match the actual correct answer.”

So KD transfers richer soft knowledge, while CE keeps the student anchored to the real label. Our notes describe the full objective exactly this way: KD + CE + optional regularization.

- 3. Classification case For a simple classification task, suppose:


- • input 𝑥𝑡: a medical question
- • true label 𝑦𝑡: “diabetes”
- • student predicts probabilities over all classes


If the student says:

𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡) = 0.9

then

which is small. If instead the student says:

then

which is much larger.

ℒ𝐶𝐸(𝑡) = −log⁡(0.9)

𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡) = 0.1

ℒ𝐶𝐸(𝑡) = −log⁡(0.1)

So the CE loss strongly penalizes wrong or low-confidence predictions on the true label.

- 4. For sequence generation, what changes? In generation, the target is not one class. It is a whole output sequence. Suppose the target answer is a token sequence:

𝑦𝑡 = (𝑦𝑡,1,𝑦𝑡,2,…, 𝑦𝑡,𝑛)

Then the model generates one token at a time. So instead of one CE term, we sum over all target tokens:

ℒ𝐶𝐸(𝑡) = − ∑ log⁡

𝑛

𝑚=1

𝑝𝜃(𝑦𝑡,𝑚 ∣ 𝑥𝑡,𝑦𝑡,<𝑚)

Here:

- • 𝑦𝑡,𝑚= the correct token at position 𝑚
- • 𝑦𝑡,<𝑚= all previous correct tokens before position 𝑚
- • the model is trained to predict the next correct token at each step


This is what “token-level sum over the target sequence” means.

- 5. Simple sequence example Suppose input: Translate to French: “good morning” Target output tokens might be:


(bonjour)

or if tokenized into smaller pieces, maybe: (𝑦𝑡,1, 𝑦𝑡,2)

Then the model is trained so that:

- • at position 1, it gives high probability to the first correct token


- • at position 2, it gives high probability to the second correct token given the first


The total CE loss is the sum of those token losses. If the target sequence is longer, like code or reasoning text, we sum across all tokens in that target answer.

### 6. Why this matters in our setup

In our pipeline, the selected teacher gives a soft target distribution for KD, but if we also have gold outputs in the dataset, CE prevents the student from blindly inheriting teacher mistakes.

So our combined objective becomes:

ℒ(𝑡)(𝜃) = 𝜆𝐾𝐷ℒ𝐾𝐷(𝑡) + 𝜆𝐶𝐸ℒ𝐶𝐸(𝑡) + 𝜆𝑟𝑒𝑔ℒ𝑟𝑒𝑔(𝑡)

Meaning:

- • KD = learn the teacher’s behavior
- • CE = stay correct with respect to ground truth
- • regularization = keep training stable


## Step 5. Optional regularization

Add an optional stabilization term

ℒreg(𝑡)

to represent regularization such as anti-forgetting, trust-region, consistency, or weight penalties. Our notes explicitly allow KD + CE + regularization as the training loss.

### What 𝓛𝐫𝐞𝐠(𝒕) is doing

The regularization term is an extra control term added to the main student objective so that the student does not learn in an unstable or overly aggressive way.

Without it, the KD term may push the student strongly toward the selected teacher on the current sample, and the CE term may push it strongly toward the current label. That can work, but it may also cause:

- • unstable updates
- • overwriting of previously learned skills
- • too much drift in permanent weights
- • noisy adapter growth
- • poor compatibility with our later R/F/P routing


Our notes already describe this exactly: regularization is added to “stabilize training or protect memory,” and examples listed there include weight decay, consistency penalty, anti-forgetting penalty, trust-region / KL-to-old-student penalty, and sparsity penalty on adapters. The notes also explicitly say that changing the loss composition changes the gradient 𝑔𝑡, and therefore changes the Theory 101 routing signals. So what does it mean mathematically? Our full loss is:

ℒ(𝑡)(𝜃) = 𝜆KDℒKD(𝑡) + 𝜆CEℒCE(𝑡) + 𝜆regℒreg(𝑡) .

Here:

- • ℒKD(𝑡)tells the student to imitate the selected teacher
- • ℒCE(𝑡)tells the student to fit the true label if one exists
- • ℒreg(𝑡) tells the student how to learn more safely


So the regularization term does not usually teach new knowledge directly. Instead, it shapes the manner of updating.

### Why it matters especially in our setup

Our setup is not ordinary single-teacher distillation. It is multi-teacher distillation with downstream memory routing. That means one bad or overly large update can do more damage:

- • the selected teacher may be strong on the current sample but may conflict with older skills
- • the base model may drift too much
- • F-store adapters may become noisy
- • P-store updates may become too aggressive
- • the gradient statistics read by Theory 101 may become distorted


That is why our notes say regularization is especially relevant for the R/F/P idea and that strong regularization can make gradients smaller or more stable.

Common forms of 𝓛𝐫𝐞𝐠(𝒕) Here are the main types we already referenced, explained more properly.

- 1. Weight decay This penalizes very large parameter values:

ℒwd(𝑡) =∥ 𝜃 ∥22

or sometimes only on selected parameter groups. Purpose:

- • prevents weights from growing too large
- • improves optimization stability
- • reduces overfitting


This is the most basic form of regularization.

- 2. Consistency penalty


This encourages the student to behave consistently under small perturbations, dropout variations, or nearby input views.

Example form:

ℒcons(𝑡) = 𝐷 ⁣(𝑝𝜃(⋅∣ 𝑥𝑡),𝑝𝜃(⋅∣ 𝑥̃𝑡))

where 𝑥̃𝑡is a slightly perturbed version of the same sample and 𝐷may be KL divergence or MSE.

Purpose:

- • makes the student less brittle
- • reduces sensitivity to small noise
- • improves stable generalization


- 3. Anti-forgetting penalty This term discourages updates that damage previously learned knowledge. A generic form is:

ℒAF(𝑡) = ∑ 𝜔𝑖

𝑖

(𝜃𝑖 − 𝜃𝑖old)2

where 𝜃oldare older trusted parameters and 𝜔𝑖measures how important parameter 𝑖is for earlier skills.

Purpose:

- • protects previously consolidated knowledge
- • reduces catastrophic forgetting
- • especially important before writing into permanent memory This fits very naturally with our proposal’s goal of protecting long-term expert retention.


- 4. Trust-region or KL-to-old-student penalty


This is one of the most relevant ones for we. Our notes mention trust-region / KL-to-oldstudent explicitly.

The idea is: do not let the updated student move too far from its previous behavior in one step.

A behavioral version is:

ℒTR(𝑡) = KL ⁣(𝑝𝜃old(⋅∣ 𝑥𝑡) ∥ 𝑝𝜃(⋅∣ 𝑥𝑡))

Purpose:

- • prevents abrupt behavioral jumps
- • makes updates more conservative
- • especially useful when writing to P-store or base weights


This is very aligned with our framework because we often want fast learning in F, but controlled learning in P.

### 5. Sparsity penalty on adapters

If F-store uses LoRA or adapter parameters, we may want only a small part of those parameters to change strongly.

A simple form is:

ℒsparse(𝑡) =∥ 𝜙𝐹 ∥1

where 𝜙𝐹are fast-store adapter parameters. Purpose:

• keeps fast memory compact • avoids noisy over-adaptation • helps F-store remain targeted instead of diffuse

Our notes explicitly mention sparsity penalty on adapters as one possible regularizer.

## Step 6. Final distillation objective

The per-sample student loss is

ℒ(𝑡)(𝜃) = 𝜆KDℒKD(𝑡) + 𝜆CEℒCE(𝑡) + 𝜆regℒreg(𝑡) .

This is still purely the distillation block. At this stage, nothing from Theory 101 has been applied yet.

## Step 7. Backpropagation and gradient extraction

Compute the student gradient:

𝑔𝑡 = ∇𝜃ℒ(𝑡)(𝜃𝑡)

If needed at module level,

𝑔𝑡(𝑗) = ∇𝜃(𝑗)ℒ(𝑡)(𝜃𝑡)

with norm

𝑟𝑡(𝑗) =∥ 𝑔𝑡(𝑗) ∥2.

This is the exact handoff point. Our notes state that the controller reads the gradient induced by the training objective, rather than replacing normal backpropagation.

From here, the Theory 101 block starts Everything above is the distillation training algorithm only. The Theory 101 block starts after:

𝑔𝑡 = ∇𝜃ℒ(𝑡)(𝜃𝑡)

has already been computed. From this point onward, the downstream controller reads:

- • the gradient 𝑔𝑡
- • module gradients 𝑔𝑡(𝑗)
- • optimizer memory states 𝑚𝑡−1(𝑗) ,𝑣𝑡−1(𝑗) and then computes surprise, stability, repetition, and the final R/F/P write action. That separation is exactly how our documents describe the system: teacher routing first, then loss/backprop, then the controller reads optimizer-memory signals to decide where the knowledge should be written.


Compact algorithm form Algorithm 1: Prototype-based distillation block Input: training sample 𝑥𝑡, optional label 𝑦𝑡, teacher set 𝒯, teacher prototypes {𝑐𝑘}𝑘=1𝐾 , student parameters 𝜃𝑡

- 1. Compute routing embedding: ℎ𝑡 = 𝑓emb(𝑥𝑡)
- 2. Score each teacher: 𝑠𝑘(𝑥𝑡) = cos⁡(ℎ𝑡, 𝑐𝑘)
- 3. Select teacher:

𝑘𝑡∗ = arg⁡ max⁡

𝑘

𝑠𝑘(𝑥𝑡)

- 4. Obtain selected teacher distribution:

𝑝𝑇

𝑡

∗

𝜏 (⋅∣ 𝑥𝑡)

- 5. Obtain student distribution: 𝑝𝜃𝜏(⋅∣ 𝑥𝑡)
- 6. Compute KD loss:

ℒKD(𝑡) = 𝜏2KL ⁣(𝑝𝑇

𝑡

∗

𝜏 ∥ 𝑝𝜃𝜏)

- 7. If labels exist, compute:

ℒCE(𝑡) = −log⁡𝑝𝜃(𝑦𝑡 ∣ 𝑥𝑡)

- 8. Form total loss:


ℒ(𝑡)(𝜃) = 𝜆KDℒKD(𝑡) + 𝜆CEℒCE(𝑡) + 𝜆regℒreg(𝑡)

- 9. Backpropagate: 𝑔𝑡 = ∇𝜃ℒ(𝑡)(𝜃𝑡)
- 10.Stop here for the distillation block. From this point, the Theory 101 block starts.


