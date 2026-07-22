000

001

002

## TRIHOPE: ROUTING DISTILLATION UPDATES ACROSS RETRIEVAL, FAST, AND PERMANENT MEMORY WITH OPTIMIZER-STATE SIGNALS

Paper under double-blind review

003

004

005

006

007

008

## Anonymous authors

009

010

011

012

## ABSTRACT

Distillation answers which teacher a student should follow. In a stream, that is only half of the learning decision: the update produced by a good teacher may still be too rare, too unstable, or too local to place immediately in shared weights. We present TriHOPE, an online controller that routes each distillation update to one of three memory substrates. An isolated observation can remain in an external retrieval store (R). A recurring but unsettled pattern can be written to sparse LoRA fast weights (F). A repeated, stable adaptation can be merged into permanent base weights (P). The controller works at Transformer-block granularity and reads only pre-update Adam state, using the current gradient together with first- and second-moment history to estimate surprise, directional agreement, volatility, and recurrence. Its masked AdamW implementation closes coordinates exactly: a closed coordinate receives no gradient update, moment update, weight decay, or bias-correction aging. On a five-seed controlled continual-distillation stream, TriHOPE sends every isolated novel batch to retrieval, routes 74.7% of recurrent new-domain actions to fast weights, and opens 11.9% of indexed coordinates per step. Old-domain forgetting falls to 24.2 ± 9.2 points, compared with 100.0 for full fine-tuning and 51.9 ± 7.5 for LoRA, although new-domain accuracy is lower. Removing retrieval reverses much of that trade-off: new-domain accuracy rises, while forgetting increases to 77.9 ± 7.6 points. A separate consolidation test preserves logits within 1.2 × 10−7. The evidence is intentionally narrow–it validates the routing mechanism on a small synthetic stream rather than making a large-model performance claim.

013

014

015

016

017

018

019

020

021

022

023

024

025

026

027

028

029

030

031

032

033

034

035

036

## 1 INTRODUCTION

A student learning from several teachers faces two different choices. The familiar one is about supervision: which teacher is best suited to the current sample? The less studied choice comes one step later. Once that teacher has produced a loss and a gradient, where should the new information live?

The distinction is easy to miss in ordinary batch training, where every useful gradient is expected to become part of one parameter vector. It becomes important in a stream. A large update may reflect a durable new rule, but it may also come from a one-off fact, an outlier, or a local correction that will not recur. Writing all of these cases into shared weights invites interference, a long-standing problem in continual learning (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017; De Lange et al., 2022). Restricting adaptation to LoRA narrows the writable surface (Hu et al., 2022), yet a single adapter still has to hold information with very different lifetimes. Retrieval avoids immediate parameter change (Lewis et al., 2020; Gutiérrez et al., 2025), but retrieval alone does not explain when a pattern has become common enough to deserve parametric learning. [URL 🔗](#page-0)

Multi-teacher distillation has made the supervision choice increasingly adaptive. Recent methods learn teacher weights, exploit specialized experts, or route teacher signals through LoRA modules (Yang et al., 2025; Kim et al., 2025; Feng et al., 2025). Those methods decide who teaches. They [URL 🔗](#page-0)

037

038

039

040

041

042

043

044

045

046

047

048

049

050

051

052

053


054

generally leave the destination of the resulting update to the optimizer. Our starting question is therefore simple: can the training history already carried by the optimizer help decide whether an update should remain external, enter fast weights, or become permanent?

055

056

057

TriHOPE uses Adam’s state as that history. The first moment records whether recent gradients have pointed in a persistent direction; the second moment records their coordinate-wise scale (Kingma & Ba, 2015). Compared with the current gradient, these quantities provide an inexpensive view of novelty and consistency. The controller first identifies the attention or feed-forward blocks under the greatest learning pressure. It then assigns each selected block to retrieval (R), sparse LoRA (F), or post-update LoRA-to-base consolidation (P). Teacher routing and write routing remain separate throughout: the former chooses the source of the target distribution, while the latter chooses the storage medium for the induced update. [URL 🔗](#page-0)

058

059

060

061

062

063

064

065

We make the idea concrete with block-level signals from pre-update Adam state, a recurrence estimate, and coordinate-wise write masks. Those masks govern gradients, Adam moments, decoupled weight decay, and per-coordinate bias-correction counters. The implementation also includes a merge path whose functional invariance is proved under the conditions used in the experiments. The controlled study is set up to expose the method’s weakness alongside its benefit: retrieval protects earlier behavior, but delays learning of the new domain.

066

067

068

069

070

071

072

073

## 2 RELATED WORK

Knowledge distillation began as a way to transfer a teacher’s predictive behavior into a smaller student (Bucila et al., 2006; Hinton et al., 2015). Later work broadened the transferred signal to intermediate features, relations, and ensembles (Gou et al., 2021). With several teachers, simple averaging is often a poor fit because expertise is uneven across samples and domains. Adaptive multi-teacher methods therefore weight teachers from their performance or their distance from the student (Yang et al., 2025), recover signal from active and inactive mixture-of-experts components (Kim et al., 2025), or place heterogeneous teacher knowledge in routed LoRA experts (Feng et al., 2025). These methods make the source of supervision selective. After the loss is formed, however, the gradient usually enters the model through an ordinary update rule. [URL 🔗](#page-0)

Continual learning shows why that last step deserves its own decision. Sequential updates can erase previously useful behavior, which has led to regularization, replay, exemplar memory, architectural separation, and parameter isolation (Kirkpatrick et al., 2017; Rebuffi et al., 2017; De Lange et al., 2022). The common concern is not simply whether the current target is correct. It is whether the evidence is strong and persistent enough to justify changing durable parameters. This question is especially sharp in multi-teacher training: a specialist can be correct on the present sample while the corresponding update remains too local to share with the rest of the model. [URL 🔗](#page-0)

The available memory mechanisms cover different parts of this problem. Retrieval-augmented systems keep information outside the weights (Lewis et al., 2020), and HippoRAG 2 frames structured retrieval as non-parametric continual learning (Gutiérrez et al., 2025). LoRA provides a small, reversible parametric workspace (Hu et al., 2022). Other recent approaches protect important parameters (Ling et al., 2025), allocate adapter rank over time (Bhat et al., 2025), update sparse memory slots (Lin et al., 2025), or merge adapters as learning proceeds (Qiao & Mahdavi, 2025). Taken together, these works suggest a useful division of labor: uncertain observations can remain external, recurring patterns can occupy adapters, and stable adaptations can eventually be folded into the base model. They do not, by themselves, give a single online rule for moving between those destinations. [URL 🔗](#page-0)

Optimizer state offers a candidate rule because it is already a summary of recent learning dynamics.

Adam maintains exponentially weighted first and second moments (Kingma & Ba, 2015); [URL 🔗](#page-0)

separates adaptive gradient steps from weight decay (Loshchilov & Hutter, 2019); and AdaBelief changes the step size according to disagreement between a gradient and its expected direction (Zhuang et al., 2020). Optimizer choice is also known to affect representation dynamics (Sharon & Dar, 2024). A related line of work treats learned updates or test-time optimization as a memory process (Sun et al., 2024; Behrouz et al., 2025b;a). Our use of optimizer state is narrower. We do not turn the optimizer into a sequence model. We use its pre-update moments as local evidence about whether the current gradient is unusual, aligned with recent gradients, and part of a recurring input pattern. [URL 🔗](#page-0)

074

075

076

077

078

079

080

081

082

083

084

085

086

087

088

089

090

091

092

093

094

095

096

097

098

099

100

101

AdamW

102

103

104

105

106

107


The gap lies at the intersection of these lines of work. Teacher-routing methods decide whose target to trust. Continual-learning methods decide how aggressively to change the model. Retrieval and adapter methods provide memories with different persistence. TriHOPE connects those pieces at the update level: it uses the gradient induced by the selected teacher to choose an architectural block, assign a storage destination, and make AdamW respect that assignment coordinate by coordinate.

## 3 LEARNING PROBLEM

Let T = {T1, . . . , TK} be a set of teachers and let Sθ be a causal Transformer student with L layers. Each layer contains an attention block and a feed-forward block, indexed by b ∈ {attn, ffn}. Base matrices constitute permanent parameters θP, while LoRA factors attached to the same projections constitute fast parameters θF. An external buffer serves as the retrieval store R.

Teacher selection happens before write routing. A shared embedding ht = femb(xt) is compared with prototypes ck built from seed examples for each teacher. The controlled study uses the hard assignment

A soft mixture would fit the same downstream controller, but it is not used in the reported experiment.

At temperature τ, the chosen teacher and the student define pτ T∗ and pτ θ . The training loss is

t

Write routing is based on gradients of this complete objective. It therefore responds to the learning signal that the optimizer would actually apply, rather than to an independent novelty score computed from the input alone.

## 4 TRIHOPE

## 4.1 LOCATING THE UPDATE

For parameter subset s ∈ {P, F} in layer ℓ and block b, define

The base and LoRA subsets belonging to the same architectural block are paired under a = (ℓ, b). Their combined pressure is

The controller keeps the M blocks with the largest pressure. This first decision answers where the sample is trying to change the network. The store decision is made only after that location has been identified.

## 4.2 READING THE OPTIMIZER’S HISTORY

Every signal is computed from state available before the current optimizer step. For parameter group j, Adam holds a first moment m(j) t−1 and a second moment v(j) t−1. Surprise is the present squared gradient measured against the scale accumulated in the second moment:

A high value means that the group is receiving an update larger than its own recent history would predict. Because the normalization is group-specific, the score is less sensitive to raw scale differences between layers.


- 162

Direction and magnitude carry different information. We measure directional agreement with

- 163

- 164

- 165

Let a(j) t and b(j) t be exponential moving averages of r(j) t and (r(j) t )2. The normalized volatility is

- 166

- 167

- 168

- 169

- 170

The cosine term asks whether the update points in a familiar direction. The volatility term asks whether its magnitude has been steady enough to trust.

- 171

- 172

Recurrence is estimated from three bounded sources:

- 173

- 174

- 175

176

- 177

- 178

- 179

Here h is a bucket identifier, nt(h) counts prior occurrences of the bucket, and Ht counts nearby routing embeddings. The three terms capture repeated gradient direction, repeated bucket identity with declining surprise, and repeated proximity in retrieval space. Signals from the paired base and LoRA groups are combined with gradient-norm weights before the policy is evaluated.

- 180

- 181

- 182

- 183

- 184

## 4.3 CHOOSING A STORE

For each selected architectural block, the policy is

The order matters. A pattern that has become both recurrent and stable is allowed to consolidate even when it is still producing a substantial gradient. A surprising pattern with little evidence of recurrence stays in retrieval. The middle region–evidence that repeats but has not settled–is assigned to fast LoRA weights.

- 185

- 186

- 187

- 188

- 189

- 190

- 191

- 192

- 193

- 194

- 195

- 196

## 4.4 SPARSE WRITES TO FAST MEMORY

For a LoRA update ∆W = BA of rank q, rank component k receives the score

Only the K highest-scoring components are opened. The resulting gradient can be written as

These masks authorize coordinates rather than merely zeroing the visible gradient. The optimizer also leaves the corresponding first moment, second moment, optional AMSGrad maximum, decoupled weight decay, and coordinate-wise update counter untouched. A coordinate that is reopened later therefore resumes from the number of updates it actually received.

- 197

- 198

- 199

- 200

- 201

- 202

- 203

- 204

- 205

- 206

- 207

## 4.5 CONSOLIDATING FAST MEMORY

A P action applies the current fast update and then folds the accumulated LoRA delta into the base matrix:

The adapter’s optimizer state is reset because the factors have been reinitialized. With LoRA dropout disabled–as in the controlled study–or in evaluation mode, the pre-merge and post-merge linear maps are equal in exact arithmetic. The appendix states this condition explicitly and proves the network-level result.

- 208

- 209

- 210

- 211

- 212

- 213

- 214

- 215


*Figure 1: Teacher selection determines the supervision signal. Write routing begins after backprop- agation: pre-update optimizer state identifies affected blocks and assigns their updates to retrieval, sparse LoRA, or consolidation.*

## 4.6 FORMAL GUARANTEES FOR ROUTED WRITES AND CONSOLIDATION

The write policy is meaningful only if the optimizer respects a closed coordinate and if consolidation does not alter the model at the instant of promotion. The following statements formalize those two requirements. The appendix gives the complete recurrences, assumptions, and proofs, including the all-open reduction to ordinary AdamW and the correspondence between store actions and writable coordinates.

Theorem 1 (Exact state isolation under MaskedAdamW). Let at,i ∈ {0, 1} be the authorization mask for coordinate i at optimizer call t. Ifat,i = 0, then the parameter, first moment, second moment, optional AMSGrad maximum, and coordinate-local update counter are all unchanged. Decoupled weight decay is also omitted. Ifat,i = 1, the coordinate follows the ordinary AdamW recurrence with bias correction based on its own number ofaccepted updates.

Corollary 1 (Coordinate-local bias correction). If coordinate i has received r accepted updates before it is reopened, its next update uses the corrections 1 − βr+1 1 and 1 − βr+1 2 , regardless ofhow many optimizer calls occurred while the coordinate was closed.

Theorem 2 (Function-preserving LoRA consolidation).

- s = α/q and with the LoRA dropout acting as the identity. Replacing W by W+ = W + sBA, resetting B to zero, and reinitializing A preserves the layer output for every input in exact arithmetic. Replacing any finite collection ofsuch layers therefore preserves the composed network function. The implementation additionally resets the adapter’s optimizer state after the merge.

Proposition 1 (Exact Top-K adapter budget). For A ∈ Rq×din and B ∈ Rdout×q, opening K rank components opens exactly K(din + dout) ofthe q(din + dout) LoRA-factor coordinates. Thus the writable fraction inside the selected adapter is exactly K/q.

Consider a LoRA linear map with scale


```
Input: xt, yt, teachers T, student Sθ, AdamW state, stores R, F, P
```

```
k∗ t ← route femb(xt) to a teacher prototype;
Compute Lt by Equation (2); backpropagate once;
```

```
foreach indexed base/LoRA group j do
```

```
read g(j) t ,m(j) t−1, v(j) t−1;
```

```
compute S(j) t , C(j) t , V(j) t , R(j) t ;
```

```
end
```

```
Aggregate paired groups and retain the Top-M architecture blocks;
```

```
foreach selected block a do
```

```
za ← π(Sa, Ra, Ca, Va);
```

```
if za = R then
append embedding and metadata; close the block’s weights
```

```
else
```

```
if za = F then
```

```
open the Top-K LoRA ranks
```

```
else
open LoRA; queue a post-step merge
```

```
end
```

```
end
```

```
end
```

```
Apply exact masked AdamW; complete queued merges and reset adapter state;
```

```
Algorithm 1: One TriHOPE training step
```

## 5 CONTROLLED STUDY

## 5.1 STREAM CONSTRUCTION

The experiment is deliberately small. Its purpose is to ask whether the controller distinguishes isolated events from recurring patterns, not whether a tiny Transformer is competitive on language modeling. The vocabulary contains 48 tokens. The student has two causal Transformer layers, width 32, four attention heads, and rank-4 LoRA on Q, K, V, O, and the three feed-forward projections. Domain tokens identify two deterministic sequence rules. In domain A, each content token advances by one modulo the content vocabulary; in domain B, it advances by seven. A separate perfect teacher emits a near-one-hot next-token distribution for each rule.

The base model is pretrained for 120 domain-A steps with LoRA frozen. The adaptation stream then presents 20 recurrent A steps, 10 independently generated novel batches with unique bucket identifiers, 70 recurrent B steps, and a 40-step alternating A/B tail. Each batch contains eight sequences of length ten. Because the first content token is sampled randomly, its transition is excluded from the loss. Every reported result uses five random seeds.

## 5.2 COMPARISONS AND MEASUREMENTS

Full fine-tuning updates the entire model. The LoRA baseline freezes base and shared parameters and updates every adapter. The no-retrieval ablation removes the R branch, while the no-consolidation ablation prevents P actions. The full controller retains the four highest-pressure architecture blocks and, in this study, opens all rank components within an authorized adapter. AdamW uses a learning rate of 10−3 and weight decay of 10−2.

We report fixed-set next-token accuracy on domains A and B, their final mean, and A forgetting, defined as A accuracy after the A warm phase minus A accuracy after the B phase. We also record the fraction of controller-indexed coordinates opened at each step, action shares by stream phase, and wall-clock time. Shared embeddings and normalization parameters are outside the controller index and follow their configured training rule. The released package contains the stream generator, all five seed traces, result tables, plotting code, and the test suite used for the optimizer and consolidation checks.


*Table 1: Controlled stream results over five seeds (mean ± standard deviation, percentage points). Lower forgetting and active fraction are better; higher accuracies are better.*

| Method | Final A Final B Mean Forgetting ↓ | Active ↓ |
| --- | --- | --- |
| Full FT | 34.9 ± 8.0 100.0 ± 0.0 67.4 ± 4.0 100.0 ± 0.0 100.0 ± 0.0 |   |
| LoRA |   | 70.2 ± 10.4 42.3 ± 6.5 56.3 ± 2.1 51.9 ± 7.5 16.4 ± 0.0 |
| TriHOPE |   | 86.2 ± 13.1 36.5 ± 9.6 61.3 ± 3.7 24.2 ± 9.2 11.9 ± 0.4 |
| No retrieval |   | 52.5 ± 12.0 87.0 ± 7.5 69.8 ± 5.7 77.9 ± 7.6 17.5 ± 0.0 |
| No consolidation 85.5 ± 14.0 39.5 ± 11.1 62.5 ± 3.6 25.4 ± 8.5 11.9 ± 0.5 |   |   |

*Figure 2: Final old-domain accuracy, new-domain accuracy, and forgetting. The retrieval branch gives the strongest retention and the weakest immediate adaptation. Error bars show one standard deviation over five seeds.*

## 6 RESULTS

The strongest effect is on retention. Full fine-tuning learns domain B perfectly but erases the A rule during the B phase, giving 100.0 points of forgetting. LoRA reduces forgetting to 51.9 ± 7.5. TriHOPE reduces it further to 24.2±9.2 while opening 11.9±0.4% of indexed coordinates. Its final A accuracy is 86.2 ± 13.1%, compared with 34.9 ± 8.0% for full fine-tuning and 70.2 ± 10.4% for LoRA.

That retention is purchased with slower adaptation. TriHOPE reaches 36.5 ± 9.6% on B, whereas full fine-tuning reaches 100% and the no-retrieval ablation reaches 87.0±7.5%. Mean final accuracy is 61.3 ± 3.7%: higher than LoRA’s 56.3 ± 2.1%, but lower than full fine-tuning’s 67.4 ± 4.0% and no retrieval’s 69.8 ± 5.7%. The controller therefore moves the stability–plasticity operating point rather than dominating every baseline.

The action trace explains that behavior. Every isolated novel batch is assigned to R. Repetition changes the decision: during recurrent B, 74.7 ± 3.9% of actions go to F, and that share rises to 93.2±4.7% in the mixed tail. P actions remain almost absent because the short stream rarely satisfies the conservative repetition and stability thresholds. The main retention result should therefore be attributed to retrieval and sparse fast writes, not to consolidation.

The no-retrieval ablation isolates the cost of that conservative choice. Relative to TriHOPE, it improves B accuracy by 50.5 points but increases forgetting by 53.7 points. Removing consolidation has little effect because the full policy almost never selects P in 140 adaptation steps. That null result is useful: the benchmark establishes the mechanics of a safe merge, but it does not establish a downstream accuracy benefit from deciding when to merge.

We test the merge path separately. An attention LoRA is trained, one queued P action is applied, and logits are compared immediately before and after consolidation. Across five seeds, the largest


*Table 2: TriHOPE action shares by stream phase. Active is the mean percentage of indexed coordi- nates opened.*

| Phase | R share | F share | P share Active |
| --- | --- | --- | --- |
| Recurrent A 73.2 ± 12.9 26.7 ± 12.9 0.0 ± 0.0 4.6 |   |   |   |
| Isolated novel 100.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 |   |   |   |
| Recurrent B 25.1 ± 3.9 74.7 ± 3.9 0.1 ± 0.2 13.1 |   |   |   |
| Mixed A/B | 6.7 ± 4.7 93.2 ± 4.7 0.0 ± 0.0 16.4 |   |   |

Recurrent B

Recurrent A

Isolated novel

*Figure 3: Action composition over the stream. Isolated observations remain external; recurrent observations increasingly enter fast weights. The chosen horizon produces almost no permanent writes.*

absolute discrepancy is 1.20×10−7, and the reset LoRA moment norms are zero. This is a numerical check of Equation (15); it is not evidence that consolidation improves generalization. [URL 🔗](#page-0)

The present implementation also has a modest runtime cost. On the same CPU, one adaptation run takes 1.20 ± 0.12 seconds for TriHOPE and 0.94 ± 0.04 seconds for full fine-tuning. Fewer writable coordinates do not yet translate into higher throughput because signal aggregation and masking are implemented in Python. A fused kernel could change this result, but the current code should be treated as a correctness-oriented reference implementation.

## 7 DISCUSSION

The experiment answers the narrow question for which it was built. Pre-update optimizer state is sufficient to distinguish the unique-bucket phase from the recurrent phases, and exact masking turns that decision into an actual optimizer boundary. The resulting model retains more of domain A because many early or uncertain updates never touch shared parameters. The same mechanism explains its weakness on B: retrieval protects information from parameter interference, but the released inference path does not read that store, and the training loop does not replay retrieved items into F.

This observation points to the most important next step. A practical version should connect the three stores rather than treating retrieval as a terminal action. Repeated retrieval hits could trigger replay into fast weights, and a validation signal could govern later promotion into permanent weights. Such a path would preserve the current safety bias while giving stored observations a route back into prediction. Learned thresholds, hysteresis, or an explicit write budget are plausible ways to calibrate the transition.

Consolidation remains the least tested part of the controller. The algebra and implementation show that a merge can be performed without changing the deterministic function at the merge point. They do not show that the policy has chosen the right time to make knowledge permanent. Longer streams with returning tasks are needed to create enough repeated, stable evidence for that question. Routing


granularity will matter as well: finer masks may improve plasticity at the cost of noisier signals, while layer-level decisions would be cheaper but less selective.

## 8 LIMITATIONS

The study uses deterministic teachers, a two-rule synthetic stream, and a small student. It does not test natural language, teacher disagreement, imperfect labels, heterogeneous tokenizers, soft teacher mixtures, or billion-parameter memory costs. The thresholds are fixed rather than tuned on held-out streams, and the observed standard deviations are sizable for some metrics. Retrieval stores embeddings and metadata but is not consulted during inference. Shared embeddings and normalization parameters are not controller-indexed. The signals inherit Adam’s coordinate scaling, and the reference implementation is not fused.

The method also creates a security concern that is not visible in the synthetic benchmark. A continually updated system can be exposed to erroneous or malicious observations. External storage, fast adaptation, and permanent consolidation should therefore have different audit and rollback requirements. In particular, permanent writes should require stronger evidence than fast writes, retrieval contents should be inspectable, and a deployment should retain checkpoints that allow a promoted update to be reversed.

The official ICLR 2027 author kit was not public when this version was built. The package uses the latest public ICLR 2026 style as a formatting baseline and must be migrated when the 2027 files are released.

## 9 CONCLUSION

Distillation usually ends its routing decision when a teacher has been selected. TriHOPE extends the decision to the update itself. The current gradient and pre-update Adam state identify where learning pressure is concentrated and whether it looks isolated, recurrent, or stable. Retrieval, LoRA, and base weights then serve as memories with increasing commitment, while exact masks keep the optimizer from crossing the chosen boundary.

The controlled stream shows both sides of this design. It preserves old-domain behavior far better than full fine-tuning or ordinary LoRA, but it learns the new domain more slowly because retrieved observations are not yet brought back into inference or replay. That trade-off is the main empirical result. The next evaluation should therefore combine retrieval-aware prediction, replay-based promo- tion, and longer natural-language streams in which permanent consolidation occurs often enough to be judged by task performance rather than by invariance alone.

## REFERENCES

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. Nested learning: The illusion of deep learning architectures. arXiv preprint arXiv:2512.24695, 2025a.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2025b.

Prashant Shivaram Bhat, Shakib Yazdani, Elahe Arani, and Bahram Zonooz. Parameter efficient continual learning with dynamic low-rank adaptation. arXiv preprint arXiv:2505.11998, 2025.

Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings ofthe 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 535–541, 2006.

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3366–3385, 2022.

Kaidong Feng, Zhu Sun, Hui Fang, Jie Yang, Wenyuan Liu, and Yew-Soon Ong. Routing distilled knowledge via mixture of LoRA experts for large language model based bundle generation. arXiv preprint arXiv:2508.17250, 2025.


- 486 487 488 Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal ofComputer Vision, 129:1789–1819, 2021.

- 489 Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory: 490 491 Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.

- 492 Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

- 493 494 495 496 497 Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

- 498 Gyeongman Kim, Gyouk Chu, and Eunho Yang. Every expert matters: Towards effective knowledge distillation for mixture-of-experts language models. arXiv preprint arXiv:2502.12947, 2025.

- 499 500 501 502 Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

- 503 James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. 504 505 506 Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe National Academy ofSciences, 114 (13):3521–3526, 2017.

- 507 Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K"uttler, Mike Lewis, Wen-tau Yih, Tim Rockt"aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, 2020.

- 508 509 510 511 512 513 514 Jessy Lin, Luke Zettlemoyer, Gargi Ghosh, Wen-Tau Yih, Aram Markosyan, Vincent-Pierre Berges, and Barlas O˘guz. Continual learning via sparse memory finetuning. arXiv preprint arXiv:2510.15103, 2025.

- 515 516 Shimou Ling, Liang Zhang, Jiangwei Zhao, Lili Pan, and Hongliang Li. LoRA-based continual learning with constraints on critical parameter changes. arXiv preprint arXiv:2504.13407, 2025.

- 517 Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Confer- ence on Learning Representations, 2019.

- 518 519 520 521 Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology ofLearning and Motivation, 24:109–165, 1989.

- 522 Fuli Qiao and Mehrdad Mahdavi. Merge before forget: A single LoRA continual learning via continual merging. arXiv preprint arXiv:2512.23017, 2025.

- 523 524 525 526 527 Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. iCaRL: Incremental classifier and representation learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.

- 528 Yuval Sharon and Yehuda Dar. How do the architecture and optimizer affect representation learn- 529 530 ing? on the training dynamics of representations in deep neural networks. arXiv preprint arXiv:2405.17377, 2024.

- 531 Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): RNNs with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

- 532 533 534 535 536 537 Chuanguang Yang, Xinqiang Yu, Han Yang, Zhulin An, Chengqing Yu, Libo Huang, and Yongjun Xu. Multi-teacher knowledge distillation with reinforcement learning for visual recognition. arXiv preprint arXiv:2502.18510, 2025.

- 538 Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar Tatikonda, Nicha Dvornek, Xenophon Pa- 539 pademetris, and James S. Duncan. AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients. In Advances in Neural Information Processing Systems, 2020.


- 540

## APPENDIX: DETAILED PROOFS FOR ROUTED OPTIMIZATION

The claims below match the released execution path. Controller-indexed optimizer state is initialized before routed training. Gradients are dense and real-valued; one Boolean mask is fixed for each optimizer call; and the non-fused, non-foreach, non-capturable, non-differentiable AdamW path is used. Unsupported modes are rejected by the constructor. The experiments minimize the loss, although the same statements hold in maximize mode after replacing gt,i by −gt,i. For consolidation, LoRA dropout is zero or the module is in evaluation mode, so the dropout map is the identity.

Let at,i ∈ {0, 1} indicate whether coordinate i is open at optimizer call t. Let mt,i and vt,i denote the first and second moments, nt,i the number of accepted updates, and gt,i the gradient used by the optimizer. For AMSGrad, let rt,i denote the running maximum of the second moment. The coordinate-wise state recurrence is

vt,i

When AMSGrad is enabled, the implementation also applies

For an open coordinate, define st,i = vt,i for AdamW and st,i = rt,i for AMSGrad, and set

- 541

- 542

- 543

- 544

- 545

- 546

- 547

- 548

- 549

- 550

- 551

- 552

= (1

− at,i)vt−1,i + at,i β2vt−1,i + (1 − β2)g2 t,i  .

- 553

- 554

- 555

- 556

- 557

- 558

- 559

- 560

## The parameter update is

- 561

- 562

- 563

- 564

sbt,i

ϵ > + ϵ 0.

- 565

Here ηt is the learning rate,

λ is decoupled weight decay, and

- 566

The scalar compatibility field

step

is

retained in the optimizer state, but neither the update nor its bias correction reads it; both use the coordinate tensor coord_step, represented by nt,i.

- 567

- 568

- 569

- ProofofTheorem 1. Fix an optimizer call t and a coordinate i with at,i = 0. Substitution into Equation (16) gives nt,i = nt−1,i. The same substitution into Equations (17) and (18) yields mt,i = mt−1,i and vt,i = vt−1,i. If AMSGrad is active, Equation (19) similarly gives rt,i = rt−1,i. Finally, the closed branch of Equation (21) gives θt,i = θt−1,i; in particular, the factor 1 − ηtλ is not applied, so decoupled weight decay cannot move the coordinate. [URL 🔗](#page-0)

- 570

- 571

- 572

- 573

574 The equalities hold for one call without approximation. Applying the same argument inductively over any 575 interval of closed calls shows that the entire coordinate state remains equal to its value immediately before 576 the interval. Equivalently, if Ot = {i : at,i = 1}, then the supports of θt − θt−1, mt − mt−1, vt − vt−1,

nt − nt−1, and, when present, rt − rt−1 are all subsets of Ot.

- 577

- 578 For at,i = 1, Equations (17) and (18) are the usual Adam moment updates, Equation (20) applies the standard 579 bias corrections, and the open branch of Equation (21) applies decoupled weight decay followed by the adaptive [URL 🔗](#page-0)

step. The only difference from tensor-level AdamW is that the exponent is the coordinate’s accepted-update count rather than a shared tensor clock. This proves both exact isolation of closed coordinates and ordinary AdamW behavior on each open coordinate. [URL 🔗](#page-0)

- 580 581 582

- 583

ProofofCorollary 1. Unrolling Equation (16) from the zero initialization gives [URL 🔗](#page-0)

- 584

- 585

- 586

- 587

Suppose coordinate i has been accepted exactly r times before call t and is open at call t. Then nt−1,i = r and [URL 🔗](#page-0)

- 588

at,i = 1, so nt,i = r + 1. Substituting this value into Equation (20) gives the factors 1 − βr+1 1 and 1 − βr+1 2 . [URL 🔗](#page-0)

- 589

Calls with aτ,i = 0 contribute zero to Equation (22) and, by Theorem 1, do not alter the moments. Therefore [URL 🔗](#page-0)

- 590

neither the bias-correction clock nor the state being corrected ages while the coordinate is closed. [URL 🔗](#page-0)

- 591

- 592 Lemma 1 (All-open reduction to ordinary AdamW). If at,i = 1 for every coordinate and every call, all 593 coordinate counters are initialized at zero, and the same hyperparameters are used, Equations (16) to (19) and (21) reduce coordinate by coordinate to ordinary AdamW, including its AMSGrad variant. [URL 🔗](#page-0)


- 594

Proof. Under the all-open mask, Equation (22) gives nt,i = t for every i. Equations (17) and (18) become the standard first- and second-moment recurrences, and Equation (19) becomes the standard running maximum when AMSGrad is enabled. Equation (20) therefore uses the ordinary factors 1 − βt 1 and 1 − βt 2. The open branch of Equation (21) is exactly AdamW’s decoupled shrinkage and bias-corrected adaptive update. Equality holds for every coordinate, hence for the full parameter tensors. The repository regression suite also compares this path numerically with torch.optim.AdamW over multiple steps. [URL 🔗](#page-0)

- 595

- 596

- 597

- 598

- 599

- 600

The next result formalizes the post-step transition from fast to permanent memory. Let A ∈ Rq×din , B ∈ Rdout×q, W ∈ Rdout×din , bias b, and scale s = α/q. When the LoRA dropout acts as the identity, the layer applied to a row-vector input x is

- 601

- 602

- 603

- 604

- 605

ProofofTheorem 2. Let the merge use the adapter values after the authorized optimizer step, and define [URL 🔗](#page-0)

- 606

- 607

The implementation then resets the adapter by drawing some new A+ and setting B+ = 0. For every input x,

- 608

- 609

- 610

- 611

- 612

- 613

- 614

The second equality uses B+ = 0, and the third uses (BA)⊤ = A⊤B⊤. Thus the pre-merge and post-merge layers are pointwise identical in exact arithmetic, with the bias unchanged.

- 615

- 616

Now replace any finite collection of LoRA layers in a network. Order the affected modules along the forward computation. The first replacement leaves its output unchanged for every possible input, so every downstream module receives exactly the same value. Repeating this argument for each replacement proves by induction that the final network output is unchanged. Active training-time dropout is excluded because the random map applied to x need not be representable by one fixed matrix W+; zero dropout or evaluation mode makes the identity in Equation (23) valid. [URL 🔗](#page-0)

- 617

- 618

- 619

- 620

- 621

- 622

Corollary 2 (Fresh fast-memory state after consolidation). After a completed merge, the fast branch contributes zero and its MaskedAdamW moments, AMSGrad maximum, scalar compatibility step, and coordinate counters are all zero. Consequently, the next accepted adapter update starts from a fresh optimizer state and uses first-update bias correction.

- 623

- 624

- 625

- 626

Proof. The adapter reset sets B+ = 0, so s x(A+)⊤(B+)⊤ = 0 for every input. The writer then calls reset_state_for_params on both LoRA factors. That routine zeros exp_avg, exp_avg_sq, the op- tional max_exp_avg_sq, coord_step, and the compatibility field step. Hence no pre-merge momentum is carried into the reinitialized adapter. By Equation (16), its next open call changes each authorized counter from zero to one, producing first-update bias correction. [URL 🔗](#page-0)

- 627

- 628

- 629

- 630

- 631

ProofofProposition 1. For component k ∈ {1, . . . , q}, define the factor-coordinate sets [URL 🔗](#page-0)

- 632

- 633

- 634

The component contains |IA k | + |IB k | = din + dout coordinates. Sets associated with different components are disjoint because they use different rows of A and different columns of B. If K is the set of selected components and |K| = K, then

- 635

- 636

- 637

- 638

- 639

The two factor tensors contain qdin + doutq = q(din + dout) coordinates in total. Dividing gives the exact fraction K/q. The implementation clips a requested value to K⋆ = min{K, q}, so the same count holds with K⋆ if a request exceeds the rank.

- 640

- 641

- 642

- 643

Implementation correspondence. Before actions are processed, the writer assigns a zero mask to every controller-

- 644

indexed parameter. An R

action leaves those masks closed. An

F

action opens only the selected LoRA rows

- 645

of A and columns of B. A queued P action opens the corresponding adapter, performs the accepted step, and consolidates afterward. Therefore Theorem 1 and Proposition 1 confine optimizer changes to the declared write set, while Theorem 2 preserves the deterministic function during promotion. Shared or otherwise unindexed parameters follow their separately configured training rule and are outside this claim. [URL 🔗](#page-0)

- 646

- 647
