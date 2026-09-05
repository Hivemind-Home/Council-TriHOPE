# Related work (draft)

**Multi-teacher distillation decides who teaches.** Distillation transfers
a teacher's behaviour into a student (Hinton et al., 2015), and with
several teachers the supervision itself becomes selective — weighting by
performance or distance (Yang et al., 2025), mixture-of-experts recovery
(Kim et al., 2025), routed LoRA experts (Feng et al., 2025). All decide
*whose* target the student follows; the gradient then enters the weights
through an ordinary optimizer step. That the teacher may be untrusted is a recognised threat: Hong et al.
(2023) show a backdoored teacher's behaviour transfers through data-free
distillation at over 90 % attack success and suppress the transfer at
training time. We do not attempt prevention; we make every permanent write
attributable to the teacher that caused it and revertible afterwards.

**Continual learning decides how much to change.** Sequential updates
erase earlier behaviour (McCloskey & Cohen, 1989; Kirkpatrick et al.,
2017), and every remedy asks the same question: is the evidence strong
enough to change durable parameters? Attribution-guided fine-tuning (Liu et al., 2026) gates each
weight's gradient by an LRP *importance* score computed offline per task;
Gradient Routing (Cloud et al., 2024) confines a labelled data source to a
chosen subregion with a user-supplied backward mask and ablates it to
unlearn. Both need the label at ingestion time. Our routing never reads
the label; provenance is recorded after the write, so a bad source can be
excised without having been anticipated.

**Memory substrates differ in persistence.** Retrieval keeps information
external: HippoRAG 2 (Gutiérrez et al., 2025) frames it as non-parametric
continual learning with the model as a fixed reader, and ReGrad (Su et
al., 2026) goes further by applying retrieved document gradients as
temporary, reverted LoRA deltas at inference — neither states when
external knowledge should become parametric. LoRA gives a small reversible
workspace, and recent methods manage it over time: Online-LoRA (Wei et al.,
2025) merges the current adapter into the base when a loss peak is
followed by a plateau; SLAO (Qiao & Mahdavi, 2025) merges each task's
adapter into a single running LoRA with a 1/√i schedule at task boundaries;
sparse memory finetuning (Lin et al., 2025) writes only the memory slots a
batch activates more than pretraining did; STABLE (Hoy & Celik, 2025)
accepts, rescales or rejects each LoRA→base merge against a forgetting
budget measured on previously edited anchors. Each is two-store and decides *where* an update lands once, from a global
loss, a task boundary, a usage count, or an extrinsic probe.

**Optimizer state as evidence.** Several lines read Adam's moments (Kingma & Ba, 2015). Nested Learning
(Behrouz et al., 2025) recasts momentum and Adam as associative memories
over gradients and organises memory into levels of fixed update frequency;
Titans (Behrouz et al., 2024) defines surprise as the gradient of a memory
loss and uses it, with momentum and a decay gate, to scale writes into a
learned memory module. MoLF (Tang et al., 2026) is closest to our
mechanism: it routes each module's update to a dense or LoRA expert by an
expected-preconditioned-descent score on Adam's moments under a Top-1
masked AdamW, then fuses once after training; its universal momentum
tracking keeps the losing expert's moments advancing, which we adopt for
the controller's evidence only — the optimizer's own state stays exactly
masked. Hu et al. (2026) show that
gradient attenuation fed to both moments inflates the effective step by
1/(1−α) through the second-moment denominator, which motivates our exact
mask: a closed coordinate receives no parameter, moment, weight-decay or
bias-correction change. Against all of these the gap is the same: every
rival is two-store and decides where once; we are three-store, decide
*when to promote* — R defers, F holds tentatively, P consolidates on
sustained stability and recurrence — enforce it exactly at the optimizer,
and can attribute and revert permanent writes.
