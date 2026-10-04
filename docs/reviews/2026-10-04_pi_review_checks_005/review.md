# PI review (with advisor) of the no-GPU checks C1–C3, design 005 v1 and errata 01 draft — as delivered, 2026-10-04

Two parts: the summary review, and the pasted change list. The PI's instruction: "accept this … review only if it makes sense".

---

## Bottom line

- **The three checks were well run.** The agent pre-registered them and honestly corrected me twice. I'd accept C1–C3, with one sharper reading of each.
- **The 005 draft has one real flaw: where the text readers read.** As written, P11a is close to guaranteed, so it has to change before the smoke run. Four smaller fixes come with it.
- **Approve the errata** with two wording edits.

## Where the agent corrected me

**Viswanath.** The post does report a reversal with rank. At r = 200 the model named the non-J part about half the time and the J part never; by r = 2000 it was the other way round. I missed this because I read the post through a summarizing fetch. Any web claim that changes a decision should be checked against a quote; I'll do that from now on. Deleting the line is still right, and the lesson (the J boundary depends on how you construct it) is now stronger.

**C2.** My premise was wrong. In my note I called mass moving to the intermediates "the patch working". But the recipient's own intermediate z gains as much as the partner's z′, so this isn't transfer from the partner. The better description is that the patched output becomes a mixture of the two trajectories:

- the answer moves toward the partner's (a → a′);
- both intermediates rise, which fits Brew's known property that the intermediate is the model's runner-up answer;
- about 0.04 of probability (absolute) leaks to non-colour tokens.

## C1–C3: what they actually show

**C1 (chain re-score).** CW-7 survives at about a third of its original size. The larger finding is two other things:

- **The Oracle's chain result isn't Oracle-specific.** J-lens also scores 11/60 at L56. Since J-lens is a linear readout and can't re-solve, its 11/60 means the intermediate is linearly readable at L56 in some items. The Oracle's hits that J-lens doesn't share (7 of 11) are the ones relevant to reading vs re-solving. With one readout per cell, they're noise-level for now.
- **Chain cannot test the NLA at all.** At L42, its training layer, the content probably isn't there yet: J-lens finds 2/60, and 001 found zero whole-cell leverage at the final cell at L40–44. At L56 the NLA is off-layer. So the NLA's 1/60 is uninformative, not a failure.

Rewrite §14.7's CW-7 entry to say exactly this.

**C2 (damage at ":").** Accept "influence" at L42, described as the mixture above. Write the damage-rule rewrite now, labelled post-hoc and applying only to future data: for whole-cell partner patches, damage means non-colour mass rising above a preset threshold, and spreading across palette colours is reported separately. Don't spend GPU decomposing the other layers; no decision depends on it.

**C3 (removal energy).** The overlooked number is the size of the removals: both take out only 0.1–1% of the activation's energy. Removing that much kills J-lens hits (15 → 0), as it must, since J-lens reads exactly those directions. Prose readers keep 99% of the activation, including everything they would need to re-solve. So "survives J-removal" was always a weak test. It can't separate reading outside J-space from re-solving; only an eraser shown to remove the information (a probe dropping to chance) can. Put that sentence in A4.

## 005 design: what to change

**1. Read the text readers at the report position, not the injected position.** This is the flaw.

- C_q (position 110) is one of the injected positions. The injection is added to the residual stream at L32–38, and the stream carries it forward, so at L40–44 it is still sitting there almost intact.
- So the "model-carried" read at C_q is effectively direct access. A reader detecting the non-J part there mostly shows that the vector is linearly present, which the probe already tells you.
- The fair comparison with the model's report is C_o, the open quote. It isn't injected, so anything there arrived by attention, which is what the model itself has to do to report.

The fix, at the same cost:

- **Direct reads** (the clean reader-sensitivity test) become primary for **P17**.
- **Carried reads at C_o** become primary for **P11a and P11b**.
- Drop the carried read at C_q for the text readers. Keep it for the token readers and the probe.
- Relabel C_q reads at L ≥ 40 as "injected position", never "model-carried".

**2. Screen out concepts the readers say anyway.** Words like "information", "process", "summary", "method", "review", "time", "case", "line", "frame", "role" and "free" are routine NLA prose. Their base rate will swamp any detection. Before any GPU run, compute each eligible concept's naming rate in existing NLA and Oracle readouts where it wasn't the target (Cathryn's nla-rl readouts and 002's). Exclude concepts above a preset threshold (I'd use 5%). The clean reads in the run then confirm the rate in place.

**3. Fix power by changing the allocation, not by adding GPU.** The dev set only gates and calibrates, so it needs about 10 concepts: the 6 gate concepts plus gel and earth plus two more. Every other screened concept goes to test, likely 35–45 instead of 12.

Pay for it on the text-reader side:

- NLA at 4 samples per read (your floor). Between-concept variance dominates a cluster bootstrap, so more concepts beat more samples.
- Text-reader conditions cut to full, J part, non-J part and rand-dir (2, 3, 4, 7). Pure J-lens vector (1) is optional; rand-⊥ (9) adds little beyond rand-dir for text readers.

That keeps you near the current 6–8 h budget with roughly 3× the test concepts. GPU cost is trivial; time and power are what bind.

**4. Report J-lens's rank of c under the non-J part as a check on the split.** J atoms span the whole space, so u^⊥ is orthogonal to its 16 selected atoms, not to c's J content everywhere. If J-lens often ranks c in its top 10 under u^⊥, the split hasn't removed c's J content, and every "non-J" claim inherits that. The data already comes from the token readers; just add this interpretation rule.

**5. Small fixes:**

- **rand-J pool.** Draw its tokens from the 194 baseline words only. The current pool includes other core concepts, which contaminates their mismatched-concept false-positive rates.
- **Oracle "k = 4".** Confirm whether the Oracle's contract means 4 phrases per readout or 4 samples. The budget assumes samples.

**Keep as drafted:**

- the tautology statement;
- the G1 report gate and its calibration and stop rules;
- the clamp scored on teacher-forced rank (the paper's swap clamp was rank-based too);
- the probe with validity gate G4;
- the injection band L32–38 with the L24–38 fallback;
- the concept-vector read position;
- Holm correction across the P-family;
- test outcomes unread until the analysis code is committed.

**P17's premise:** accept it as "largely follows A.9.2". The argument rests on a frozen reconstructor and a whitened-FVE reward over a phrase dictionary, and the card confirms both. Whether the reward refits by NNLS doesn't change the logic, so there's no need to ask the checkpoint author.

## Errata

- **Accept as written:** E2–E6 and E8.
- **E1:** replace "mass moving to the intermediates is part of the transfer" with the mixture description, and mark the damage-rule rewrite as post-hoc and prospective.
- **E7:** add C3's point that both removals take 0.1–1% of the energy, so the survives-removal tag means little without validated erasure.

---

## Pasted change list (as delivered)

**Must change before the smoke run**

Read position (§6, §8). C_q is an injected position, so at L40–44 the residual stream still carries the injection almost intact. A reader there is effectively getting direct access. Change three things:
Text readers read carried at C_o, which is primary for P11a and P11b. Information reaches C_o only by attention, as it must for the model to report it.
Direct reads are primary for P17, since the vector is known to be present there.
Drop the carried C_q read for text readers; keep it for token readers and the probe. Relabel every C_q read at L ≥ 40 as "injected position".

"Doesn't report" and "doesn't change behaviour" must use the most sensitive model measure (§8). P11a currently qualifies a strength when the model's 8-sample naming rate is at the random-direction level. The model can raise the concept's probability a lot without naming it in 8 samples, so this makes P11a easy to "support".
P11a: a strength qualifies only if naming is at the random level and the concept's log-prob at C_o (R1′) is within the random-direction 95th percentile.
P11b: a strength qualifies only if both the neutral-prompt KL (B1) and the concept log-prob impulse (B2) are within that percentile.
The strict versions are primary; the current versions become secondary.

Choose the text readers' strengths after G1, not now (§4). With fixed strengths {1/16, 1/4, 1}, there may be no strength at which the model is silent but which the readers were also run at. Set the three strengths in the G1 amendment to bracket where the model starts responding to the non-J part on the dev concepts: below, at and above that onset. Freeze them before the run.

Screen out common words, and move concepts from dev to test (§2).
Exclude concepts whose naming rate in existing NLA and Oracle readouts, where they weren't the target, exceeds a threshold set in advance (I'd use 5%). No GPU needed. Words like "information", "process", "method", "summary", "review", "time", "case", "line", "frame", "role" and "free" will otherwise swamp detection.
Dev only gates and calibrates, so it needs about 10 concepts: the 6 gate concepts, gel, earth, and two more. Everything else that passes the screen goes to test: roughly 35–45 concepts instead of 12.
Pay for it on the text-reader side: 4 NLA samples, and conditions full, J part, non-J part and rand-dir (2, 3, 4, 7). The pure J-lens vector (1) is optional.

The clamp needs its own null (§4, M2). The clamp set T_c is the union of the support tokens across all band layers plus the spelling tokens. At any one layer, the other layers' atoms are not orthogonal to the injected non-J part, so the clamp partly deletes the injection rather than only blocking it from re-entering J-space.
Add one model-side condition: the non-J part plus a clamp on a random token set of the same size, at the non-J part's best strength, teacher-forced.
Interpret M2 only if the c-specific clamp lowers report more than the random clamp.
Report the share of injected norm each clamp removes.

Add a split-quality gate, G5 (§9). J-lens atoms span the whole space, so the non-J part is orthogonal to its 16 selected atoms, not to all of c's J content. Measure J_NP's rank of c under the non-J part, read directly at the injection layer. If c is in the top 10 for more than a preset share of concepts, flag the split and qualify every "non-J" claim. The token-reader data already contains this; it just needs the rule.

**Should change (cheap)**

Baseline for the concept vectors (§3). The paper says the vectors were mean-subtracted over "100 other concepts", and the concept list has 101 entries. So the most plausible reading is a leave-one-out mean over the other 100 concepts; that is my inference, since the protocol file gives no list.
The draft's 194 category-instance words differ systematically from the concept list (concrete category members vs mixed abstract nouns). That can leave a component shared by every concept vector.
Use leave-one-out as primary and the 194 words as a sensitivity check. Report the cosine between the two versions; if it's above 0.9, the choice doesn't matter.

Conditions 5 and 6 (natural norm) have no contrast attached. Add M3: report for the J part vs the non-J part vs the full vector, all at natural norm.
At matched norm the J part is scaled up about 4× relative to its natural size (at roughly 6% J share), so M3 is the version that respects natural scale.
Otherwise drop 5 and 6.

Measure behaviour in the same forward the readers see (§5). P11b currently compares reader detection on the introspection prompt with behaviour on a different, neutral prompt. Add B4: the mean KL over the prefill positions of the introspection prompt, from the same forward as the reader captures. Use it alongside B1 and B2.

rand-J pool (§3). Draw its tokens from the 194 baseline words only. The current pool includes core concepts, which inflates their mismatched-concept false-positive rates.

Oracle "k = 4" (§6). Confirm whether the contract means 4 phrases per readout or 4 samples. The budget assumes 4 samples.
