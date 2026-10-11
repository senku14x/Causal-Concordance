# Interpretive text for make_page.py (exec'd there with A, C, SM, R1, HC, h, d, frac, TEXT). Numbers only via h() / d().
oo, on, nn, no = C["oracle_texts|oracle_ar"], C["oracle_texts|nla_ar"], C["nla_texts|nla_ar"], C["nla_texts|oracle_ar"]
rc = A["reader_contrasts_nla_minus_oracle"]
fx = lambda s: f"{h(s['mean'])} [{h(s['ci95'][0])}, {h(s['ci95'][1])}]"                                   # noqa: E731
raw = lambda x: f"{h(x['all']['raw']['mean'])} [{h(x['all']['raw']['ci95'][0])}, {h(x['all']['raw']['ci95'][1])}]"  # noqa: E731

TEXT["lede"] = (
    "013 found that the NLA's false claims help its own reconstructor rebuild the activation, through misquotes of the prompt. "
    "015 ran the same pipeline on the Oracle Lens and scored every version of both readers' texts with both readers' "
    "reconstructors. Under the Oracle's reconstructor, the true answer carries the most signal, and more than the same "
    "tokens cut from another bullet's opening. The Oracle's false claims also carry signal, and they are mostly wrong "
    "answers, though that part is not position-controlled. Under the NLA's reconstructor, none of this moves. The Oracle's "
    "reconstructor barely reads the NLA's texts, so the readers can only be compared under the NLA's reconstructor. There "
    "they differ, but no reader-specific claim survives the pre-registered rule.")

TEXT["design"] = (
    "<p>Two readers wrote text about the same final-token activations: the Oracle Lens (bullets that continue the prompt, "
    "read at L44) and the NLA (prose descriptions, read at L42). That is 87 multihop items × three states (clean, J-lens "
    "swap, random swap) × 2 samples = 522 texts each. Each reader has a co-trained reconstructor (AR) that maps text "
    "back to an activation. The test deletes or rewrites parts of a text, then asks how much worse the reconstruction gets "
    "compared with a matched control. Each text is scored by both ARs, so a result can be pinned to the reader's text or "
    "to the reconstructor.</p>")

TEXT["gates"] = (
    f"<p class=\"sub\">All gates pass. Two defects were found and fixed before any full-run score was written: the answer-span "
    f"locator deleted repeated answers only once, and the judge's paraphrases echoed the prompt's delimiters. Each fix is a "
    f"dated ledger entry below. Judge spend was $6.76 against a $10 cap.</p>")

TEXT["answer"] = (
    f"<p>Under the Oracle AR, deleting the Oracle's answer statements costs Δ_A {fx(oo['dA_RA_minus_CA'])} points more than "
    f"deleting the same number of tokens as random clauses. On the {oo['dAP_PA_minus_CA']['n']} readouts where a position-matched "
    f"control exists (the same token count cut from the opening of a bullet with no answer), the answer still costs "
    f"Δ_AP {fx(oo['dAP_PA_minus_CA'])}. Δ_A on those same readouts is {fx(oo['dA_on_PA_subset'])}, so the opening position accounts "
    f"for about a fifth of the effect there. Under the NLA AR, the Oracle's answers don't matter "
    f"({fx(on['dA_RA_minus_CA'])}). The NLA's own answer statements carry a little under its AR ({fx(nn['dA_RA_minus_CA'])}), "
    f"which is a lot per token: they are {h(A['answer_arm_coverage']['nla']['answer_token_share_mean'] * 100, 1, False)}% of its text.</p>")

TEXT["primary"] = (
    f"<p>The pre-registered expectation was that the Oracle's false claims would be padding for its reconstructor. They are "
    f"not: the primary Δ is {fx(oo['primary_R1_minus_C1'])}. By kind, the signal is in wrong answers (wrong_entity "
    f"{fx(oo['kind_wrong_entity']['delta_clause'])}; the judge's \"misquote\" tags on Oracle text are also wrong answers, every "
    f"one in the hand checks, {fx(oo['kind_misquote']['delta_clause'])}). Invented context is padding "
    f"({fx(oo['kind_invented_context']['delta_clause'])}). The wrong-answer effect is not position-controlled: cutting the "
    f"opening of an answer-free bullet costs only about {h(oo['dA_on_PA_subset']['mean'] - oo['dAP_PA_minus_CA']['mean'], 1, False)} "
    f"points more than random clauses, so for wrong answers content vs position is unresolved. On NLA texts, 013's result reproduces exactly "
    f"({fx(nn['primary_R1_minus_C1'])}, carried by misquotes {fx(nn['kind_misquote']['delta_clause'])}).</p>")

TEXT["rewrites"] = (
    f"<p>013's factual rewrite F is mostly deletion for the Oracle: it removes a net "
    f"{h((A['rewrites']['oracle']['F_composition_sum']['deleted'] + A['rewrites']['oracle']['F_composition_sum']['replaced_O'] - A['rewrites']['oracle']['F_composition_sum']['replaced_F'] - A['rewrites']['oracle']['F_composition_sum']['inserted']) / A['rewrites']['oracle']['F_composition_sum']['words_O'] * 100, 0, False)}% "
    f"of the words, and P removes none. So F − P ({fx(oo['F_minus_P'])} under the Oracle AR, {fx(on['F_minus_P'])} under the "
    f"NLA AR) is dominated by deletion size, and there is no size-matched control for F. The NLA's F − P under its own AR is {fx(nn['F_minus_P'])}. A correction-only rewrite (Fc, post hoc: fix each "
    f"flagged span in place, delete nothing) does not help either reconstructor. Fc − Pc is {fx(oo['Fc_minus_Pc'])} under the "
    f"Oracle AR and {fx(on['Fc_minus_Pc'])} under the NLA AR, and it is negative in clean states as well (post hoc check). "
    f"CW-20 also found that correcting the wrong pieces changes nothing.</p>")

TEXT["contrasts"] = (
    f"<p>A reader-specific claim needed the same sign under both reconstructors, on English-only text, and per unit of "
    f"deleted text. None qualifies. The false-claim contrast (NLA − Oracle) is {raw(rc['primary']['oracle_ar'])} under the "
    f"Oracle AR and {raw(rc['primary']['nla_ar'])} under the NLA AR. The answer contrast is {raw(rc['dA']['oracle_ar'])} and "
    f"{raw(rc['dA']['nla_ar'])}. The Oracle AR barely reads NLA texts (mean cosine {h(no['cos_mean']['O'], 3, False)}), so "
    f"contrasts under it are mostly the Oracle side, and cross-scoring can adjudicate only under the NLA AR. There the "
    f"readers differ, but a one-scorer difference is not a reader-specific claim under the rule. What the 2 × 2 does show "
    f"is that on the same Oracle texts, which both reconstructors read, the Oracle AR rewards answer slots and the NLA AR "
    f"does not. The scale-free column is descriptive only.</p>")

TEXT["caveats"] = (
    "<ul>"
    "<li>The reconstructors are each reader's co-trained critic. This measures what each critic rewards in text, not what Qwen computes.</li>"
    "<li>Scope: WorkspaceBench multihop through Jan Chen's bank, the final prompt token, Oracle L44 and NLA L42, Gemini 3.8 Flash and GLM-5.3-flash judges, 2 samples per state.</li>"
    "<li>Judge recall misses leave some answer text in CA, so Δ_A is if anything understated. The Oracle's \"misquote\" kind is unreliable: its tags were wrong in 6 of 6 hand checks.</li>"
    "<li>The position-matched control exists only for readouts with an answer-free bullet (202 of 408), so Δ_AP comes from readouts with more wrong-answer bullets.</li>"
    "<li>The Fc arm, the scale-free view and the per-state Fc check are post hoc and descriptive. Everything is exploratory.</li>"
    "<li>Paraphrase matching is imperfect (Oracle P within 20% of F's edit size on 308 of 522); the matched subset gives the same signs.</li>"
    "</ul>")

sw, sn = SWP["cells"]["oracle_texts|oracle_ar"], SWP["cells"]["oracle_texts|nla_ar"]
rw_ = RUP["by_state"]["wrong"]["all"]
rt_ = RUP["by_state"]["true"]["all"]
TEXT["followups"] = (
    f"<p>Two post hoc checks prompted by a red-team review, both designed in the ledger before they ran. <b>Answer swaps:</b> "
    f"in the {sw['n']} Oracle readouts that have both a true-answer opening and a wrong-answer opening, the openings were moved using "
    f"only the readout's own strings. Under the Oracle AR every rearrangement costs points: true → wrong {fx(sw['O_minus_ALLW'])}, "
    f"wrong → true {fx(sw['ALLT_minus_O'])}, and exchanging them with the same strings {fx(sw['O_minus_SWAP'])}. All-true against "
    f"all-wrong is unresolved ({fx(sw['ALLT_minus_ALLW'])}). So the Oracle AR seems to reward each bullet as the inverter wrote it, more "
    f"than the true answer itself (a hypothesis). The NLA AR, which reads the whole text at once, is unmoved by the exchange "
    f"({fx(sn['O_minus_SWAP'])}), as it should be.</p>"
    f"<p><b>Runner-ups:</b> each wrong-answer opening was scored as Qwen's continuation of the prompt, at the read cell, in that "
    f"readout's state, with Stage B's edits replayed. It scores {fx(rw_['logp_own_minus_null'])} nats higher there than at another "
    f"item's read cell. Its first token has median rank {h(rw_['rank_own_median'], 1, False)}, against "
    f"{h(rw_['rank_null_median'], 1, False)} at another item's cell. The model's own answer, the positive control, scores "
    f"{fx(rt_['logp_own_minus_null'])}. So the Oracle's \"wrong answers\" are mostly the model's next-best candidates at that "
    f"position. The pre-stated gate (argmax equals Stage B's greedy token at all 261 cells) missed on 6 near-tied cells; "
    f"without them the result is {h(RUP['wrong_all_excluding_mismatch_cells']['mean'])}.</p>")
