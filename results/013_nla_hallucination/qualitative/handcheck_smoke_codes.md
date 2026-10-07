# G2 hand check, smoke (20 explanations, 63 spans flagged "contradicted" by judge A), coded before any Δ was computed

**Precision of A's contradicted spans: 62/63 = 0.98** (bar ≥ 0.80): **G2 passes.**
- The one disagreement: b3-mh-lisbon-language|clean|1 C0 (`The sentence follows a structured template: "… [X] that [descriptor], which is in [region], is"`). It describes a template with placeholders rather than quoting the prompt; the "[region]" clause is not in the prompt. Borderline; counted as a disagreement.
- Borderline but agreed: carnival-ocean|rand|0 C3 (an invented "repeated structure"); inv-sunrise-opposite|rand|1 C1 (a misquote of the prompt).

**Missed or under-categorised (descriptive recall):**
- bf-eiffel-currency|rand|1, sentence 2: `The question "What is the currency of France?" has been restated` is invented text, not flagged (B flags it).
- b3-mh-windycity-state|clean|1, sentence 4: the invented quote `"The state that contains the city that hosts..."` is not flagged.
- Invented quotes filed as *unsupported* instead of contradicted (they are deleted in C2, not C1):
  - b3-mh-aztec-language|rand|1 U0 (`"The language spoken in ancient Mexico."`) and U1 (`"Ancient people to modern locations"`);
  - b3-mh-redplanet-moons|J|1 U0 (`Exam preparation context (BIO-100, …)`).

**Observation for interpretation** (not a gate): most contradicted spans are of two kinds.
- **(i) Invented context:** headers, quiz names, earlier questions ("Think Fast", "Hint", "Answer: C").
- **(ii) Misquotes of the prompt that still carry the bridge or answer:**
  - "the country that has the Tokyo tower [Japan]" after the Russia → Japan swap;
  - "The country of the Vatican is located on the continent of";
  - "The number of moons of Mars is".

Kind (ii) is contradicted by rule (a) but contains true, relevant content. Deleting it removes bridge information along with the falsehood. A positive Δ therefore partly means "misquoted paraphrases carry the bridge". I propose (post hoc, labelled) to split C1's spans into (i) and (ii) in the full run, with judge B tagging each span's kind. This would be a written amendment before the full run, not a change to the smoke.
