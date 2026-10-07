# Hand checks before full-run scoring (013 spec §7 G2; amendment A1 G2k). Coded before any full-run Δ was computed.

## G2, A's contradicted-span precision
- Smoke: 62/63. Extra 10 (seed 29, outside the smoke): 27/27. **Combined 89/90 = 0.99** (bar 0.80): pass.
- Recall misses in the extra 10:
  - (corrected) b3-mh-sphinx-capital|rand|1 is not a recall miss. A flagged its misquotes, but one span was not a verbatim substring, so the explanation fails G1 (A_ok false) and is excluded; the hand-check file shows no spans for G1 failures;
  - bf-moonwalk-firstname|rand|1: two repeated misquotes ("The first name of the astronaut who first walked on the moon is") are not flagged, although the first one is;
  - pred-valentines-prevmonth|rand|1: "likely … November or March" conflicts with the continuation ("January") and is not flagged;
  - bf-rio2016-language|clean|0: the invented quote "…won the soccer World Cup in [year]…" is filed as unsupported.

## G2k, judge B's span kinds (rule: the first kind that applies; misquote before invented_context)
- Smoke: 57/63. Extra 10: 18/27. **Combined 75/90 = 0.83** (bar 0.80): passes narrowly.
- **Every disagreement is a misquote tagged invented_context:** a distorted paraphrase of the prompt, often with changed clue content or invented framing around it. Examples: "type of flower that blooms in month number" (prompt: "Emerald is the birthstone for month number"), "the mountain range in the country in Europe I mentioned above is the", "Fact 1: The 'month' before this time period is".
- **The precision of the invented_context tag is 24/39 = 0.62.** The invented_context set therefore mixes pure framing ("Think Fast", "Hint" headers, "</question>", "The following sentences are correct") with distorted paraphrases of the prompt. Its contamination (misquote-like spans that may carry prompt content) would push Δi toward "signal". The misquote tag is reliable (no false misquote tags seen).
- Consequence: Δi is reported with this caveat; a Δi near the signal side cannot be attributed to pure confabulation.
