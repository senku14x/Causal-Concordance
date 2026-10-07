### typo-across|clean|0  (src None -> tgt across)
PROMPT: She picked up her bag and walked slowly accross
CONTINUATION: ' the room.\nA. accross'
EXPLANATION:
Informal Chinese narrative fiction writing style, suggesting a first-person English story exercise or translation attempt by a Chinese speaker.

The phrase "was very quiet" sets a mood of tension or melancholy, establishing scene-setting that typically precedes action or observation in a narrative.

"across" ends a sentence fragment "The young man stood up and walked slowly acrossross" — "acrossross" is likely a misspelling of "across," so the next token probably completes a location phrase (e.g., "the room," "the street") or continues with a description of where he walked.
</br>"the young man walked slowly accross" is mid-sentence, strongly expecting a spatial object like "the room" or "the bridge."
"across" signals British-style spelling variant, suggesting continuation of informal spelling.

A contradicted (kind from B):
  C0 [invented_context]: 'The phrase "was very quiet" sets a mood of tension or melancholy, establishing scene-setting that typically precedes action or observation in a narrative.'
  C1 [misquote]: '"across" ends a sentence fragment "The young man stood up and walked slowly acrossross"'
  C2 [misquote]: '"the young man walked slowly accross" is mid-sentence, strongly expecting a spatial object like "the room" or "the bridge."'
A unsupported:
  U0: 'Informal Chinese narrative fiction writing style, suggesting a first-person English story exercise or translation attempt by a Chinese speaker.'
B labels: [(1, 'unsupported'), (2, 'contradicted'), (3, 'contradicted'), (4, 'contradicted'), (5, 'contradicted')]

### typo-across|clean|1  (src None -> tgt across)
PROMPT: She picked up her bag and walked slowly accross
CONTINUATION: ' the room.\nA. accross'
EXPLANATION:
Non-native or informal writing style signals continuation of a simple narrative scene, likely a student composition or creative writing piece in Chinese-to-English style.

The prompt "Write something in english?" establishes a personal/fictional writing task, expecting story-like descriptive sentences about a character.

"John looked worried and walked slowly accross" is mid-action narrative describing movement; "accross" is an incomplete prepositional phrase requiring a destination noun (e.g., "the room," "the street") or the word likely intends "across."
"John, sitting for a moment, walked slowly accross" — the typo "accross" (rather than "across") suggests early draft text; next token likely continues spatial movement or corrects to "across" followed by a location word like "the."
The sentence describes slow, purposeful movement.

A contradicted (kind from B):
  C0 [invented_context]: 'The prompt "Write something in english?" establishes a personal/fictional writing task, expecting story-like descriptive sentences about a character.'
  C1 [misquote]: '"John looked worried and walked slowly accross" is mid-action narrative describing movement;'
  C2 [misquote]: '"John, sitting for a moment, walked slowly accross" — the typo "accross" (rather than "across") suggests early draft text;'
A unsupported:
  U0: 'likely a student composition or creative writing piece in Chinese-to-English style.'
B labels: [(1, 'unsupported'), (2, 'contradicted'), (3, 'contradicted'), (4, 'contradicted'), (5, 'unsupported')]

### typo-action|clean|1  (src None -> tgt action)
PROMPT: He decided to take immediate aciton
CONTINUATION: ' to stop the spread of the disease.'
EXPLANATION:
Educational/test format established by "Directions:" header signals a correction or grammar exercise context. The student may be asked to identify errors in English.

The sentence structure "He decided to take imediate actiton" suggests an error-correction task involving a misspelling ("imediately"/"actiton"), common in Chinese-to-English writing practice.

"action" (misspelled as "actiton") ends a phrase "He decided to take quick actiton" — a simple past declarative sentence expecting completion with an object, prepositional phrase, or subordinate clause like "on the issue" or "immediately."
"actiton" (typo) sets up expected continuation describing *what action* or qualifying it, likely followed by the word "action" or a contrasting clause.
"He decided to take imediat action"

A contradicted (kind from B):
  C0 [invented_context]: 'Educational/test format established by "Directions:" header signals a correction or grammar exercise context.'
  C1 [misquote]: 'The sentence structure "He decided to take imediate actiton" suggests an error-correction task involving a misspelling ("imediately"/"actiton"), common in Chinese-to-English writing practice.'
  C2 [misquote]: '"action" (misspelled as "actiton") ends a phrase "He decided to take quick actiton" — a simple past declarative sentence expecting completion with an object, prepositional phrase, or subordinate clause like "on the issue" or "immediately."'
  C3 [wrong_entity]: '"actiton" (typo) sets up expected continuation describing *what action* or qualifying it, likely followed by the word "action" or a contrasting clause.'
  C4 [misquote]: '"He decided to take imediat action"'
A unsupported:
  U0: 'The student may be asked to identify errors in English.'
B labels: [(1, 'contradicted'), (2, 'unsupported'), (3, 'contradicted'), (4, 'contradicted'), (5, 'contradicted'), (6, 'contradicted')]

### typo-address|clean|0  (src None -> tgt address)
PROMPT: Please write your full name and your current adress
CONTINUATION: "\n\n<think>\nHere's a thinking process"
EXPLANATION:
Form/document instruction pattern established: "Please write your [field]" structure signals a form-filling prompt requiring personal information.
The phrase "your full name and current adress" sets up a dual-request prompt, expecting a response containing both pieces of information in a simple, direct format.
Final token "adress" (misspelled "address") is the second requested field in an instruction, with immediate expectation of a continuation like "address" correction, a comma, or city/country field completing the location details.
"adress" likely signals a student or casual register form prompt.
"Please write your full name and current adress" is a personal information collection instruction in English — expect continuation naming an address or similar field like "current adress in the following" or a colon.

A contradicted (kind from B):
  C0 [invented_context]: 'The phrase "your full name and current adress" sets up a dual-request prompt, expecting a response containing both pieces of information in a simple, direct format.'
  C1 [invented_context]: '"Please write your full name and current adress" is a personal information collection instruction in English — expect continuation naming an address or similar field like "current adress in the following" or a colon.'
A unsupported:
  U0: '"adress" likely signals a student or casual register form prompt.'
B labels: [(1, 'consistent'), (2, 'consistent'), (3, 'consistent'), (4, 'unsupported'), (5, 'consistent')]

### typo-address|clean|1  (src None -> tgt address)
PROMPT: Please write your full name and your current adress
CONTINUATION: "\n\n<think>\nHere's a thinking process"
EXPLANATION:
Form or prompt format initiated by "Please write your full name and current adress" signals a completion task, likely requesting personal information in a structured template or form context.

Language errors suggest non-native English speaker ("adress" instead of "address"), implying informal or practice-writing context with lower standards for correctness.

"adress" is the final word of an instruction prompt ("Please write your full name and current adress"), immediately requiring a completion word like "ss" correction or a continuation specifying location details, or punctuation/next field label.
"adress" likely continues the prompt as a personal information field label ("Please write your full name and current adress"), expecting either a colon, comma, or the next expected detail (e.g., "address").
"Please write your full name and current adress" — address or residence information expected.

A contradicted (kind from B):
  C0 [invented_context]: 'Form or prompt format initiated by "Please write your full name and current adress" signals a completion task, likely requesting personal information in a structured template or form context.'
  C1 [invented_context]: '"adress" is the final word of an instruction prompt ("Please write your full name and current adress"), immediately requiring a completion word like "ss" correction or a continuation specifying location details, or punctuation/next field label.'
  C2 [invented_context]: '"adress" likely continues the prompt as a personal information field label ("Please write your full name and current adress"), expecting either a colon, comma, or the next expected detail (e.g., "address").'
  C3 [invented_context]: '"Please write your full name and current adress" — address or residence information expected.'
A unsupported:
  U0: 'Language errors suggest non-native English speaker ("adress" instead of "address"), implying informal or practice-writing context with lower standards for correctness.'
B labels: [(1, 'consistent'), (2, 'unsupported'), (3, 'contradicted'), (4, 'contradicted'), (5, 'consistent')]
