# 015 smoke hand checks (G2, G2k, G2a on Oracle readouts; G2a on 10 NLA explanations; F/P)

## atomic-26-symbol|rand_L24-35|1 (src iron -> tgt copper)  non-English: True
PROMPT: Fact: The chemical symbol for the element with atomic number 26 is
CONTINUATION: ' Fe.\n'

READOUT:
- Co. Its atomic number is
- “Fe”, and its relative atomic mass is approximately 55.9. What is the symbol for the element with atomic number 5
- Fe, and its atomic number is 26. (2 marks) The abbreviation for the chemical symbol of the element with atomic
- **Fe**. The atomic mass of this element is approximately 55.85. Its electron configuration ends in 4s²3d�

A contradicted [kind]:
  C0 [wrong_entity]: 'Co.'
A unsupported:

ANSWER spans:
  S0: '“Fe”'
  S1: 'Fe'
  S2: '**Fe**'

F (words changed 28):
- Fe. Its atomic number is 26.
- “Fe”, and its relative atomic mass is approximately 55.85.
- Fe, and its atomic number is 26.
- **Fe**. The atomic mass of this element is approximately 55.85. Its electron configuration ends in 4s²3d⁶

P (target 28, changed 27, re-asked False):
- Co. The atomic number of it is
- “Fe”, while its relative atomic mass stands near 55.9. Which symbol represents the element having atomic number 5
- Fe, where its atomic number equals 26. (2 marks) The short form of the chemical symbol for the element with atomic
- **Fe**. The atomic weight for this element equals roughly 55.85. The electron configuration for it concludes with 4s²3d

Fc (POST HOC; words changed 2, same line count True):
- Fe. Its atomic number is
- “Fe”, and its relative atomic mass is approximately 55.9. What is the symbol for the element with atomic number 5
- Fe, and its atomic number is 26. (2 marks) The abbreviation for the chemical symbol of the element with atomic
- **Fe**. The atomic mass of this element is approximately 55.85. Its electron configuration ends in 4s²3d

Pc (target 2, changed 3, re-asked True):
- Co. Its atomic number is
- “Fe”, and its relative atomic mass is roughly 55.9. What is the sign for the element with atomic number 5
- Fe, and its atomic number is 26. (2 marks) The abbreviation for the chemical symbol of the element with atomic
- **Fe**. The atomic mass of this element is approximately 55.85. Its electron configuration ends in 4s²3d

## atomic-79-symbol|J_L24-35|0 (src gold -> tgt silver)  non-English: False
PROMPT: Fact: The chemical symbol for the element with atomic number 79 is
CONTINUATION: ' Au.\n'

READOUT:
- Au, and its name is Gold. If the atomic number of this element is
- “Pt”, and its relative atomic mass is 195. What is the full name of the element with symbol
- Au, and its atomic number is 79. (What is the full name of this chemical element?) The symbol for platinum is
- ‘Pt’. Its atomic number is 78, and its standard state is solid. The element with atomic number 195 is

A contradicted [kind]:
  C0 [misquote]: '“Pt”, and its relative atomic mass is 195.'
  C1 [misquote]: '‘Pt’.'
  C2 [invented_context]: 'The element with atomic number 195 is'
A unsupported:

ANSWER spans:
  S0: 'Au'
  S1: 'Au'

F (words changed 35):
- Au, and its name is Gold. The atomic number of this element is 79.
- Au, and its relative atomic mass is 197.
- Au, and its atomic number is 79. The symbol for gold is Au.
- Au. Its atomic number is 79, and its standard state is solid.

P (target 35, changed 37, re-asked True):
- Au, and it is called Gold. In the event that the atomic number belonging to this element is
- “Pt”, and its standard atomic weight is 195. Which full name belongs to the element possessing the symbol
- Au, and it has an atomic number of 79. (What constitutes the full title of this chemical element?) The chemical symbol representing platinum is
- ‘Pt’. It has an atomic number of 78, and it exists standardly as a solid. The chemical element possessing atomic number 195 is

Fc (POST HOC; words changed 4, same line count True):
- Au, and its name is Gold. If the atomic number of this element is
- Au, and its relative atomic mass is 197. What is the full name of the element with symbol
- Au, and its atomic number is 79. (What is the full name of this chemical element?) The symbol for platinum is
- ‘Au’. Its atomic number is 78, and its standard state is solid. The element with atomic mass 195 is

Pc (target 4, changed 3, re-asked True):
- Au, and its name is Gold. If the atomic number of this substance is
- “Pt”, and its relative atomic mass is 195. What is the full name of the element with symbol
- Au, and its atomic number is 79. (What is the complete name of this chemical element?) The symbol for platinum is
- ‘Pt’. Its atomic number is 78, and its standard phase is solid. The element with atomic number 195 is

## b3-mh-aztec-language|clean|1 (src Mexico -> tgt Japan)  non-English: True
PROMPT: Fact: The language spoken today in the country of the ancient Aztecs is
CONTINUATION: ' Spanish.\n'

READOUT:
- _____ (Spanish). Today, the native languages of Mexico are
- 西班牙语 (Spanish). The Aztecs spoke a different language. What language is spoken in Mexico today
- 西班牙语 (Spanish), which is a descendant of the language spoken by the Aztecs. What is the current language of Mexico
- ________, while the indigenous people still speak various languages. (Answer: Spanish)2. What is the currency of Mexico? (Answer:

A contradicted [kind]:
  C0 [false_fact]: 'which is a descendant of the language spoken by the Aztecs'
A unsupported:

ANSWER spans:
  S0: '_____ (Spanish)'
  S1: '西班牙语 (Spanish)'
  S2: '西班牙语 (Spanish)'

F (words changed 38):
- Spanish. Today, the native languages of Mexico are
- 西班牙语 (Spanish). The Aztecs spoke a different language.
- 西班牙语 (Spanish).
- Spanish, while the indigenous people still speak various languages.

P (target 38, changed 33, re-asked False):
<<<- _____ (Spanish). Currently, the indigenous tongues of Mexico are
- 西班牙语 (Spanish). The Aztec civilization used another tongue. Which language do people speak in Mexico currently
- 西班牙语 (Spanish), which evolved from the tongue used by the Aztecs. Which is the present-day language of Mexico
- ________, though native populations continue to use diverse tongues. (Answer: Spanish)2. Which is the money of Mexico? (Answer:>>>

Fc (POST HOC; words changed 1, same line count True):
- _____ (Spanish). Today, the native languages of Mexico are
- 西班牙语 (Spanish). The Aztecs spoke a different language. What language is spoken in Mexico today
- 西班牙语 (Spanish), which is not a descendant of the language spoken by the Aztecs. What is the current language of Mexico
- ________, while the indigenous people still speak various languages. (Answer: Spanish)2. What is the currency of Mexico? (Answer:

Pc (target 1, changed 1, re-asked True):
- _____ (Spanish). Today, the indigenous languages of Mexico are
- 西班牙语 (Spanish). The Aztecs spoke a different language. What language is spoken in Mexico today
- 西班牙语 (Spanish), which is a descendant of the language spoken by the Aztecs. What is the current language of Mexico
- ________, while the indigenous people still speak various languages. (Answer: Spanish)2. What is the currency of Mexico? (Answer:

## b3-mh-ikea-capital|rand_L24-35|1 (src Sweden -> tgt Norway)  non-English: True
PROMPT: Fact: The capital of the country where IKEA was founded is
CONTINUATION: ' Stockholm.\n'

READOUT:
- [Answer]. The capital of the country where that company is founded is
- 柏林 (Berlin). What is the capital of the country where that famous footwear brand was founded
-  Warsaw, while the capital of the country where its main sport was invented is Berlin. Identify the capital
- Berlin (or another capital, depending on the country). The country's oldest city is X."* (This is a riddle style, but the capital is

A contradicted [kind]:
  C0 [wrong_entity]: '柏林 (Berlin)'
  C1 [wrong_entity]: 'Warsaw'
  C2 [wrong_entity]: 'Berlin (or another capital, depending on the country)'
A unsupported:

ANSWER spans:


F (words changed 5):
- Stockholm. The capital of the country where that company is founded is
- 斯德哥尔摩 (Stockholm). What is the capital of the country where that famous footwear brand was founded
- Stockholm, while the capital of the country where its main sport was invented is Berlin. Identify the capital
- Stockholm (or another capital, depending on the country). The country's oldest city is X."* (This is a riddle style, but the capital is

P (target 5, changed 6, re-asked False):
<<<- [Answer]. The capital city of the nation where that company is founded is
- 柏林 (Berlin). What is the capital of the country where that famous footwear brand was established
-  Warsaw, while the capital of the country where its main sport was invented is Berlin. Name the capital
- Berlin (or another capital, depending on the country). The country's oldest city is X."* (This is a riddle style, but the capital is>>>

Fc (POST HOC; words changed 4, same line count True):
- [Answer]. The capital of the country where that company is founded is
- 斯德哥尔摩 (Stockholm). What is the capital of the country where that famous footwear brand was founded
-  Stockholm, while the capital of the country where its main sport was invented is Berlin. Identify the capital
- Stockholm (or another capital, depending on the country). The country's oldest city is X."* (This is a riddle style, but the capital is

Pc (target 4, changed 4, re-asked True):
<<<- [Answer]. The capital of the nation where that company was founded is
- 柏林 (Berlin). What is the capital of the country where that famous footwear brand was founded
-  Warsaw, while the capital of the country where its main sport was invented is Berlin. Identify the capital
- Berlin (or another capital, depending on the country). The country's oldest city is X."* (This is a riddle style, but the capital is>>>

## b3-mh-lisbon-language|rand_L24-35|0 (src Portugal -> tgt Japan)  non-English: True
PROMPT: Fact: The language spoken in the country whose capital is Lisbon is
CONTINUATION: ' Portuguese.\n'

READOUT:
- [Language X]. Clue 2: The currency used in Portugal is
- Portuguese. (Because the main language spoken in Portugal is Portuguese.) Fact 2: The country where Lisbon is
- 葡萄牙语 (Portuguese), and the country's capital is Lisbon." | Step 1: Determine language spoken in Portugal. |
- [language spoken there] (Hint: Portugal's main language is Portuguese)." (Fact: The primary language of Portugal is
- Portuguese. (Fact: The language spoken in Portugal is Portuguese.) Rule 2: People who live in Portugal are

A contradicted [kind]:
  C0 [wrong_entity]: '[Language X]'
A unsupported:

ANSWER spans:
  S0: 'Portuguese.'
  S1: '葡萄牙语 (Portuguese)'
  S2: 'Portuguese.'

F (words changed 51):
- Portuguese.
- Portuguese. (Because the main language spoken in Portugal is Portuguese.)
- 葡萄牙语 (Portuguese), and the country's capital is Lisbon.
- Portuguese.
- Portuguese. (Fact: The language spoken in Portugal is Portuguese.)

P (target 51, changed 37, re-asked True):
<- [Tongue X]. Clue 2: The money utilized across Portugal is
- Portuguese. (Since the dominant tongue spoken within Portugal is Portuguese.) Fact 2: The nation where Lisbon exists
- 葡萄牙语 (Portuguese), and that nation's seat of government is Lisbon." | Step 1: Identify tongue spoken within Portugal. |
- [tongue spoken in that place] (Tip: Portugal's primary tongue is Portuguese)." (Fact: The chief tongue of Portugal represents
- Portuguese. (Fact: The tongue used within Portugal is Portuguese.) Rule 2: Individuals residing within Portugal are>

Fc (POST HOC; words changed 2, same line count True):
- Portuguese. Clue 2: The currency used in Portugal is
- Portuguese. (Because the main language spoken in Portugal is Portuguese.) Fact 2: The country where Lisbon is
- 葡萄牙语 (Portuguese), and the country's capital is Lisbon." | Step 1: Determine language spoken in Portugal. |
- [language spoken there] (Hint: Portugal's main language is Portuguese)." (Fact: The primary language of Portugal is
- Portuguese. (Fact: The language spoken in Portugal is Portuguese.) Rule 2: People who live in Portugal are

Pc (target 2, changed 4, re-asked True):
<<<- [Language X]. Clue 2: The currency utilized in Portugal is
- Portuguese. (Because the main language spoken in Portugal is Portuguese.) Fact 2: The nation where Lisbon is
- 葡萄牙语 (Portuguese), and the country's capital is Lisbon." | Step 1: Determine language spoken in Portugal. |
- [language spoken there] (Hint: Portugal's main language is Portuguese)." (Fact: The primary language of Portugal is
- Portuguese. (Fact: The language spoken in Portugal is Portuguese.) Rule 2: People who live in Portugal are>>>

## b3-mh-operahouse-country|J_L24-35|0 (src Sydney -> tgt Cairo)  non-English: True
PROMPT: Fact: The country of the famous sail-shaped opera house is
CONTINUATION: ' Australia.\n'

READOUT:
- 澳大利亚, and the famous building is located in its capital city. What is the country
- 澳大利亚 (Australia), and the city where this famous opera house is located is known for its
- ______ (hint: it's known for the Great Barrier Reef and Opera House). In that country, the currency used is ______." (The country is Australia;
- 澳大利亚 (Australia), and the building’s famous symbol is the Opera House. What is the country’s capital?

A contradicted [kind]:
  C0 [false_fact]: 'the famous building is located in its capital city'
A unsupported:

ANSWER spans:
  S0: '澳大利亚'
  S1: '澳大利亚'
  S2: 'The country is Australia'
  S3: '澳大利亚'

F (words changed 13):
- 澳大利亚, and the famous building is located in its capital city. What is the country
- 澳大利亚 (Australia), and the city where this famous opera house is located is known for its
- Australia (hint: it's known for the Great Barrier Reef and Opera House).
- 澳大利亚 (Australia), and the building’s famous symbol is the Opera House. What is the country’s capital?

P (target 13, changed 11, re-asked False):
- 澳大利亚, and the well-known building is situated in its capital city. Which is the country
- 澳大利亚 (Australia), and the town where this famous opera house is situated is recognized for its
- ______ (clue: it's known for the Great Barrier Reef and Opera House). In that nation, the currency used is ______." (The nation is Australia;
- 澳大利亚 (Australia), and the structure's famous symbol is the Opera House. Which is the country’s capital?

Fc (POST HOC; words changed 1, same line count True):
- 澳大利亚, and the famous building is located in its largest city. What is the country
- 澳大利亚 (Australia), and the city where this famous opera house is located is known for its
- ______ (hint: it's known for the Great Barrier Reef and Opera House). In that country, the currency used is ______." (The country is Australia;
- 澳大利亚 (Australia), and the building’s famous symbol is the Opera House. What is the country’s capital?

Pc (target 1, changed 3, re-asked True):
<<<- 澳大利亚, and the famous building is situated in its capital city. What is the country
- 澳大利亚 (Australia), and the city where this famous opera house is located is known for its
- ______ (hint: it's known for the Great Barrier Reef and Opera House). In that country, the currency used is ______." (The country is Australia;
- 澳大利亚 (Australia), and the building’s famous symbol is the Opera House. What is the country’s capital?>>>

## bf-nile-desert|rand_L24-35|0 (src Africa -> tgt Chile)  non-English: True
PROMPT: Fact: The largest hot desert, found on the continent where the Nile flows, is the
CONTINUATION: ' Sahara.\nQuestion: Which'

READOUT:
- 撒哈拉沙漠 (Sahara Desert). Here is the reasoning why the
- Great Sandy Desert in North Africa. However, here is a common trivia fact: The largest hot desert in the world is the
- Sahara Desert, which is the largest hot desert in the world. Here is the correct answer and a
- Libyan Desert (or Sahara). Actually, the question likely means the largest desert *in Europe* is the Sahara, which spans into Africa, but the

A contradicted [kind]:
  C0 [misquote]: 'Great Sandy Desert in North Africa'
  C1 [wrong_entity]: 'Libyan Desert (or Sahara)'
  C2 [false_fact]: 'the largest desert *in Europe* is the Sahara, which spans into Africa'
A unsupported:

ANSWER spans:
  S0: '撒哈拉沙漠 (Sahara Desert)'
  S1: 'Sahara Desert'

F (words changed 20):
- 撒哈拉沙漠 (Sahara Desert). Here is the reasoning why the
- Sahara Desert in North Africa. However, here is a common trivia fact: The largest hot desert in the world is the
- Sahara Desert, which is the largest hot desert in the world. Here is the correct answer and a
- Sahara. Actually, the largest hot desert, found on the continent where the Nile flows, is the Sahara, but the

P (target 20, changed 20, re-asked True):
<<<- 撒哈拉沙漠 (Sahara Desert). Below is the rationale for why the
- Great Sandy Desert in North Africa. However, here is a frequent trivia fact: The biggest hot desert in the world is the
- Sahara Desert, which is the world's largest hot desert. Here is the right response and a
- Libyan Desert (or Sahara). In fact, the query probably implies the largest desert *in Europe* is the Sahara, which extends into Africa, but the>>>

Fc (POST HOC; words changed 9, same line count True):
- 撒哈拉沙漠 (Sahara Desert). Here is the reasoning why the
- Sahara Desert in North Africa. However, here is a common trivia fact: The largest hot desert in the world is the
- Sahara Desert, which is the largest hot desert in the world. Here is the correct answer and a
- Sahara. Actually, the question likely means the largest desert *in Africa* is the Sahara, which spans across North Africa, but the

Pc (target 9, changed 11, re-asked True):
<<<- 撒哈拉沙漠 (Sahara Desert). Here is the reasoning behind why the
- Great Sandy Desert in North Africa. Yet, here is a familiar trivia fact: The largest hot desert in the world is the
- Sahara Desert, which is the biggest hot desert globally. Here is the correct answer and a
- Libyan Desert (or Sahara). Actually, the question likely intends the largest desert *in Europe* is the Sahara, which extends into Africa, but the>>>

## etym-caesar-monthnum|clean|1 (src July -> tgt October)  non-English: True
PROMPT: Fact: The month named after Julius Caesar is month number
CONTINUATION: ' 7.\nQuestion'

READOUT:
- 4 in the calendar year. What is the month number
- 5 (or month number 6) in the Gregorian calendar. October is month number
- （number） 8 in the standard Gregorian calendar. Which month is Spring 2?" (Note: In the Northern Hemisphere, March is month number
- 3 (or month number 6) in the year. What is the name of the season that October belongs to? The answer is

A contradicted [kind]:
  C0 [wrong_entity]: '4 in the calendar year.'
  C1 [wrong_entity]: '5 (or month number 6) in the Gregorian calendar.'
  C2 [wrong_entity]: '（number） 8 in the standard Gregorian calendar.'
  C3 [wrong_entity]: '3 (or month number 6) in the year.'
A unsupported:

ANSWER spans:


F (words changed 13):
- 7 in the calendar year. What is the month number
- 7 in the Gregorian calendar. October is month number
- 7 in the standard Gregorian calendar. Which month is Spring 2?" (Note: In the Northern Hemisphere, March is month number
- 7 in the year. What is the name of the season that October belongs to? The answer is

P (target 13, changed 12, re-asked True):
- 4 during the calendar year. Which is the month count
- 5 (or month number 6) within the Gregorian calendar. October represents month number
- （number） 8 in the standard Gregorian calendar. Which month denotes Spring 2?" (Note: Throughout the Northern Hemisphere, March is month count
- 3 (or month number 6) in the year. What is the name of the season where October belongs? The solution is

Fc (POST HOC; words changed 13, same line count True):
- 7 in the calendar year. What is the month number
- 7 in the Gregorian calendar. October is month number
- 7 in the standard Gregorian calendar. Which month is Spring 2?" (Note: In the Northern Hemisphere, March is month number
- 7 in the year. What is the name of the season that October belongs to? The answer is

Pc (target 13, changed 11, re-asked False):
- 4 in the annual year. Which is the month's number
- 5 (or month number 6) in the Gregorian calendar. October has month number
- （number） 8 in the official Gregorian calendar. What month is Spring 2?" (Note: For the Northern Hemisphere, March is month number
- 3 (or month number 6) within the year. What is the name of the season which October pertains to? The reply is

## nhop-spider-element|rand_L24-35|0 (src 8 -> tgt six)  non-English: True
PROMPT: Count the items in a dozen. The element at that position on the periodic table is magnesium. Count the legs on a spider. The element at that position on the periodic table is
CONTINUATION: ' oxygen. Count'

READOUT:
- Zinc (Zn). If the first letter is a number (like 1 for Hydrogen
- Neon (atomic number 10). (Note: In chemistry puzzles, the 4th element is often
- 硫 (Sulfur). (If we count the letters in the word "Love", the 4th letter is V, atomic number 5 is
- Chromium (Cr). (Note: The 6th element is Carbon, but let's look at the next letter for the second element

A contradicted [kind]:
  C0 [wrong_entity]: 'Zinc (Zn)'
  C1 [wrong_entity]: 'Neon (atomic number 10)'
  C2 [wrong_entity]: '硫 (Sulfur)'
  C3 [false_fact]: 'atomic number 5 is'
  C4 [wrong_entity]: 'Chromium (Cr)'
A unsupported:

ANSWER spans:


F (words changed 63):
- Oxygen (O).
- Oxygen (atomic number 8).
- 氧 (Oxygen).
- Oxygen (O).

P (target 63, changed 64, re-asked False):
- Zinc (Zn). Provided that the initial character represents a digit (such as 1 representing Hydrogen
- Neon (element number 10). (Remark: Within science riddles, element number 4 frequently represents
- 硫 (Sulfur). (Supposing one enumerates every character inside the term "Love", fourth stands as V, while element 5 stands as
- Chromium (Cr). (Remark: Element 6 corresponds to Carbon; however, consider examining the subsequent character regarding element two

Fc (POST HOC; words changed 9, same line count True):
- Oxygen (O). If the first letter is a number (like 1 for Hydrogen
- Oxygen (atomic number 8). (Note: In chemistry puzzles, the 4th element is often
- 氧 (Oxygen). (If we count the letters in the word "Love", the 4th letter is V, atomic number 23 is
- Oxygen (O). (Note: The 6th element is Carbon, but let's look at the next letter for the second element

Pc (target 9, changed 9, re-asked False):
- Zinc (Zn). In case the initial letter is a number (such as 1 for Hydrogen
- Neon (atomic number 10). (Note: In chemistry puzzles, the 4th element is frequently
- 硫 (Sulfur). (If we count the letters in the word "Love", the 4th letter is V, atomic number 5 is
- Chromium (Cr). (Note: The 6th element is Carbon, but let's check the following letter for the second element

## super-smallest-continent|rand_L24-35|0 (src Vatican -> tgt Japan)  non-English: False
PROMPT: Fact: The smallest country in the world is located on the continent of
CONTINUATION: ' Europe.\n'

READOUT:
- Asia. What continent is the headquarters of this largest company located in?
- Asia, and its owner is a mammal from the class Mammalia. If this company's headquarters is located on the continent of
- Asia (or specifically, the continent of Asia). If this company is located on the continent of Africa, how many letters are in the name of
- [Continent A] and its headquarters is located on the continent of [Continent B]. Which continent is the Statue of Liberty located on

A contradicted [kind]:
  C0 [wrong_entity]: 'Asia.'
  C1 [wrong_entity]: 'Asia,'
  C2 [wrong_entity]: 'Asia (or specifically, the continent of Asia).'
A unsupported:

ANSWER spans:


F (words changed 56):
- Europe.
- Europe, and its owner is a mammal from the class Mammalia.
- Europe (or specifically, the continent of Europe).
- Europe. Which continent is the Statue of Liberty located on

P (target 56, changed 47, re-asked False):
- Asia. Which continent contains the corporate base of this leading firm?
- Asia, while its proprietor represents a mammal within class Mammalia. Should this enterprise's corporate base reside upon the landmass of
- Asia (more precisely, the landmass of Asia). Were this enterprise based within the landmass of Africa, what is the letter count for
- [Continent A] while its corporate base resides upon the landmass of [Continent B]. What continent holds the Statue of Liberty

Fc (POST HOC; words changed 4, same line count True):
- Europe. What continent is the headquarters of this largest company located in?
- Europe, and its owner is a mammal from the class Mammalia. If this company's headquarters is located on the continent of
- Europe (or specifically, the continent of Europe). If this company is located on the continent of Africa, how many letters are in the name of
- [Continent A] and its headquarters is located on the continent of [Continent B]. Which continent is the Statue of Liberty located on

Pc (target 4, changed 4, re-asked True):
- Asia. Which continent is the headquarters of this largest company located in?
- Asia, and its owner is a mammal belonging to the class Mammalia. If this company's headquarters is situated on the continent of
- Asia (or specifically, the continent of Asia). If this company is located on the continent of Africa, how many letters are in the name of
- [Continent A] and its headquarters is located on the continent of [Continent B]. Which continent is the Statue of Liberty located on
