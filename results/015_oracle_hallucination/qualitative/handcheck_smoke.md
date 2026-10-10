# 015 smoke hand checks (G2, G2k, G2a on Oracle readouts; G2a on 10 NLA explanations; F/P)

## atomic-79-symbol|rand_L24-35|1 (src gold -> tgt silver)  non-English: False
PROMPT: Fact: The chemical symbol for the element with atomic number 79 is
CONTINUATION: ' Au.\n'

READOUT:
- "Pt", and its relative atomic mass is approximately 195.094. What is the element whose symbol is
- “Au”, and its atomic number is 79. The element with the symbol “Pt” is
- Au. The atomic weight of the element with this symbol is 107.94. What is the common name of
- **Pt**, and its atomic number is 78. Its year of discovery is a multiple of 4."*

A contradicted [kind]:
  C0 [misquote]: '"Pt"'
  C1 [false_fact]: 'The atomic weight of the element with this symbol is 107.94.'
  C2 [misquote]: '**Pt**'
A unsupported:

ANSWER spans:
  S0: '“Au”'
  S1: 'Au.'

F (words changed 5):
- "Au", and its relative atomic mass is approximately 196.967. What is the element whose symbol is
- “Au”, and its atomic number is 79. The element with the symbol “Pt” is
- Au. The atomic weight of the element with this symbol is 196.97. What is the common name of
- **Au**, and its atomic number is 79. Its year of discovery is a multiple of 4."*

P (target 5, changed 4, re-asked True):
<<<- "Pt", and its relative atomic mass is roughly 195.094. What is the element whose symbol is
- “Au”, and its atomic number is 79. The element with the symbol “Pt” is
- Au. The atomic mass of the element with this symbol is 107.94. What is the common name of
- **Pt**, and its atomic number is 78. Its year of discovery is a multiple of 4."*>>>

## b3-mh-aztec-language|rand_L24-35|1 (src Mexico -> tgt Japan)  non-English: True
PROMPT: Fact: The language spoken today in the country of the ancient Aztecs is
CONTINUATION: ' Nahuatl'

READOUT:
- Spanish. The ancient language spoken by the Aztecs is now
- 现代西班牙语 (Modern Spanish), not the language spoken by the Aztecs today. Mexico City is the capital of
- _________, while the old language of the region is Nahuatl. What language is primarily spoken in modern-day Mexico?
- 墨西哥语 (Mexican Spanish), not the language spoken by the ancient civilization. | The primary language spoken in Mexico today is Spanish. The Az

A contradicted [kind]:
  C0 [wrong_entity]: 'Spanish.'
A unsupported:

ANSWER spans:
  S0: 'Nahuatl'

F (words changed 10):
- Nahuatl. The ancient language spoken by the Aztecs is now
- 现代纳瓦特尔语 (Modern Nahuatl), the language spoken by the Aztecs today. Mexico City is the capital of
- Nahuatl, while the old language of the region is Nahuatl. What language is primarily spoken in modern-day Mexico?
- 纳瓦特尔语 (Nahuatl), the language spoken by the ancient civilization. | The primary language spoken in Mexico today is Nahuatl. The Az

P (target 10, changed 10, re-asked False):
<<<- Spanish. The historic language spoken by the Aztecs is now
- 现代西班牙语 (Modern Spanish), not the tongue spoken by the Aztecs today. Mexico City is the capital of
- _________, while the ancient language of the area is Nahuatl. What tongue is mainly spoken in modern-day Mexico?
- 墨西哥语 (Mexican Spanish), not the language used by the ancient civilization. | The main language spoken in Mexico today is Spanish. The Az>>>

## b3-mh-fjords-capital|J_L24-35|0 (src Norway -> tgt Sweden)  non-English: True
PROMPT: Fact: The capital of the country famous for fjords is
CONTINUATION: ' Oslo.\n'

READOUT:
- [Answer]. The capital of Norway is
- 奥斯陆 (Oslo), and its currency is the krone. If you were to visit the capital of this Nordic country
- Oslo, while the capital of its largest neighboring Scandinavian country is Stockholm.** *(Clue: Sweden’s capital is Stockholm
- Stockholm (or more precisely, the capital of the Nordic country whose name starts with N). 2. The national animal of this country is

A contradicted [kind]:
  C0 [misquote]: 'Stockholm (or more precisely, the capital of the Nordic country whose name starts with N).'
A unsupported:

ANSWER spans:
  S0: '奥斯陆 (Oslo)'
  S1: 'Oslo'

F (words changed 3):
- Oslo. The capital of Norway is
- 奥斯陆 (Oslo), and its currency is the krone. If you were to visit the capital of this Nordic country
- Oslo, while the capital of its largest neighboring Scandinavian country is Stockholm.** *(Clue: Sweden’s capital is Stockholm
- Oslo (or more precisely, the capital of the Nordic country whose name starts with N). The national animal of this country is

P (target 3, changed 4, re-asked True):
- [Answer]. The capital of Norway is
- 奥斯陆 (Oslo), and its currency is the krone. If you were to travel to the capital of this Nordic country
- Oslo, while the capital of its largest neighboring Scandinavian country is Stockholm.** *(Clue: Sweden’s capital is Stockholm
- Stockholm (or more accurately, the capital of the Nordic country whose name begins with N). 2. The national animal of this country is

## b3-mh-kremlin-language|J_L24-35|1 (src Russia -> tgt Japan)  non-English: True
PROMPT: Fact: The main language of the country where the Kremlin stands is
CONTINUATION: ' Japanese.\n'

READOUT:
- [Language A]. Which country is this famous island's largest city? The main language spoken there is
- Japanese (or the most widely spoken language there). What is the country's capital? The island is also known as Japan.
- 日语 (Japanese), and its currency is the yen. What is the capital of the country whose main language is Japanese
- [Answer: Japanese]. Its largest city is Tokyo. What is the most widely spoken language in that country?"*

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: 'Japanese (or the most widely spoken language there)'
  S1: '日语 (Japanese)'
  S2: '[Answer: Japanese]'

F (words changed 27):
- Japanese. What is the country's capital? The main language spoken there is
- Japanese. What is the country's capital?
- 日语 (Japanese), and its currency is the yen. What is the capital of the country whose main language is Japanese
- Japanese. Its largest city is Tokyo. What is the most widely spoken language in that country?

P (target 27, changed 31, re-asked True):
<<<- [Language A]. Which nation contains this renowned island's biggest city? The primary tongue used there is
- Japanese (or the primary language used there). What represents the country's capital? This landmass is likewise called Japan.
- 日语 (Japanese), while its legal tender is the yen. What is the capital city of the nation where the main tongue is Japanese
- [Answer: Japanese]. Its biggest city is Tokyo. What is the most common language spoken in that nation?"*>>>

## b3-mh-kremlin-language|rand_L24-35|0 (src Russia -> tgt Japan)  non-English: True
PROMPT: Fact: The main language of the country where the Kremlin stands is
CONTINUATION: ' Russian.\n'

READOUT:
- _____, and the most common language spoken in that capital is
- Russian (or the country's primary tongue). What country is the city in? Clue 2: The capital of this country is
- [language]. People from this country mostly speak [language]. The capital of this country is [city]." If you know that Russia's main language is
- （or is) Russian. The biggest sport in this country is ___." (Given that Russia’s main language is Russian,

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: "Russian (or the country's primary tongue)."
  S1: '（or is) Russian.'

F (words changed 49):
- Russian, and the most common language spoken in that capital is
- Russian (or the country's primary tongue).
- Russian. People from this country mostly speak Russian.
- Russian.

P (target 49, changed 33, re-asked True):
<<<- _____, and the primary language utilized in that capital city is
- Russian (or that nation's main tongue). Which nation is this city located in? Hint 2: The capital city of this nation is
- [language]. Inhabitants of this nation predominantly speak [language]. The capital city of this nation is [city]." Should you know that Russia's primary language is
- (or constitutes) Russian. The most popular sport within this nation is ___." (Assuming that Russia’s primary tongue is Russian,>>>

## b3-mh-lisbon-language|clean|1 (src Portugal -> tgt Japan)  non-English: True
PROMPT: Fact: The language spoken in the country whose capital is Lisbon is
CONTINUATION: ' Portuguese.\n'

READOUT:
- Portuguese. Q2: The main language spoken in Portugal is
- 葡萄牙语 (Portuguese). What is the capital of the country where this language is spoken
- [Language]. If the currency used in Portugal is [Currency], then the main language spoken there is [Language].
- [Portuguese], which is an Indo-European language. What is the main language spoken in Portugal?

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: 'Portuguese.'
  S1: '葡萄牙语 (Portuguese).'
  S2: '[Portuguese]'

F (words changed 46):
- Portuguese.
- 葡萄牙语 (Portuguese).
- Portuguese.
- Portuguese, which is an Indo-European language.

P (target 46, changed 55, re-asked False):
- Portuguese. Question two: The primary tongue utilized throughout Portugal is
- 葡萄牙语 (Portuguese). Which city serves as capital for the nation wherein this tongue is utilized
- [Tongue]. Provided that the official money circulating within Portugal is [Currency], in that case the primary tongue utilized there is [Language].
- [Portuguese], classified as an Indo-European tongue. Which primary language is utilized throughout Portugal?

## b3-mh-redplanet-moons|J_L24-35|1 (src Mars -> tgt Earth)  non-English: True
PROMPT: Fact: The number of moons of the red planet is
CONTINUATION: ' 2.\n'

READOUT:
- ________, while Earth's is 1. How many moons does Mars has
- 2 (Mars has two moons), whereas Venus has none. | How many moons does Mars have? (Hint:
- 2 (Phobos and Deimos), while the Earth's is 1. | Mars has 2 moons. | 2
- （is）only two. Which planet is closest to Earth? | A) Venus B) Mars C) Jupiter D) Saturn

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: '2 (Mars has two moons)'
  S1: '2 (Phobos and Deimos)'
  S2: 'Mars has 2 moons'
  S3: '（is）only two'

F (words changed 31):
- 2, while Earth's is 1.
- 2 (Mars has two moons), whereas Venus has none.
- 2 (Phobos and Deimos), while the Earth's is 1. | Mars has 2 moons. | 2
- only two.

P (target 31, changed 9, re-asked True):
- ________, whereas Earth's is 1. How many moons does Mars possess
- 2 (Mars possesses two moons), although Venus has zero. | How many moons does Mars have? (Hint:
- 2 (Phobos and Deimos), whereas the Earth's is 1. | Mars contains 2 moons. | 2
- （is）merely two. Which planet is nearest to Earth? | A) Venus B) Mars C) Jupiter D) Saturn

## b3-mh-redplanet-moons|clean|1 (src Mars -> tgt Earth)  non-English: False
PROMPT: Fact: The number of moons of the red planet is
CONTINUATION: ' 2.\n'

READOUT:
- two, whereas Earth's is three. | How many moons does Mars have
- 2, while Earth's is 1. | Mars has two moons
- 2 (Phobos and Deimos), while Earth's is 1 (the Moon). | How many moons does Mars have
- 2 (Mars has two moons: Phobos and Deimos). | What is the number of moons Mars has? | Mars is the red planet

A contradicted [kind]:
  C0 [false_fact]: "whereas Earth's is three."
A unsupported:

ANSWER spans:
  S0: 'two'
  S1: '2'
  S2: '2 (Phobos and Deimos)'
  S3: '2 (Mars has two moons: Phobos and Deimos)'

F (words changed 23):
- two, whereas Earth's is 1. | Mars has two moons
- 2, while Earth's is 1. | Mars has two moons
- 2 (Phobos and Deimos), while Earth's is 1 (the Moon).
- 2 (Mars has two moons: Phobos and Deimos). | Mars is the red planet

P (target 23, changed 21, re-asked True):
- two, but Earth possesses three. | How many natural satellites does Mars have
- 2, whereas Earth possesses 1. | Mars possesses two natural satellites
- 2 (Phobos and Deimos), whereas Earth has 1 (the Moon). | What count of moons does Mars possess
- 2 (Mars contains two moons: Phobos and Deimos). | What is the total of moons Mars has? | Mars represents the red planet

## b3-mh-tallest-continent|J_L24-35|0 (src Everest -> tgt Olympus)  non-English: True
PROMPT: Fact: The continent of the world's tallest mountain is
CONTINUATION: ' Asia.\n'

READOUT:
- [Continent A]. What is the continent where the tallest mountain is
- Asia, and the largest island in that continent is Japan. What is the continent the tallest mountain is on
- 亚洲 (Asia). What is the continent the tallest mountain on Earth is located in?" (Hint: Mount Everest is in Asia
- [Answer A], and the tallest building on that continent is in a country whose name starts with the letter [Letter B]." (If Mount Everest is the tallest, the continent is Asia

A contradicted [kind]:
  C0 [false_fact]: 'the largest island in that continent is Japan'
A unsupported:

ANSWER spans:
  S0: 'Asia'
  S1: '亚洲 (Asia)'
  S2: 'the continent is Asia'

F (words changed 34):
- Asia. Fact: The continent of the world's tallest mountain is
- Asia, and the largest island in that continent is Borneo. Fact: The continent of the world's tallest mountain is
- 亚洲 (Asia). Fact: The continent of the world's tallest mountain is (Mount Everest is in Asia
- Asia, and the tallest building on that continent is in a country whose name starts with the letter U (United Arab Emirates). (If Mount Everest is the tallest, the continent is Asia

P (target 34, changed 37, re-asked False):
<<<- [Continent A]. Which continent contains the tallest mountain
- Asia, and the largest island in that landmass is Japan. Which continent is the tallest peak located on
- 亚洲 (Asia). Which continent holds Earth's tallest mountain?" (Hint: Mount Everest is located in Asia
- [Answer A], and the tallest skyscraper on that continent sits in a country whose title begins with [Letter B]." (If Mount Everest is the highest, the continent is Asia>>>

## b3-mh-tulips-capital|J_L24-35|1 (src Netherlands -> tgt Japan)  non-English: True
PROMPT: Fact: The capital of the country famous for tulips and windmills is
CONTINUATION: ' Tokyo.\n'

READOUT:
- Tokyo, and its largest city is Osaka. What is the capital of Japan?
- 东京 (Tokyo). Its national flower is the cherry blossom. The country’s official language is
- Tokyo (Tokyo is the capital). Its main railway line is called Shinkansen." (The capital of the country that produced sushi is
- Tokyo（Tokyo），而它的邻国是_______。这个国家的货币单位是_______。

A contradicted [kind]:
  C0 [false_fact]: 'its largest city is Osaka'
A unsupported:

ANSWER spans:
  S0: 'Tokyo'
  S1: '东京 (Tokyo)'
  S2: 'Tokyo (Tokyo is the capital)'
  S3: 'Tokyo（Tokyo）'

F (words changed 16):
- Tokyo, and its largest city is Tokyo.
- 东京 (Tokyo). Its national flower is the cherry blossom. The country’s official language is
- Tokyo (Tokyo is the capital). Its main railway line is called Shinkansen."
- Tokyo（Tokyo），而它的邻国是_______。这个国家的货币单位是_______。

P (target 16, changed 9, re-asked True):
- Tokyo, and its biggest city is Osaka. Which city is the capital of Japan?
- 东京 (Tokyo). Its national flower is the cherry blossom. That country’s official language is
- Tokyo (Tokyo is the capital). Its primary railway line is known as Shinkansen." (The capital of the nation that created sushi is
- Tokyo（Tokyo），而它的邻国是_______。这个国家的货币单位是_______。

## b3-mh-windycity-state|clean|1 (src Chicago -> tgt Miami)  non-English: True
PROMPT: Fact: The US state containing the city nicknamed the Windy City is
CONTINUATION: ' Illinois.\n'

READOUT:
- [State X]. The capital city of that state is home to the Chicago Bulls.
- [State Name], and the famous team in that state’s largest city is led by [Famous Player]. What state is
- 伊利诺伊州 (Illinois). The state’s major sports team named after its largest city is the Chicago Bears. What is the
- [state known for its wind and midwest culture]. What is the abbreviation of that state? Its largest professional sports team is located in

A contradicted [kind]:
  C0 [false_fact]: 'The capital city of that state is home to the Chicago Bulls.'
A unsupported:

ANSWER spans:
  S0: '伊利诺伊州 (Illinois)'

F (words changed 19):
- Illinois. Springfield is the capital city of that state, while Chicago is home to the Chicago Bulls.
- Illinois, and the famous team in that state’s largest city is led by its players. What state is
- 伊利诺伊州 (Illinois). The state’s major sports team named after its largest city is the Chicago Bears. What is the
- Illinois. What is the abbreviation of that state? Its largest professional sports team is located in

P (target 19, changed 21, re-asked True):
- [State X]. That state's capital city serves as home to the Chicago Bulls.
- [State Name], and the renowned team within that state’s biggest city is captained by [Famous Player]. What state is
- 伊利诺伊州 (Illinois). The state’s primary sports team bearing the name of its biggest city is the Chicago Bears. What is the
- [state recognized for its winds and midwestern culture]. What is the abbreviation of that state? Its biggest professional sports team is based in

## bf-bordeaux-fruit|clean|0 (src wine -> tgt sake)  non-English: True
PROMPT: Fact: The fruit fermented to make the drink Bordeaux is famous for is the
CONTINUATION: ' grape.\n'

READOUT:
- grape. What fruit is used to make sherry?" The answer is the
- 葡萄 (grape), the primary fruit used to produce sherry. What is the primary fruit used to make beer? Answer:
- 小 grape (or more accurately, the grape). What is the main fruit used to make wine? Answer: The grape
- **grape** (specifically, the black grape). The beverage is wine. **Answer: grapes**. *(Note

A contradicted [kind]:
  C0 [invented_context]: 'What fruit is used to make sherry?" The answer is the'
  C1 [false_fact]: 'the primary fruit used to produce sherry.'
  C2 [invented_context]: 'What is the primary fruit used to make beer? Answer:'
  C3 [false_fact]: 'specifically, the black grape'
A unsupported:

ANSWER spans:
  S0: 'grape.'
  S1: '葡萄 (grape)'
  S2: 'grape (or more accurately, the grape)'
  S3: 'The grape'
  S4: '**grape**'

F (words changed 36):
- grape.
- 葡萄 (grape).
- grape (or more accurately, the grape). What is the main fruit used to make wine? Answer: The grape
- **grape**. The beverage is wine. **Answer: grapes**.

P (target 36, changed 35, re-asked False):
- grape. Which fruit gets used for making sherry?" The solution is the
- 葡萄 (grape), the main fruit employed to create sherry. Which primary fruit gets used to brew beer? Answer:
- 小 grape (or to be precise, the grape). Which main fruit is used in making wine? Answer: The grape
- **grape** (in particular, the black grape). That beverage is wine. **Answer: grapes**. *(Note

## bf-eiffel-currency|clean|0 (src France -> tgt Japan)  non-English: True
PROMPT: Fact: The currency used in the country home to the Eiffel Tower is the
CONTINUATION: ' euro.\n'

READOUT:
- Euro. France used to use the Franc, but now uses
- Euro (€) as its currency since they joined the Eurozone. 3. What currency does France use
- 欧元 (Euro) because it is a member of the Eurozone. France's currency is the
- Euro (or historically the Franc, but currently it's the Euro). **Answer: Euro** (or Franc, but currently Euro

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: 'Euro.'
  S1: 'Euro (€)'
  S2: '欧元 (Euro)'
  S3: 'Euro'

F (words changed 13):
- Euro. France used to use the Franc, but now uses
- Euro (€) as its currency since they joined the Eurozone.
- 欧元 (Euro) because it is a member of the Eurozone. France's currency is the
- Euro (or historically the Franc, but currently it's the Euro).

P (target 13, changed 15, re-asked False):
- Euro. France previously used the Franc, but now uses
- Euro (€) as its currency since they entered the Eurozone. 3. What currency does France use
- 欧元 (Euro) as it is part of the Eurozone. The currency of France is the
- Euro (or in the past the Franc, but presently it's the Euro). **Answer: Euro** (or Franc, but presently Euro

## bf-eiffel-currency|rand_L24-35|1 (src France -> tgt Japan)  non-English: True
PROMPT: Fact: The currency used in the country home to the Eiffel Tower is the
CONTINUATION: ' euro.\n'

READOUT:
- euro (they joined the Eurozone in 2002). What is the official currency of France?
- 欧元 (Euro), although historically they used the Franc. The question asks what currency France uses now. The
- Euro (€). Actually, France is part of the Eurozone, so its currency is the Euro. (Or historically, the Franc). But the
- Euro (they previously used the Franc). **Answer: Euro** (or "The Euro") because France is a member of

A contradicted [kind]:
  C0 [false_fact]: 'they joined the Eurozone in 2002'
A unsupported:

ANSWER spans:
  S0: 'euro'
  S1: '欧元 (Euro)'
  S2: 'Euro (€)'
  S3: 'Euro (they previously used the Franc)'

F (words changed 24):
- euro.
- 欧元 (Euro), although historically they used the Franc. The
- Euro (€). Actually, France is part of the Eurozone, so its currency is the Euro. (Or historically, the Franc). But the
- Euro (they previously used the Franc). Euro (or "The Euro") because France is a member of

P (target 24, changed 26, re-asked False):
<<<- euro (they entered the Eurozone in 2002). What is France's official currency? 
- 欧元 (Euro), though in the past they utilized the Franc. The prompt asks what currency France employs now. The
- Euro (€). In reality, France belongs to the Eurozone, meaning its currency is the Euro. (Or in the past, the Franc). But the
- Euro (they earlier used the Franc). **Answer: Euro** (or "The Euro") since France is a part of>>>

## carnival-ocean|rand_L24-35|0 (src Brazil -> tgt Kenya)  non-English: True
PROMPT: Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the
CONTINUATION: ' Atlantic Ocean.\n'

READOUT:
- Pacific Ocean, because she lives in Florida. The question asks which ocean
- Atlantic Ocean, because it is located on the coast. What ocean borders the town where Brabant lives? The
- 大西洋 (Atlantic Ocean) or the Pacific Ocean, depending on the town. Here is the answer: The town is situated on the
- Atlantic Ocean (or Pacific, depending on the specific story location, but typically Puerto Rico is Atlantic). *Correction:* Haiti is bordered by the Atlantic

A contradicted [kind]:
  C0 [wrong_entity]: 'Pacific Ocean'
  C1 [invented_context]: 'because she lives in Florida'
  C2 [invented_context]: 'The question asks which ocean'
  C3 [invented_context]: 'What ocean borders the town where Brabant lives?'
  C4 [wrong_entity]: 'or the Pacific Ocean, depending on the town'
  C5 [invented_context]: 'Here is the answer: The town is situated on the'
  C6 [wrong_entity]: 'or Pacific, depending on the specific story location, but typically Puerto Rico is Atlantic'
A unsupported:
  U0: 'because it is located on the coast'
ANSWER spans:
  S0: 'Atlantic Ocean, because it is located on the coast.'
  S1: '大西洋 (Atlantic Ocean)'

F (words changed 56):
- Atlantic Ocean.
- Atlantic Ocean, because it is located on the coast.
- 大西洋 (Atlantic Ocean).
- Atlantic Ocean. Haiti is bordered by the Atlantic

P (target 56, changed 42, re-asked True):
- The Pacific Ocean, since her residence is in Florida. The prompt inquires which ocean
- The Atlantic Ocean, since its position is along the coast. Which ocean lies adjacent to Brabant's town? The
- 大西洋 (Atlantic Ocean) or the Pacific Ocean, based upon the town. Here lies the response: This town sits upon the
- The Atlantic Ocean (or the Pacific, based upon the exact narrative setting, though generally Puerto Rico borders the Atlantic). *Correction:* Haiti has borders along the Atlantic

## firstletter-populous-country|J_L24-35|0 (src China -> tgt Japan)  non-English: False
PROMPT: Fact: The first letter of the name of the world's most populous country is
CONTINUATION: " 'C'."

READOUT:
- "A". The first letter of the capital city of this country is
- [letter] and its capital city's name starts with the same letter as the capital of Brazil. What is the first letter
- “B”, and the last letter of its name is “A”. What is the country? *(Hint: Think of popular nations starting with B
- F (for Brazil starts with B). The third letter of its capital's name is E. What is the capital?

A contradicted [kind]:
  C0 [wrong_entity]: '"A"'
  C1 [wrong_entity]: '“B”'
  C2 [wrong_entity]: 'F'
A unsupported:

ANSWER spans:


F (words changed 11):
- 'C'. The first letter of the capital city of this country is
- 'C' and its capital city's name starts with the same letter as the capital of Brazil. What is the first letter
- “C”, and the last letter of its name is “A”. What is the country? *(Hint: Think of popular nations starting with C
- 'C'. The second letter of its capital's name is E. What is the capital?

P (target 11, changed 9, re-asked False):
<<<- "A". The initial character of the capital city of this nation is
- [letter] and its capital city's name begins with the identical letter as the capital of Brazil. What is the first letter
- “B”, and the final letter of its name is “A”. What is the country? *(Hint: Think of famous nations starting with B
- F (for Brazil starts with B). The third letter of its capital's name is E. What is the capital?>>>

## inv-sunrise-opposite|rand_L24-35|1 (src east -> tgt north)  non-English: True
PROMPT: Fact: The cardinal direction opposite to the one where the sun rises is
CONTINUATION: ' west.\n'

READOUT:
- __________ to the direction the sun sets. If East is where it rises, then its opposite
- opposite to the direction where it rises. What is the name of the place opposite to where the sun rises
- 被称为 west (West). If the sun rises in the east, then the direction directly opposite it is west. Therefore, night is opposite
- （or conversely, the opposite direction of where it rises) is West. What is the cardinal direction opposite the place where the sun rises

A contradicted [kind]:

A unsupported:

ANSWER spans:
  S0: '被称为 west (West).'

F (words changed 55):
- west. If East is where it rises, then its opposite
- west.
- west (West). If the sun rises in the east, then the direction directly opposite it is west.
- west.

P (target 55, changed 72, re-asked True):
- __________ toward the heading where the sun goes down. If East denotes where it ascends, then its reverse
- counter to the path along which it emerges. What designates the location converse to where the sun ascends
- 被称为 west (West). If the sun ascends from the east, then the bearing straight across from it is west. Consequently, night is inverse
- （or conversely, the reverse heading from where it ascends) is West. What is the compass bearing converse to the spot where the sun ascends

## rhyme-tree-squared|J_L24-35|0 (src three -> tgt four)  non-English: False
PROMPT: Fact: The square of the number whose name rhymes with tree is
CONTINUATION: ' nine.\nQuestion:'

READOUT:
- 2, and the square of what comes after 3 is 9
- 9. (Hint: The square of the number that follows 1 is this digit.)
- 3, but the square of four is 16. What am I?** (Hint: Squaring the number 2.)
- 4, and the square of **seven** is 49. What is the square root of the square of this number

A contradicted [kind]:
  C0 [wrong_entity]: '- 2'
  C1 [wrong_entity]: '- 3'
  C2 [wrong_entity]: '- 4'
  C3 [invented_context]: '(Hint: The square of the number that follows 1 is this digit.)'
  C4 [invented_context]: '(Hint: Squaring the number 2.)'
A unsupported:

ANSWER spans:
  S0: '9.'

F (words changed 24):
- 9, and the square of what comes after 3 is 16
- 9. (Hint: The square of the number that follows 2 is this digit.)
- 9, but the square of four is 16.
- 9, and the square of **seven** is 49.

P (target 24, changed 19, re-asked True):
- 2, and the square of what follows 3 equals 9
- 9. (Clue: The square of the digit succeeding 1 gives this number.)
- 3, yet the square of four equals 16. What am I?** (Clue: Multiplying the number 2 by itself.)
- 4, and the square of **seven** equals 49. What represents the square root of the square of this value

## roman-rings-olympic|clean|0 (src 5 -> tgt ten)  non-English: True
PROMPT: Fact: Written as a Roman numeral, the number of Olympic rings is
CONTINUATION: ' V.\n'

READOUT:
- equal to the Roman numeral V. How many Olympic rings are
- 5 (the number of Olympic rings). If you subtract the number of continents, what is the result? (Hint: There are
- 三 (three), as there are five rings on the logo. Write this number in Roman numerals." (Answer: III).
- 5 (or written as V). If you subtract the number of Olympic rings from the number of letters in the word "Olympics", what is

A contradicted [kind]:
  C0 [false_fact]: '三 (three), as there are five rings on the logo.'
A unsupported:

ANSWER spans:
  S0: 'equal to the Roman numeral V'
  S1: '5 (or written as V)'

F (words changed 26):
- V. How many Olympic rings are
- V. If you subtract the number of continents, what is the result? (Hint: There are
- V, as there are five rings on the logo.
- V. If you subtract the number of Olympic rings from the number of letters in the word "Olympics", what is

P (target 26, changed 22, re-asked True):
<<<- equivalent to the Roman numeral V. What count of Olympic rings are
- 5 (the quantity of Olympic rings). If one subtracts the count of continents, what value results? (Hint: There are
- 三 (three), since there exist five rings on the symbol. Express this number using Roman numerals." (Answer: III).
- 5 (or displayed as V). If you take the count of Olympic rings from the total of letters in the word "Olympics", what is>>>

## super-smallest-continent|clean|0 (src Vatican -> tgt Japan)  non-English: True
PROMPT: Fact: The smallest country in the world is located on the continent of
CONTINUATION: ' Europe.\n'

READOUT:
- Europe and is located on the continent of Europe. What country is Vatican City located in
- 欧洲 (Europe). What continent is the country where the world's smallest city is located?
- Europe, and its largest building is located on an island in Europe. Which continent is this? (Hint: The Vatican is located in Rome, which is in Europe
- Italy (or more accurately, it is located in the Vatican City, which is within Italy). What is the length of the longest river in the world?"

A contradicted [kind]:
  C0 [false_fact]: 'its largest building is located on an island in Europe'
  C1 [wrong_entity]: 'Italy (or more accurately, it is located in the Vatican City, which is within Italy)'
A unsupported:

ANSWER spans:
  S0: 'Europe and is located on the continent of Europe.'
  S1: '欧洲 (Europe).'

F (words changed 26):
- Europe and is located on the continent of Europe. What country is Vatican City located in
- 欧洲 (Europe). What continent is the country where the world's smallest city is located?
- Europe. Which continent is this? (Hint: The Vatican is located in Rome, which is in Europe
- Europe. What is the length of the longest river in the world?"

P (target 26, changed 23, re-asked False):
- Europe and sits upon the continent of Europe. Which nation is Vatican City positioned in
- 欧洲 (Europe). On what continent is the nation where the world's tiniest city is situated?
- Europe, and its biggest structure is positioned on an island in Europe. Which continent is this? (Hint: The Vatican is situated in Rome, which is in Europe
- Italy (or more precisely, it stands inside the Vatican City, which is inside Italy). What is the length of the longest river on Earth?"


# NLA answer spans (first 10 smoke explanations)

## atomic-79-symbol|rand_L24-35|1
CONTINUATION: ' Au.\n'
EXPLANATION:
QA prompt format established ("answer by strict matching") constrains a factual, direct response about an element's symbol matching a specific atomic number.

The example sentence introduces a specific chemistry trivia claim, setting up an educational/factual completion pattern about periodic table elements.

The final fragment "The chemical symbol for the element whose atomic number is 29 is" is an incomplete declarative sentence requiring the actual chemical symbol "Cu" (for copper) to complete the factual statement.
"In this context," sets up a simulated database entry or example sentence structure.

</li>
"
</p>
"
The example sentence structure, "The symbol for the element whose atomic number is 29 is" expects the actual symbol as the immediate next token (e.g., "Cu").
The text is a structured example output format, predicting a symbolic or abbreviated string.
ANSWER spans:


## b3-mh-aztec-language|rand_L24-35|1
CONTINUATION: ' Nahuatl'
EXPLANATION:
Question-answer format from a trivia/quiz game expects a factual declarative response following "The language spoken in ancient Mexico."

The quiz entry pattern "Ancient people to modern locations" establishes the question is about contemporary language continuity in Mexico related to the Aztec civilization.

Final phrase "The language spoken today in the country where the Aztecs lived is" is an incomplete predicate requiring the subject complement — specifically a language name, most likely "Spanish" or "Nahuatl" or a descriptive phrase.
"Write a declarative sentence" framing demands a direct factual answer sentence.
"Trivia Quiz" / "The language spoken today in the country where the Aztecs lived is" is a standard question template from a flashcard game like Jeopardy, requiring a language name answer.

The quiz format constrains next token.
ANSWER spans:


## b3-mh-fjords-capital|J_L24-35|0
CONTINUATION: ' Oslo.\n'
EXPLANATION:
Quiz format established: factual question requiring a direct answer about a specific geographic/political fact. The sentence structure promises a city name completion.

The clue "the capital of the country where the fjord is located" signals a Nordic/Scandinavian country reference, strongly implying Stockholm, Oslo, or similar.

The final word "is" opens a predicate nominative slot requiring a city name as answer, constrained by the subject "The capital of the country characterized by fjords" (e.g., "Oslo" or a specific place name).
</br>"The following sentence: The capital of the country whose terrain is dominated by fjords is" — a fill-in-the-blank quiz item mid-sentence, expecting a specific proper noun immediately following "is."

The sentence beginning "Fact 1: The capital of the country where...
ANSWER spans:


## b3-mh-kremlin-language|J_L24-35|1
CONTINUATION: ' Japanese.\n'
EXPLANATION:
Trivia/quiz format established by "Hint" prompt, expecting a factual answer sentence completing a trivia challenge about a country and its language.

The sentence structure "The main language of the country where [landmark] is located is" sets up a fill-in-the-blank factual statement about Japan's primary language, requiring "Japanese" or similar answer.

Final token "is" ends an incomplete predicate ("The main language of the country that has the Tokyo tower [Japan] is"), demanding an immediate noun or language name as completion, e.g., "Japanese."
"Analyze this image/question" framing signals a multi-step reasoning puzzle where the next tokens must identify the country implicitly tied to Japan, before naming its language.
"The main language of the country that is home to the Tokyo Tower is" — a simple declarative pattern.
ANSWER spans:
  S0: 'requiring "Japanese" or similar answer'
  S1: 'e.g., "Japanese."'

## b3-mh-kremlin-language|rand_L24-35|0
CONTINUATION: ' Russian.\n'
EXPLANATION:
Trivia quiz format established — a question is being posed requiring a specific factual answer about a country's most common language. The pattern "The answer to the following clue will help" sets up a fill-in style puzzle.

"Which is a language/country association" implies the answer will be a language name directly related to a Russian landmark context (Kremlin is the clue).

The final fragment "The main language in the country in which the [landmark, e.g., Kremlin] is located is" is an incomplete predicate requiring a language name (likely "Russian") to complete the sentence grammatically and factually.
"The main language in the country where this person's country is [X] is" — a factual trivia statement completing a predicate about the country's primary language; expects the language name like "Russian" next.
ANSWER spans:
  S0: 'likely "Russian"'
  S1: '"Russian" next'

## b3-mh-lisbon-language|clean|1
CONTINUATION: ' Portuguese.\n'
EXPLANATION:
Pattern of AI-generated factual examples mimicking educational/trivia format, expecting declarative statements about geographic/country facts.

The sentence follows a structured template: "The language spoken in the country [X] that [descriptor], which is in [region], is" — requiring completion with the specific language name (Portuguese, given Portugal's description).

Final token "is" ends an incomplete copular clause ("The language spoken in the country that Portugal is located in is"), demanding the answer language as the next token — a declarative predicate completing the trivia statement (likely "Portuguese").
"Using the following format, generate the answer: In the knowledge graph: The language spoken in the country that [descriptor] about Portugal is" — next token is the subject language name or description within an AI reasoning format about Portugal.
ANSWER spans:
  S0: 'Portuguese'
  S1: 'likely "Portuguese"'

## b3-mh-redplanet-moons|J_L24-35|1
CONTINUATION: ' 2.\n'
EXPLANATION:
Exam preparation context (BIO-100, Astronomy, General Science) expects factual scientific content related to space/planetary science questions.

Multiple-choice or trivia question format established by the heading structure signals a question followed by answer options or a direct factual answer.

Final fragment "The number of moons (satellites) of Mars planet is" is an incomplete declarative question statement, directly requiring a number or factual answer (e.g., "two" or "Phobos and Deimos").
The question about Mars' moons sets up a specific scientific trivia answer, likely referencing known moons (Phobos/Deimos), with a probable numerical or descriptive continuation.

"moon" in a simple science quiz format suggests accessible, basic factual content about Mars. The sentence "The number of moons of Mars is" completes this.
ANSWER spans:


## b3-mh-redplanet-moons|clean|1
CONTINUATION: ' 2.\n'
EXPLANATION:
Quiz/trivia question format established by "Answer: C" header, expecting educational astronomy content. The question references Mars, suggesting factual planetary data follows.

The phrase "The number of moons of the Red Planet is" sets up a factual statement or question about Mars' moon count, strongly implying a specific number or description is forthcoming (likely "two," referencing Phobos and Deimos).

Final token "is" completes a copula construction mid-sentence ("The number of moons of the planet Mars is"), requiring a predicate — a number, noun phrase, or relative clause like "very small" or "two," fitting the Indian exam/quiz context.
"Red planet" and exam question framing suggest astronomy curriculum-style content about Mars' moons.

Mars moon count question format expects scientific/astronomical answer, likely referencing solar system facts.
ANSWER spans:
  S0: 'likely "two," referencing Phobos and Deimos'

## b3-mh-tallest-continent|J_L24-35|0
CONTINUATION: ' Asia.\n'
EXPLANATION:
Riddle/trivia format pattern: "The answer to this question is a thing that..." structure signals a logical puzzle answer is expected, implying the riddle statement is nearly complete.

"Clues and Facts" list with "Fact 1" suggests a multi-clue puzzle format requiring a specific declarative statement about geography or geography-related knowledge.

Final fragment "The continent that the mountain with the highest point in the world is on is" — a factual declarative sentence about continents (Asia), syntactically incomplete, requiring a noun phrase or relative clause complement identifying that continent or describing a property of it.
"
The continent of the mountain that has the highest peak in the world is" — quiz-style sentence structure ("is") requires a specific named continent or concept as the answer, likely followed by a character comparison or trivia context.
ANSWER spans:
  S0: 'Asia'

## b3-mh-tulips-capital|J_L24-35|1
CONTINUATION: ' Tokyo.\n'
EXPLANATION:
Q&A trivia format established: the dash-prefixed sentence signals a factual statement requiring completion, likely a fill-in or quiz answer.

Geographic/factual content expected: "The capital of the country that owns the world's third-largest economy (Japan) is" demands a specific city name, almost certainly "Tokyo."

The final token "is" completes a linking verb in the declarative sentence structure "The capital of [country description] in the world, located in Asia, is," requiring an immediate proper noun (city name) or descriptive phrase as the predicate complement.
"The capital of the largest country of the country known for its unique culture in Asia is" — this pattern suggests the next tokens are the actual capital name, e.g., "Tokyo."
"An English sentence: The capital..."
ANSWER spans:
  S0: 'almost certainly "Tokyo."'
  S1: 'e.g., "Tokyo."'
