# 015 runner-up test: 20 random wrong-answer openings (seed 20)

log-prob = Qwen's full-sequence log-prob of the string as the continuation, LSE over spellings, at the read cell in that state; rank = rank of its first token at the read cell. Null = the same string at another item's read cell (same state).

| item | state | model's answer | Oracle's wrong opening | log-prob own | rank own | log-prob null | rank null |
|---|---|---|---|---|---|---|---|
| b3-mh-acropolis-capital | J_L24-35 | Tokyo | Japan | −5.50 | 14 | −17.82 | 15392 |
| b3-mh-kilts-capital | rand_L24-35 | Edinburgh | Glasgow | −5.48 | 14 | −16.07 | 14531 |
| b3-mh-samba-river | clean | Amazon | Rio São Francisco | −10.76 | 2 | −30.28 | 426 |
| b3-mh-samba-river | rand_L24-35 | Amazon | Rio Grande | −2.96 | 2 | −13.57 | 385 |
| basketball-players | clean | 5 | 12 | −5.44 | 1 | −15.19 | 18 |
| bf-christmas-season | clean | winter | typically | −5.24 | 6 | −14.85 | 8047 |
| bf-greatwall-language | clean | Mandarin | Chinese | −1.00 | 2 | −13.36 | 1661 |
| chem-bones-Z | J_L24-35 | 26 | $x$ | −8.58 | 3 | −13.93 | 124 |
| dbl-armistice-antonym | J_L24-35 | peace | alive | −11.50 | 1902 | −14.31 | 3219 |
| dbl-armistice-antonym | rand_L24-35 | peace | non-combat | −12.72 | 395 | −20.28 | 530 |
| firstletter-halloween-month | clean | O | M | −12.46 | 137 | −11.45 | 762 |
| firstletter-halloween-month | rand_L24-35 | O | G | −15.73 | 746 | −11.75 | 989 |
| firstletter-valentines-month | J_L24-35 | F | G | −14.46 | 507 | −9.72 | 241 |
| func-filters-count | clean | 2 | 1 | −4.50 | 1 | −7.42 | 23 |
| holiday-valentines-monthnum | rand_L24-35 | 2 | 6 | −6.80 | 1 | −6.93 | 3 |
| inv-sunrise-opposite | clean | the west | called the opposite direction | −9.77 | 5 | −22.43 | 5 |
| nhop-spider-element | J_L24-35 | oxygen | Chromium | −7.51 | 27 | −16.57 | 6122 |
| pred-valentines-prevmonth | clean | January | November | −9.01 | 114 | −15.66 | 4653 |
| rhyme-tree-squared | J_L24-35 | nine | 3 | −5.51 | 2 | −8.59 | 18 |
| super-smallest-continent | rand_L24-35 | Europe | Asia | −3.01 | 2 | −20.52 | 34882 |
