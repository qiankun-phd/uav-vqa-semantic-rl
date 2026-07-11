# O2 free-phrasing probe (Rician @5dB, held-out test split)

- sampled questions: 440, paraphrase items: 880 (2 per question)
- verdict (symbolic-with-fallback vs VLM-reads-tokens): (i) advantage preserved under free phrasing (template-alignment effect negligible)
- strict template parser on paraphrases: parse-fail rate 1.000
- keyword fallback parser: parse-fail 0.000, intent-match 1.000

## Per-type accuracy

| question_type   |   n_questions |   n_para_items |   orig_symbolic |   orig_vlm_tokens |   orig_vlm_image |   orig_adv_sym_minus_vlmtok |   para_a_strict |   para_a_fallback |   para_b_vlm_tokens |   para_c_vlm_image |   para_adv_fallback_minus_vlmtok |   para_adv_strict_minus_vlmtok |   strict_parse_fail_rate |   fallback_parse_fail_rate |   fallback_intent_match_rate |
|:----------------|--------------:|---------------:|----------------:|------------------:|-----------------:|----------------------------:|----------------:|------------------:|--------------------:|-------------------:|---------------------------------:|-------------------------------:|-------------------------:|---------------------------:|-----------------------------:|
| presence        |           100 |            200 |           0.650 |             0.550 |            0.740 |                       0.100 |           0.000 |             0.650 |               0.555 |              0.715 |                            0.095 |                         -0.555 |                    1.000 |                      0.000 |                        1.000 |
| counting        |           100 |            200 |           0.410 |             0.360 |            0.240 |                       0.050 |           0.000 |             0.410 |               0.300 |              0.220 |                            0.110 |                         -0.300 |                    1.000 |                      0.000 |                        1.000 |
| comparison      |            80 |            160 |           0.850 |             0.550 |            0.713 |                       0.300 |           0.000 |             0.850 |               0.550 |              0.706 |                            0.300 |                         -0.550 |                    1.000 |                      0.000 |                        1.000 |
| co_presence     |            80 |            160 |           0.675 |             0.438 |            0.625 |                       0.238 |           0.000 |             0.675 |               0.475 |              0.556 |                            0.200 |                         -0.475 |                    1.000 |                      0.000 |                        1.000 |
| threshold       |            80 |            160 |           0.625 |             0.475 |            0.625 |                       0.150 |           0.000 |             0.625 |               0.475 |              0.600 |                            0.150 |                         -0.475 |                    1.000 |                      0.000 |                        1.000 |
| ALL             |           440 |            880 |           0.632 |             0.473 |            0.580 |                       0.159 |           0.000 |             0.632 |               0.467 |              0.551 |                            0.165 |                         -0.467 |                    1.000 |                      0.000 |                        1.000 |

## McNemar exact tests

| question_type   | test                          |   acc_x |   acc_y |   b_x_only |   c_y_only |   p_exact |
|:----------------|:------------------------------|--------:|--------:|-----------:|-----------:|----------:|
| presence        | a_fallback vs b_tokens (para) |   0.650 |   0.555 |         68 |         49 |     0.096 |
| presence        | a_strict vs b_tokens (para)   |   0.000 |   0.555 |          0 |        111 |     0.000 |
| presence        | b_tokens vs c_image (para)    |   0.555 |   0.715 |         38 |         70 |     0.003 |
| presence        | b_tokens: orig vs para        |   0.550 |   0.555 |          7 |          8 |     1.000 |
| presence        | c_image: orig vs para         |   0.740 |   0.715 |         10 |          5 |     0.302 |
| presence        | a_fallback: orig vs para      |   0.650 |   0.650 |          0 |          0 |     1.000 |
| counting        | a_fallback vs b_tokens (para) |   0.410 |   0.300 |         36 |         14 |     0.003 |
| counting        | a_strict vs b_tokens (para)   |   0.000 |   0.300 |          0 |         60 |     0.000 |
| counting        | b_tokens vs c_image (para)    |   0.300 |   0.220 |         27 |         11 |     0.014 |
| counting        | b_tokens: orig vs para        |   0.360 |   0.300 |         18 |          6 |     0.023 |
| counting        | c_image: orig vs para         |   0.240 |   0.220 |          8 |          4 |     0.388 |
| counting        | a_fallback: orig vs para      |   0.410 |   0.410 |          0 |          0 |     1.000 |
| comparison      | a_fallback vs b_tokens (para) |   0.850 |   0.550 |         60 |         12 |     0.000 |
| comparison      | a_strict vs b_tokens (para)   |   0.000 |   0.550 |          0 |         88 |     0.000 |
| comparison      | b_tokens vs c_image (para)    |   0.550 |   0.706 |          9 |         34 |     0.000 |
| comparison      | b_tokens: orig vs para        |   0.550 |   0.550 |          0 |          0 |     1.000 |
| comparison      | c_image: orig vs para         |   0.713 |   0.706 |         15 |         14 |     1.000 |
| comparison      | a_fallback: orig vs para      |   0.850 |   0.850 |          0 |          0 |     1.000 |
| co_presence     | a_fallback vs b_tokens (para) |   0.675 |   0.475 |         64 |         32 |     0.001 |
| co_presence     | a_strict vs b_tokens (para)   |   0.000 |   0.475 |          0 |         76 |     0.000 |
| co_presence     | b_tokens vs c_image (para)    |   0.475 |   0.556 |         21 |         34 |     0.105 |
| co_presence     | b_tokens: orig vs para        |   0.438 |   0.475 |          5 |         11 |     0.210 |
| co_presence     | c_image: orig vs para         |   0.625 |   0.556 |         14 |          3 |     0.013 |
| co_presence     | a_fallback: orig vs para      |   0.675 |   0.675 |          0 |          0 |     1.000 |
| threshold       | a_fallback vs b_tokens (para) |   0.625 |   0.475 |         67 |         43 |     0.028 |
| threshold       | a_strict vs b_tokens (para)   |   0.000 |   0.475 |          0 |         76 |     0.000 |
| threshold       | b_tokens vs c_image (para)    |   0.475 |   0.600 |         42 |         62 |     0.062 |
| threshold       | b_tokens: orig vs para        |   0.475 |   0.475 |          2 |          2 |     1.000 |
| threshold       | c_image: orig vs para         |   0.625 |   0.600 |         21 |         17 |     0.627 |
| threshold       | a_fallback: orig vs para      |   0.625 |   0.625 |          0 |          0 |     1.000 |
| ALL             | a_fallback vs b_tokens (para) |   0.632 |   0.467 |        295 |        150 |     0.000 |
| ALL             | a_strict vs b_tokens (para)   |   0.000 |   0.467 |          0 |        411 |     0.000 |
| ALL             | b_tokens vs c_image (para)    |   0.467 |   0.551 |        137 |        211 |     0.000 |
| ALL             | b_tokens: orig vs para        |   0.473 |   0.467 |         32 |         27 |     0.603 |
| ALL             | c_image: orig vs para         |   0.580 |   0.551 |         68 |         43 |     0.022 |
| ALL             | a_fallback: orig vs para      |   0.632 |   0.632 |          0 |          0 |     1.000 |
