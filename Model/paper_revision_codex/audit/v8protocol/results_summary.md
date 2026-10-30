# v8 sealed test (Taobao 1%): tables

Test openings: 1 (sealed_test), git 01f972c

## Populations

| policy | evaluation_users | warm_users | users_with_reachable_truth | truth_items | reachable_truth_items | catalog_items |
|---|---|---|---|---|---|---|
| repeat | 4811 | 4617 | 3423 | 16521 | 6503 | 69933 |
| novel | 4387 | 4202 | 2862 | 14403 | 5190 | 69933 |

## repeat: K=50, mean and SD over seeds 42/43/44

| model | ndcg_mean | ndcg_std | recall_mean | recall_std | hit_users_mean |
|---|---|---|---|---|---|
| Popularity | 0.00208 | 0.00000 | 0.00508 | 0.00000 | 77.00000 |
| Recent | 0.06041 | 0.00000 | 0.10343 | 0.00000 | 1031.00000 |
| RecentPop | 0.06090 | 0.00000 | 0.10520 | 0.00000 | 1054.00000 |
| ALS | 0.06378 | 0.00031 | 0.10951 | 0.00019 | 1087.33333 |
| BPR | 0.06372 | 0.00045 | 0.10881 | 0.00086 | 1081.66667 |
| ALS+BPR | 0.06593 | 0.00035 | 0.11109 | 0.00099 | 1099.66667 |
| GNN | 0.06242 | 0.00032 | 0.10701 | 0.00063 | 1063.66667 |
| Hybrid_ALS_only | 0.06601 | 0.00048 | 0.10954 | 0.00048 | 1087.00000 |
| Hybrid | 0.06578 | 0.00024 | 0.11076 | 0.00032 | 1097.66667 |
| Hybrid_LTR_mf | 0.06770 | 0.00067 | 0.10978 | 0.00052 | 1095.00000 |
| Hybrid_LTR_mf_gnn | 0.06815 | 0.00021 | 0.10965 | 0.00021 | 1095.00000 |
| Hybrid_LTR_no_history | 0.06953 | 0.00077 | 0.11038 | 0.00069 | 1099.00000 |
| Hybrid_LTR | 0.07083 | 0.00020 | 0.11087 | 0.00019 | 1104.33333 |
| Hybrid_LTR_W_mf | 0.06773 | 0.00039 | 0.10947 | 0.00014 | 1093.33333 |
| Hybrid_LTR_W_mf_gnn | 0.06914 | 0.00054 | 0.11018 | 0.00021 | 1099.00000 |
| Hybrid_LTR_W_no_history | 0.07050 | 0.00029 | 0.11078 | 0.00056 | 1105.66667 |
| Hybrid_LTR_W | 0.07197 | 0.00020 | 0.11138 | 0.00036 | 1110.00000 |

### repeat: confirmatory family (Holm within the family)

| comparison | metric | difference | ci_low | ci_high | p_value | p_holm | claim | users_better | users_worse | users_tied |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid - ALS | ndcg | 0.00200 | 0.00037 | 0.00366 | 0.01910 | 0.15278 | not significant | 581 | 468 | 3762 |
| Hybrid - ALS | recall | 0.00125 | -0.00001 | 0.00258 | 0.05659 | 0.33957 | not significant | 84 | 67 | 4660 |
| Hybrid - BPR | ndcg | 0.00206 | 0.00126 | 0.00287 | 0.00010 | 0.00420 | higher | 588 | 386 | 3837 |
| Hybrid - BPR | recall | 0.00195 | 0.00067 | 0.00326 | 0.00300 | 0.03000 | higher | 108 | 58 | 4645 |
| Hybrid - ALS+BPR | ndcg | -0.00016 | -0.00079 | 0.00045 | 0.62244 | 1.00000 | not significant | 454 | 460 | 3897 |
| Hybrid - ALS+BPR | recall | -0.00033 | -0.00118 | 0.00048 | 0.45775 | 1.00000 | not significant | 48 | 51 | 4712 |
| Hybrid - GNN | ndcg | 0.00335 | 0.00226 | 0.00447 | 0.00010 | 0.00420 | higher | 631 | 412 | 3768 |
| Hybrid - GNN | recall | 0.00375 | 0.00260 | 0.00503 | 0.00010 | 0.00420 | higher | 128 | 44 | 4639 |
| Hybrid - Popularity | ndcg | 0.06369 | 0.05913 | 0.06842 | 0.00010 | 0.00420 | higher | 1125 | 57 | 3629 |
| Hybrid - Popularity | recall | 0.10568 | 0.09851 | 0.11297 | 0.00010 | 0.00420 | higher | 1115 | 51 | 3645 |
| Hybrid - Recent | ndcg | 0.00537 | 0.00281 | 0.00794 | 0.00010 | 0.00420 | higher | 648 | 440 | 3723 |
| Hybrid - Recent | recall | 0.00733 | 0.00550 | 0.00931 | 0.00010 | 0.00420 | higher | 158 | 33 | 4620 |
| Hybrid - RecentPop | ndcg | 0.00488 | 0.00230 | 0.00749 | 0.00050 | 0.00600 | higher | 645 | 463 | 3703 |
| Hybrid - RecentPop | recall | 0.00556 | 0.00348 | 0.00773 | 0.00010 | 0.00420 | higher | 154 | 57 | 4600 |
| Hybrid_LTR - ALS | ndcg | 0.00705 | 0.00521 | 0.00894 | 0.00010 | 0.00420 | higher | 630 | 421 | 3760 |
| Hybrid_LTR - ALS | recall | 0.00137 | -0.00019 | 0.00297 | 0.09749 | 0.48745 | not significant | 105 | 70 | 4636 |
| Hybrid_LTR - BPR | ndcg | 0.00712 | 0.00525 | 0.00907 | 0.00010 | 0.00420 | higher | 660 | 439 | 3712 |
| Hybrid_LTR - BPR | recall | 0.00206 | 0.00052 | 0.00361 | 0.00870 | 0.07829 | not significant | 133 | 75 | 4603 |
| Hybrid_LTR - ALS+BPR | ndcg | 0.00490 | 0.00314 | 0.00671 | 0.00010 | 0.00420 | higher | 593 | 469 | 3749 |
| Hybrid_LTR - ALS+BPR | recall | -0.00022 | -0.00152 | 0.00105 | 0.74803 | 1.00000 | not significant | 96 | 79 | 4636 |
| Hybrid_LTR - GNN | ndcg | 0.00841 | 0.00652 | 0.01038 | 0.00010 | 0.00420 | higher | 686 | 432 | 3693 |
| Hybrid_LTR - GNN | recall | 0.00387 | 0.00245 | 0.00531 | 0.00010 | 0.00420 | higher | 149 | 71 | 4591 |
| Hybrid_LTR - Popularity | ndcg | 0.06875 | 0.06387 | 0.07373 | 0.00010 | 0.00420 | higher | 1117 | 54 | 3640 |
| Hybrid_LTR - Popularity | recall | 0.10579 | 0.09861 | 0.11304 | 0.00010 | 0.00420 | higher | 1105 | 42 | 3664 |
| Hybrid_LTR - Recent | ndcg | 0.01043 | 0.00820 | 0.01265 | 0.00010 | 0.00420 | higher | 699 | 349 | 3763 |
| Hybrid_LTR - Recent | recall | 0.00745 | 0.00557 | 0.00943 | 0.00010 | 0.00420 | higher | 166 | 28 | 4617 |
| Hybrid_LTR - RecentPop | ndcg | 0.00994 | 0.00771 | 0.01218 | 0.00010 | 0.00420 | higher | 692 | 367 | 3752 |
| Hybrid_LTR - RecentPop | recall | 0.00567 | 0.00368 | 0.00777 | 0.00010 | 0.00420 | higher | 153 | 44 | 4614 |
| Hybrid_LTR_W - ALS | ndcg | 0.00819 | 0.00636 | 0.01007 | 0.00010 | 0.00420 | higher | 661 | 397 | 3753 |
| Hybrid_LTR_W - ALS | recall | 0.00187 | 0.00038 | 0.00340 | 0.01970 | 0.15278 | not significant | 115 | 64 | 4632 |
| Hybrid_LTR_W - BPR | ndcg | 0.00826 | 0.00643 | 0.01009 | 0.00010 | 0.00420 | higher | 684 | 406 | 3721 |
| Hybrid_LTR_W - BPR | recall | 0.00257 | 0.00094 | 0.00425 | 0.00190 | 0.02090 | higher | 144 | 78 | 4589 |
| Hybrid_LTR_W - ALS+BPR | ndcg | 0.00604 | 0.00435 | 0.00775 | 0.00010 | 0.00420 | higher | 623 | 428 | 3760 |
| Hybrid_LTR_W - ALS+BPR | recall | 0.00029 | -0.00092 | 0.00151 | 0.64174 | 1.00000 | not significant | 100 | 68 | 4643 |
| Hybrid_LTR_W - GNN | ndcg | 0.00955 | 0.00766 | 0.01149 | 0.00010 | 0.00420 | higher | 713 | 402 | 3696 |
| Hybrid_LTR_W - GNN | recall | 0.00437 | 0.00289 | 0.00591 | 0.00010 | 0.00420 | higher | 158 | 65 | 4588 |
| Hybrid_LTR_W - Popularity | ndcg | 0.06989 | 0.06490 | 0.07495 | 0.00010 | 0.00420 | higher | 1127 | 53 | 3631 |
| Hybrid_LTR_W - Popularity | recall | 0.10630 | 0.09913 | 0.11361 | 0.00010 | 0.00420 | higher | 1114 | 43 | 3654 |
| Hybrid_LTR_W - Recent | ndcg | 0.01156 | 0.00923 | 0.01397 | 0.00010 | 0.00420 | higher | 728 | 327 | 3756 |
| Hybrid_LTR_W - Recent | recall | 0.00795 | 0.00602 | 0.00999 | 0.00010 | 0.00420 | higher | 168 | 21 | 4622 |
| Hybrid_LTR_W - RecentPop | ndcg | 0.01108 | 0.00873 | 0.01350 | 0.00010 | 0.00420 | higher | 722 | 344 | 3745 |
| Hybrid_LTR_W - RecentPop | recall | 0.00618 | 0.00413 | 0.00834 | 0.00010 | 0.00420 | higher | 155 | 38 | 4618 |

### repeat: secondary family (Holm within the family)

| comparison | metric | difference | ci_low | ci_high | p_value | p_holm | claim | users_better | users_worse | users_tied |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid_LTR - Hybrid | ndcg | 0.00506 | 0.00332 | 0.00684 | 0.00010 | 0.00040 | higher | 589 | 478 | 3744 |
| Hybrid_LTR - Hybrid | recall | 0.00011 | -0.00102 | 0.00124 | 0.85141 | 0.85141 | not significant | 94 | 80 | 4637 |
| Hybrid_LTR_W - Hybrid | ndcg | 0.00620 | 0.00450 | 0.00790 | 0.00010 | 0.00040 | higher | 617 | 440 | 3754 |
| Hybrid_LTR_W - Hybrid | recall | 0.00062 | -0.00051 | 0.00174 | 0.28657 | 0.57314 | not significant | 99 | 67 | 4645 |

## novel: K=50, mean and SD over seeds 42/43/44

| model | ndcg_mean | ndcg_std | recall_mean | recall_std | hit_users_mean |
|---|---|---|---|---|---|
| Popularity | 0.00161 | 0.00000 | 0.00437 | 0.00000 | 63.00000 |
| Recent | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| RecentPop | 0.00161 | 0.00000 | 0.00437 | 0.00000 | 63.00000 |
| ALS | 0.00535 | 0.00029 | 0.01381 | 0.00077 | 147.33333 |
| BPR | 0.00445 | 0.00015 | 0.01206 | 0.00062 | 128.66667 |
| ALS+BPR | 0.00589 | 0.00003 | 0.01523 | 0.00037 | 157.33333 |
| GNN | 0.00293 | 0.00041 | 0.00814 | 0.00107 | 92.33333 |
| Hybrid_ALS_only | 0.00523 | 0.00049 | 0.01313 | 0.00053 | 145.66667 |
| Hybrid | 0.00544 | 0.00036 | 0.01383 | 0.00098 | 153.00000 |
| Hybrid_LTR_mf | 0.00507 | 0.00023 | 0.01387 | 0.00064 | 163.00000 |
| Hybrid_LTR_mf_gnn | 0.00548 | 0.00020 | 0.01388 | 0.00056 | 164.33333 |
| Hybrid_LTR_no_history | 0.00572 | 0.00024 | 0.01498 | 0.00109 | 175.66667 |
| Hybrid_LTR | 0.00572 | 0.00024 | 0.01498 | 0.00109 | 175.66667 |
| Hybrid_LTR_W_mf | 0.00512 | 0.00009 | 0.01446 | 0.00064 | 169.33333 |
| Hybrid_LTR_W_mf_gnn | 0.00539 | 0.00041 | 0.01410 | 0.00076 | 166.66667 |
| Hybrid_LTR_W_no_history | 0.00595 | 0.00026 | 0.01503 | 0.00027 | 174.66667 |
| Hybrid_LTR_W | 0.00595 | 0.00026 | 0.01503 | 0.00027 | 174.66667 |

### novel: confirmatory family (Holm within the family)

| comparison | metric | difference | ci_low | ci_high | p_value | p_holm | claim | users_better | users_worse | users_tied |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid - ALS | ndcg | 0.00009 | -0.00040 | 0.00060 | 0.71283 | 1.00000 | not significant | 156 | 129 | 4102 |
| Hybrid - ALS | recall | 0.00002 | -0.00149 | 0.00146 | 0.97900 | 1.00000 | not significant | 112 | 83 | 4192 |
| Hybrid - BPR | ndcg | 0.00099 | 0.00037 | 0.00165 | 0.00180 | 0.02880 | higher | 183 | 106 | 4098 |
| Hybrid - BPR | recall | 0.00177 | 0.00016 | 0.00343 | 0.02980 | 0.38736 | not significant | 140 | 79 | 4168 |
| Hybrid - ALS+BPR | ndcg | -0.00044 | -0.00091 | 0.00001 | 0.05379 | 0.59174 | not significant | 132 | 135 | 4120 |
| Hybrid - ALS+BPR | recall | -0.00140 | -0.00279 | -0.00010 | 0.04040 | 0.48475 | not significant | 84 | 77 | 4226 |
| Hybrid - GNN | ndcg | 0.00252 | 0.00194 | 0.00315 | 0.00010 | 0.00420 | higher | 203 | 83 | 4101 |
| Hybrid - GNN | recall | 0.00569 | 0.00410 | 0.00744 | 0.00010 | 0.00420 | higher | 154 | 47 | 4186 |
| Hybrid - Popularity | ndcg | 0.00384 | 0.00274 | 0.00500 | 0.00010 | 0.00420 | higher | 231 | 59 | 4097 |
| Hybrid - Popularity | recall | 0.00947 | 0.00678 | 0.01232 | 0.00010 | 0.00420 | higher | 229 | 57 | 4101 |
| Hybrid - Recent | ndcg | 0.00544 | 0.00445 | 0.00653 | 0.00010 | 0.00420 | higher | 242 | 0 | 4145 |
| Hybrid - Recent | recall | 0.01383 | 0.01142 | 0.01647 | 0.00010 | 0.00420 | higher | 242 | 0 | 4145 |
| Hybrid - RecentPop | ndcg | 0.00384 | 0.00274 | 0.00500 | 0.00010 | 0.00420 | higher | 231 | 59 | 4097 |
| Hybrid - RecentPop | recall | 0.00947 | 0.00678 | 0.01232 | 0.00010 | 0.00420 | higher | 229 | 57 | 4101 |
| Hybrid_LTR - ALS | ndcg | 0.00037 | -0.00046 | 0.00121 | 0.39206 | 1.00000 | not significant | 173 | 124 | 4090 |
| Hybrid_LTR - ALS | recall | 0.00116 | -0.00096 | 0.00333 | 0.28927 | 1.00000 | not significant | 146 | 79 | 4162 |
| Hybrid_LTR - BPR | ndcg | 0.00126 | 0.00052 | 0.00203 | 0.00110 | 0.01870 | higher | 183 | 110 | 4094 |
| Hybrid_LTR - BPR | recall | 0.00292 | 0.00108 | 0.00475 | 0.00190 | 0.02880 | higher | 156 | 72 | 4159 |
| Hybrid_LTR - ALS+BPR | ndcg | -0.00017 | -0.00101 | 0.00066 | 0.68393 | 1.00000 | not significant | 165 | 133 | 4089 |
| Hybrid_LTR - ALS+BPR | recall | -0.00026 | -0.00229 | 0.00172 | 0.79652 | 1.00000 | not significant | 138 | 88 | 4161 |
| Hybrid_LTR - GNN | ndcg | 0.00279 | 0.00208 | 0.00355 | 0.00010 | 0.00420 | higher | 209 | 105 | 4073 |
| Hybrid_LTR - GNN | recall | 0.00684 | 0.00501 | 0.00881 | 0.00010 | 0.00420 | higher | 192 | 75 | 4120 |
| Hybrid_LTR - Popularity | ndcg | 0.00411 | 0.00316 | 0.00512 | 0.00010 | 0.00420 | higher | 221 | 46 | 4120 |
| Hybrid_LTR - Popularity | recall | 0.01061 | 0.00813 | 0.01333 | 0.00010 | 0.00420 | higher | 208 | 44 | 4135 |
| Hybrid_LTR - Recent | ndcg | 0.00572 | 0.00477 | 0.00672 | 0.00010 | 0.00420 | higher | 243 | 0 | 4144 |
| Hybrid_LTR - Recent | recall | 0.01498 | 0.01245 | 0.01764 | 0.00010 | 0.00420 | higher | 243 | 0 | 4144 |
| Hybrid_LTR - RecentPop | ndcg | 0.00411 | 0.00316 | 0.00512 | 0.00010 | 0.00420 | higher | 221 | 46 | 4120 |
| Hybrid_LTR - RecentPop | recall | 0.01061 | 0.00813 | 0.01333 | 0.00010 | 0.00420 | higher | 208 | 44 | 4135 |
| Hybrid_LTR_W - ALS | ndcg | 0.00060 | -0.00015 | 0.00137 | 0.12509 | 1.00000 | not significant | 177 | 122 | 4088 |
| Hybrid_LTR_W - ALS | recall | 0.00121 | -0.00073 | 0.00316 | 0.22948 | 1.00000 | not significant | 142 | 79 | 4166 |
| Hybrid_LTR_W - BPR | ndcg | 0.00150 | 0.00068 | 0.00232 | 0.00050 | 0.00900 | higher | 193 | 103 | 4091 |
| Hybrid_LTR_W - BPR | recall | 0.00297 | 0.00098 | 0.00492 | 0.00280 | 0.03920 | higher | 161 | 74 | 4152 |
| Hybrid_LTR_W - ALS+BPR | ndcg | 0.00007 | -0.00067 | 0.00080 | 0.85671 | 1.00000 | not significant | 172 | 121 | 4094 |
| Hybrid_LTR_W - ALS+BPR | recall | -0.00021 | -0.00202 | 0.00154 | 0.82102 | 1.00000 | not significant | 129 | 80 | 4178 |
| Hybrid_LTR_W - GNN | ndcg | 0.00303 | 0.00220 | 0.00393 | 0.00010 | 0.00420 | higher | 208 | 115 | 4064 |
| Hybrid_LTR_W - GNN | recall | 0.00689 | 0.00488 | 0.00902 | 0.00010 | 0.00420 | higher | 196 | 84 | 4107 |
| Hybrid_LTR_W - Popularity | ndcg | 0.00435 | 0.00327 | 0.00548 | 0.00010 | 0.00420 | higher | 219 | 49 | 4119 |
| Hybrid_LTR_W - Popularity | recall | 0.01066 | 0.00805 | 0.01343 | 0.00010 | 0.00420 | higher | 210 | 44 | 4133 |
| Hybrid_LTR_W - Recent | ndcg | 0.00595 | 0.00490 | 0.00708 | 0.00010 | 0.00420 | higher | 247 | 0 | 4140 |
| Hybrid_LTR_W - Recent | recall | 0.01503 | 0.01251 | 0.01777 | 0.00010 | 0.00420 | higher | 247 | 0 | 4140 |
| Hybrid_LTR_W - RecentPop | ndcg | 0.00435 | 0.00327 | 0.00548 | 0.00010 | 0.00420 | higher | 219 | 49 | 4119 |
| Hybrid_LTR_W - RecentPop | recall | 0.01066 | 0.00805 | 0.01343 | 0.00010 | 0.00420 | higher | 210 | 44 | 4133 |

### novel: secondary family (Holm within the family)

| comparison | metric | difference | ci_low | ci_high | p_value | p_holm | claim | users_better | users_worse | users_tied |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid_LTR - Hybrid | ndcg | 0.00027 | -0.00047 | 0.00102 | 0.47445 | 0.59394 | not significant | 157 | 156 | 4074 |
| Hybrid_LTR - Hybrid | recall | 0.00114 | -0.00054 | 0.00281 | 0.18198 | 0.59394 | not significant | 133 | 93 | 4161 |
| Hybrid_LTR_W - Hybrid | ndcg | 0.00051 | -0.00021 | 0.00124 | 0.16988 | 0.59394 | not significant | 167 | 142 | 4078 |
| Hybrid_LTR_W - Hybrid | recall | 0.00119 | -0.00041 | 0.00280 | 0.14849 | 0.59394 | not significant | 133 | 86 | 4168 |

