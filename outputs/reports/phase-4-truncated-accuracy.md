Accuracy of truncated (cut at the 4,096 window) and finished completions, first sample, from the per-sample files of the archives; the extractor's fallback (final phrase, last number, boxed) scores a truncated completion like any other, and an extraction failure is scored wrong.

| run | eval | truncated | right when truncated | right when finished | how the truncated-right were extracted | extraction failed among truncated |
|---|---|---|---|---|---|---|
| sft think-tag | gsm8k | 370 of 1319 (28.1 %) | 31.1 % | 91.9 % | final_phrase 96, last_number 17, boxed 2 | 0 % |
| sft think-tag | math500 | 260 of 500 (52.0 %) | 13.5 % | 90.8 % | final_phrase 29, boxed 6 | 68 % |
| rft think-tag | gsm8k | 200 of 1319 (15.2 %) | 25.0 % | 87.7 % | last_number 7, final_phrase 40, boxed 3 | 0 % |
| rft think-tag | math500 | 198 of 500 (39.6 %) | 11.6 % | 85.8 % | final_phrase 19, boxed 4 | 68 % |
| grpo think-tag | gsm8k | 61 of 1319 (4.6 %) | 9.8 % | 85.8 % | final_phrase 5, last_number 1 | 0 % |
| grpo think-tag | math500 | 104 of 500 (20.8 %) | 4.8 % | 78.8 % | final_phrase 4, boxed 1 | 76 % |
| sft zero-shot | gsm8k | 228 of 1319 (17.3 %) | 67.1 % | 85.3 % | boxed 134, last_number 8, final_phrase 11 | 0 % |
| sft zero-shot | math500 | 296 of 500 (59.2 %) | 58.8 % | 69.6 % | boxed 174 | 17 % |
| rft zero-shot | gsm8k | 261 of 1319 (19.8 %) | 69.0 % | 84.1 % | boxed 165, last_number 7, final_phrase 8 | 0 % |
| rft zero-shot | math500 | 309 of 500 (61.8 %) | 61.2 % | 70.7 % | boxed 187, final_phrase 2 | 17 % |
| grpo zero-shot | gsm8k | 211 of 1319 (16.0 %) | 73.0 % | 86.5 % | boxed 140, final_phrase 10, last_number 4 | 0 % |
| grpo zero-shot | math500 | 325 of 500 (65.0 %) | 62.8 % | 67.4 % | boxed 201, final_phrase 3 | 11 % |
| base think-tag | gsm8k | 472 of 1319 (35.8 %) | 45.6 % | 70.8 % | last_number 215 | 0 % |
| base think-tag | math500 | 100 of 500 (20.0 %) | 0.0 % | 56.0 % | - | 97 % |
