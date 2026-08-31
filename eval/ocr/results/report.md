# OCR eval (300 dpi)

CER/WER against hand-verified transcriptions (lower is better).

| sample | docling CER | docling WER | docling s | qwen25vl CER | qwen25vl WER | qwen25vl s | unlimited CER | unlimited WER | unlimited s |
|---|---|---|---|---|---|---|---|---|---|
| apollo_p21 | 0.131 | 0.134 | - | 0.001 | 0.001 | 187 | 0.032 | 0.042 | 34 |
| apollo_p5 | 0.074 | 0.074 | - | 0.001 | 0.002 | - | 0.069 | 0.067 | 55 |
| lec11_p8 | 0.160 | 0.173 | 3 | 0.011 | 0.020 | 197 | 40.982 | 40.756 | 444 |
| lec7_p5 | 0.090 | 0.095 | 3 | 0.127 | 0.180 | 201 | 0.113 | 0.118 | 41 |
| ship_p3 | 0.040 | 0.041 | 30 | 0.002 | 0.003 | 219 | 0.444 | 0.428 | 44 |
| thesis_p9 | 0.068 | 0.069 | 4 | 0.075 | 0.082 | 181 | 0.078 | 0.108 | 46 |

## Means

| backend | mean CER | mean WER | total s |
|---|---|---|---|
| docling | 0.0938 | 0.0977 | 40 |
| qwen25vl | 0.0362 | 0.0478 | 986 |
| unlimited | 6.9531 | 6.9197 | 663 |
