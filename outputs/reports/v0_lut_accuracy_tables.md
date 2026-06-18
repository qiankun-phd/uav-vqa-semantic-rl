# V0.5 LUT Accuracy Tables

Mean expected accuracy aggregated over channel, freshness, and risk-level cells.

| question type | service level | poor view | medium view | good view |
|---|---:|---:|---:|---:|
| counting | 0 | 0.448 | 0.454 | 0.459 |
| counting | 1 | 0.317 | 0.469 | 0.574 |
| counting | 2 | 0.432 | 0.582 | 0.698 |
| presence | 0 | 0.496 | 0.503 | 0.508 |
| presence | 1 | 0.351 | 0.526 | 0.644 |
| presence | 2 | 0.474 | 0.641 | 0.770 |
| risk | 0 | 0.424 | 0.430 | 0.434 |
| risk | 1 | 0.300 | 0.413 | 0.501 |
| risk | 2 | 0.411 | 0.553 | 0.663 |

## Fresh + Good/Medium Sanity Check

| question type | view | cache s=0 | light s=1 | image s=2 |
|---|---|---:|---:|---:|
| counting | medium | 0.586 | 0.582 | 0.634 |
| counting | good | 0.591 | 0.706 | 0.763 |
| presence | medium | 0.649 | 0.656 | 0.700 |
| presence | good | 0.655 | 0.795 | 0.840 |
| risk | medium | 0.554 | 0.502 | 0.602 |
| risk | good | 0.560 | 0.607 | 0.722 |
