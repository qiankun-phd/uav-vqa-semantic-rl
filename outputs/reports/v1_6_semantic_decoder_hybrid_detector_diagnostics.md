# Detector-Qwen Diagnostic Report

This report checks whether detector-generated semantic tokens are reliable enough for VQA service control.
Ground-truth annotations are used only for evaluation; the `s=1` Qwen input is detector output.

## Detector Count Quality

- count diagnostic rows: `3080`
- raw detections: `22346`
- count MAE: `5.599`
- zero-detection rate: `0.274`
- under-count ratio: `0.769`
- over-count ratio: `0.130`

## Per-class Count Error

| class | samples | count MAE |
|---|---:|---:|
| pedestrian | 474 | 10.264 |
| people | 437 | 8.112 |
| car | 467 | 6.413 |
| motor | 441 | 6.236 |
| van | 384 | 3.331 |
| tricycle | 307 | 2.427 |
| awning-tricycle | 197 | 2.122 |
| truck | 250 | 2.012 |
| bus | 123 | 1.187 |

## Detector Error vs Qwen Correctness for Semantic-token Evidence

| detector count error bin | Qwen accuracy | samples |
|---|---:|---:|
| 0 | 1.000 | 2250 |
| 1-2 | 0.907 | 8235 |
| 3-5 | 0.748 | 4041 |
| >5 | 0.712 | 3312 |

## Semantic-token Qwen Accuracy by Channel

| channel | accuracy | samples |
|---|---:|---:|
| bad | 0.847 | 5946 |
| medium | 0.847 | 5946 |
| good | 0.847 | 5946 |

## Counting Failure Examples

- `0000244_04000_d_0000009` channel=bad class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=bad class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=bad class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=medium class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=medium class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=medium class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=good class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=good class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000244_04000_d_0000009` channel=good class=car GT=20 detector=59 Pred=`59` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=bad class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=bad class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=bad class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=medium class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=medium class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
- `0000215_00447_d_0000257` channel=medium class=car GT=38 detector=70 Pred=`70` Q: How many car objects are in this area?
