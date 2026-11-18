# Key Findings

From `python -m src.pipeline` (770,524 cleaned rows, 5,839 customers, Dec 2009 – Dec 2011).
All money is in **USD** at 1 GBP = 1.58 USD.

## Revenue
- **$26.0M** total revenue from 36,332 orders; the average order is **$716**.
- **Busiest season:** Sep–Nov is 36% of the year's revenue; November peaks at ~$1.78M in both years.
- **One market:** the UK is 84% of revenue. Next is Ireland (EIRE) with $0.90M, from just 3 customer accounts.
- **28%** of customers ordered only once. Getting a second order is the easiest win.

## Customer segments (RFM = Recency, Frequency, Monetary scores)
| Segment | Customers | % of customers | % of revenue | Days since last order |
|---|---:|---:|---:|---:|
| High-value | 1,254 | 21% | 69% | 19 |
| At-risk | 984 | 17% | 11% | 376 |
| Loyal | 595 | 10% | 10% | 85 |
| Regular | 1,094 | 19% | 6% | 74 |
| Inactive | 1,351 | 23% | 2% | 460 |
| New | 561 | 10% | 2% | 35 |

## Customer groups (K-Means = automatic grouping)
- **3 groups**, chosen by the best silhouette score (0.317; higher means cleaner separation):
  - *Recent · Frequent · High-spend*: 39% of customers, 88% of revenue.
  - *Lapsed · Occasional · Low-spend*: 46% of customers, 9% of revenue.
  - *Recent · Occasional · Low-spend*: 15% of customers, 4% of revenue. All 561 "New" customers are here.
- **PCA** (shrinking 7 measurements into 2) keeps 72.5% of the information:
  - Axis 1 measures customer value.
  - Axis 2 measures how long someone has been a customer.

## Buy-again prediction (SVM)
- Tested once, on the 90 days after 1 Sep 2011. 42.8% of customers bought again.
- **SVM:** ROC-AUC 0.785, precision 0.723, recall 0.582. **Simple rule:** 0.760, 0.678, 0.586.
  - ROC-AUC means ranking quality: 0.5 is a coin flip, 1.0 is perfect.
- **Gain:** +0.025 ROC-AUC (95% range +0.016 to +0.035), small but real. 123 fewer false alarms than the rule.
- **Scored as of 10 Dec 2011:**
  - 858 customers are likely to return.
  - 1,451 are on the fence.
  - 3,530 are unlikely to return.
  - The median At-risk customer has a 29% chance.

## Recommendations
1. VIP care for High-value customers, not discounts.
2. Win back At-risk customers, highest predicted chance first.
3. Spend discounts on the on-the-fence group, where they change behavior.
4. Stock up and ramp campaigns from September.
5. Grow other countries to reduce reliance on the UK.
