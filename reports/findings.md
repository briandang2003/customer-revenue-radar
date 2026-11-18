# Key Findings

All figures come from `python -m src.pipeline` on the UCI Online Retail II data (cleaned: 770,524 sales
lines, 5,839 customers, 1 Dec 2009 – 9 Dec 2011). Charts are in `figures/`, in the notebooks and in the
dashboard.

## Revenue
- Total cleaned revenue is **£16.47M** across **36,332 orders**. Average order value is **£453**, which
  reflects a large share of wholesale buyers.
- **Strong Q4 seasonality.** September–November produced 36% of 2010 revenue, and November was the peak
  month in both years (£1.12M each).
- **Concentrated market.** The United Kingdom is 84% of revenue. EIRE (£0.57M, from just 3 customer
  accounts), the Netherlands, Germany and France follow.
- 28% of customers ordered only once. Turning first orders into second orders is a clear opportunity.
- Data quality: 6,209 sale lines were reversed by an exactly matching cancellation and are excluded.
  Without that rule, two cancelled bulk orders (80,995 and 74,215 units) would rank among the top products.

## Customers (RFM)
| Segment | Customers | Share of customers | Share of revenue | Avg days since last order |
|---|---:|---:|---:|---:|
| High-value | 1,254 | 21% | 69% | 19 |
| At-risk | 984 | 17% | 11% | 376 |
| Loyal | 595 | 10% | 10% | 85 |
| Regular | 1,094 | 19% | 6% | 74 |
| Inactive | 1,351 | 23% | 2% | 460 |
| New | 561 | 10% | 2% | 35 |

## Clusters (K-Means, k = 3)
- k = 3 had the best silhouette score (0.317) among k ≥ 3. The silhouette score drops quickly after k = 4.
- *Recent · Frequent · High-spend*: 2,273 customers (39%) producing 88% of revenue.
- *Lapsed · Occasional · Low-spend*: 2,689 customers (46%), 9% of revenue.
- *Recent · Occasional · Low-spend*: 877 customers (15%), 4% of revenue. 561 of them are in the RFM *New*
  segment (median tenure 70 days).
- PCA keeps 72.5% of the variance in 2 dimensions. PC1 measures engagement and value and separates the
  high-spend and lapsed clusters. PC2 mostly measures tenure (loading 0.73), so new customers sit low
  on PC2.

## Repeat-purchase prediction (SVM)
- Test period: 90 days after 1 Sep 2011, never used for training or tuning. 42.8% of the 5,216 eligible
  customers bought again.
- SVM: ROC-AUC 0.785, precision 0.728, recall 0.571, F1 0.640.
- Rule baseline ("bought in the last 90 days"): ROC-AUC 0.760, precision 0.678, recall 0.586, F1 0.629.
- ROC-AUC gain +0.025 (bootstrap 95% CI +0.016 to +0.035). That is a small but consistent improvement in
  ranking customers. At the chosen threshold the SVM sends 145 fewer wasted contacts and finds slightly
  fewer buyers.
- Scored as of 10 Dec 2011: 855 customers are *likely to return* (≥ 70%), 1,461 are *undecided*
  (40–70%) and 3,523 are *unlikely* (< 40%). The median At-risk customer has a 29% probability.

## Recommendations
1. Protect High-value customers with VIP perks, not discounts.
2. Run win-back campaigns for At-risk customers, ranked by predicted probability.
3. Spend incentives on the *undecided* tier, where they are most likely to change behaviour.
4. Ramp stock and campaigns from September for the Q4 peak.
5. Grow strong international markets to reduce dependence on the UK.
