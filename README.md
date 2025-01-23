

mkdir deepvariant_model
wget -P deepvariant_model https://storage.googleapis.com/deepvariant/models/DeepVariant/1.5.0/wgs/model.ckpt.data-00000-of-00001
wget -P deepvariant_model https://storage.googleapis.com/deepvariant/models/DeepVariant/1.5.0/wgs/model.ckpt.index
wget -P deepvariant_model https://storage.googleapis.com/deepvariant/models/DeepVariant/1.5.0/wgs/model.ckpt.meta


```
| Type  | Filter | TRUTH.TOTAL | TRUTH.TP | TRUTH.FN | QUERY.TOTAL | QUERY.FP | QUERY.UNK | FP.gt | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score | TRUTH.TOTAL.TiTv_ratio | QUERY.TOTAL.TiTv_ratio | TRUTH.TOTAL.het_hom_ratio | QUERY.TOTAL.het_hom_ratio |
|-------|--------|-------------|----------|----------|-------------|----------|-----------|-------|---------------|-----------------|---------------|----------------|------------------------|-----------------------|-------------------------|--------------------------|
| INDEL | ALL    | 4           | 4        | 0        | 48          | 0        | 44        | 0     | 1.000000       | 1.000000         | 0.916667       | 1.000000        | NaN                    | NaN                    | 0.333333                 | 1.238095                  |
| INDEL | PASS   | 4           | 4        | 0        | 48          | 0        | 44        | 0     | 1.000000       | 1.000000         | 0.916667       | 1.000000        | NaN                    | NaN                    | 0.333333                 | 1.238095                  |
| SNP   | ALL    | 45          | 44       | 1        | 190         | 0        | 146       | 0     | 0.977778       | 1.000000         | 0.768421       | 0.988764        | 1.142857                | 2.064516                | 0.363636                 | 1.345679                  |
| SNP   | PASS   | 45          | 44       | 1        | 190         | 0        | 146       | 0     | 0.977778       | 1.000000         | 0.768421       | 0.988764        | 1.142857                | 2.064516                | 0.363636                 | 1.345679                  |

```
