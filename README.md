# Case Study: Fine-Tuning DeepVariant on RNA Sequencing Data to Evaluate Impact on Low-Coverage Reads

[DeepVariant](https://github.com/google/deepvariant) is a deep learning-based variant caller. I noticed the project on Google's 20% page, and I explored this case study to see if AI in biocomputation would match my interests. 

In this case study, I attempt to fine-tune the model on RNA sequencing data to evaluate its effectiveness in variant calling for undercovered regions. RNA reads are generally longer than DNA reads and can span areas with low sequencing depth, potentially improving variant detection. By fine-tuning DeepVariant on RNA-seq data, I aimed to explore whether the model could leverage these properties to enhance accuracy in low coverage genomic regions. We will be looking at precision, recall, NA calls and overall F1 Score.

The environment used for this study was a google cloud instance with 1 A100 GPU.


## Getting the Training Data
To train, we will use RNA reads.

**Data:** [PacBio RNA Datasets](https://www.pacb.com/connect/datasets/#RNA-datasets)

```sh
wget https://downloads.pacbcloud.com/public/dataset/Kinnex-full-length-RNA/DATA-Vega-UHRR2024/1-Sreads/segmented.bam
wget https://downloads.pacbcloud.com/public/dataset/Kinnex-full-length-RNA/DATA-Vega-UHRR2024/1-Sreads/segmented.bam.pbi
```

**Reference Genome:**

```sh
wget ftp://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```

Also, download the index or index it yourself with:

```sh
samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```


## Getting the Test Data
We want to test on DNA reads.

For simplicity, let's use DNA reads commonly used to test DeepVariant. Copy these over to your directory:

```sh
gsutil -m cp gs://deepvariant/training-case-study/BGISEQ_PE100_NA12878.sorted.chr*.bam* "${data_dir}"
gsutil -m cp -r "gs://deepvariant/training-case-study/ucsc_hg19.fa*" "${data_dir}"
gsutil -m cp -r "gs://deepvariant/training-case-study/BGISEQ-HG001/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_*" "${data_dir}"
```

This provides a DNA read, a reference genome, and an index of high-confidence regions and variant ground truth calls.
We will only test on a portion of this data by splitting out **chromosome 1**, which is generally used for testing.


## Preprocessing the Training Data
**Tools:**
- **pbmm2** from Bioconda: [pbmm2 GitHub](https://github.com/PacificBiosciences/pbmm2/)
- **samtools** for sorting and indexing

### Aligning Data
We also need to reindex the training data, as PacBio reads have a `.pbi` index, but DeepVariant accepts a `.bai` index.

For training, we ensure that the reads are sorted and aligned to the reference genome using `pbmm2`. For a **32GB dataset**, this step will take a few hours.

```sh
pbmm2 align reference.fasta updated_segmented.bam aligned_output.bam --sort --preset CCS
```


## Generating Ground Truths
Unlike with the training data, there wasn't a ground truth vcf file available for the RNA reads. So, I used another version of DeepVariant, designed for RNA data, to generate a ground truth set.

```sh
sudo docker run --gpus 1 \
  -v /home/${USER}:/home/${USER} \
  "google/deepvariant-1.8.0-gpu" \
  run_deepvariant \
  --model_type WES \
  ...
```

## Quick Tests
Now that our data is prepared, let's clarify the problem.

We can run inference on **chr1**, then benchmark using `hap.py` with the ground truth set.

| Type  | Filter | TRUTH.TOTAL | TRUTH.TP | TRUTH.FN | QUERY.TOTAL | QUERY.FP | QUERY.UNK | FP.gt | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score | TRUTH.TOTAL.TiTv_ratio | QUERY.TOTAL.TiTv_ratio | TRUTH.TOTAL.het_hom_ratio | QUERY.TOTAL.het_hom_ratio |
|-------|--------|-------------|----------|----------|-------------|----------|-----------|-------|---------------|-----------------|---------------|----------------|------------------------|-----------------------|-------------------------|--------------------------|
| INDEL | ALL    | 38202       | 36742    | 1460     | 79490       | 3194     | 37719     | 1233  | 0.961782      | 0.923535        | 0.474513      | 0.942271       | NaN                    | NaN                   | 1.309084                 | 2.461615                  |
| INDEL | PASS   | 38202       | 36742    | 1460     | 79490       | 3194     | 37719     | 1233  | 0.961782      | 0.923535        | 0.474513      | 0.942271       | NaN                    | NaN                   | 1.309084                 | 2.461615                  |
| SNP   | ALL    | 247102      | 246841   | 261      | 285311      | 296      | 38047     | 58    | 0.998944      | 0.998803        | 0.133353      | 0.998873       | 2.191042                | 2.106741               | 1.390929                 | 1.364694                  |
| SNP   | PASS   | 247102      | 246841   | 261      | 285311      | 296      | 38047     | 58    | 0.998944      | 0.998803        | 0.133353      | 0.998873       | 2.191042                | 2.106741               | 1.390929                 | 1.364694                  |


We can **downsample** this data by 50% using:

```sh
samtools view -s 0.5
```

Downsampling reduces sequencing coverage by selectively retaining a subset of the original reads. This simulates lower coverage sequencing, explored further in the next section.

Running again on downsampled data shows

| Type  | Filter | TRUTH.TOTAL | TRUTH.TP | TRUTH.FN | QUERY.TOTAL | QUERY.FP | QUERY.UNK | FP.gt | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score | TRUTH.TOTAL.TiTv_ratio | QUERY.TOTAL.TiTv_ratio | TRUTH.TOTAL.het_hom_ratio | QUERY.TOTAL.het_hom_ratio |
|-------|--------|-------------|----------|----------|-------------|----------|-----------|-------|---------------|-----------------|---------------|----------------|------------------------|-----------------------|-------------------------|--------------------------|
| INDEL | ALL    | 38202       | 27934    | 10268    | 60215       | 7154     | 24552     | 3934  | 0.731218      | 0.7994          | 0.407739      | 0.763791       | NaN                    | NaN                   | 1.30908                  | 1.84383                   |
| INDEL | PASS   | 38202       | 27934    | 10268    | 60215       | 7154     | 24552     | 3934  | 0.731218      | 0.7994          | 0.407739      | 0.763791       | NaN                    | NaN                   | 1.30908                  | 1.84383                   |
| SNP   | ALL    | 247102      | 223103   | 23999    | 264348      | 4313     | 36859     | 3689  | 0.902878      | 0.981041        | 0.139434      | 0.940338       | 2.19104                 | 2.11538                | 1.39093                  | 1.19887                   |
| SNP   | PASS   | 247102      | 223103   | 23999    | 264348      | 4313     | 36859     | 3689  | 0.902878      | 0.981041        | 0.139434      | 0.940338       | 2.19104                 | 2.11538                | 1.39093                  | 1.19887                   |


### Key Metrics

| Model         | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score |
|--------------|---------------|------------------|----------------|----------------|
| **Original**   | 0.961782      | 0.923535         | 0.474513       | 0.942271       |
| **Downsampled** | 0.731218      | 0.7994           | 0.407739       | 0.763791       |

Like expected, we see a drop across the board. This means that there is some room for improvement.

## Coverage Analysis
To analyze test and training data coverage, I used `samtools depth`.

```sh
samtools depth sorted_aligned.bam > depth.txt
```

Summarize depth:

```sh
awk '{ if ($3 >= 1 && $3 <= 7) low++;
       else if ($3 > 7 && $3 <= 20) medium++;
       else high++;
     } 
     END { 
       print "Low coverage (1-7):", low; 
       print "Medium coverage (8-20):", medium; 
       print "High coverage (>20):", high; 
     }' depth.txt > coverage_buckets.txt
```

| Dataset           | Low Coverage (1-7) | Medium Coverage (8-20) | High Coverage (>20) |
|------------------|-------------------|--------------------|------------------|
| Original        | 459,658            | 4,355,513          | 220,221,898       |
| Downsampled     | 117,313,191        | 106,047,660        | 1,170,103         |
| RNA Training    | 478,281,601        | 137,024,743        | 95,840,555        |



## Make Examples
We split the data into **training** and **validation** sets. **Chromosome 10** is used for validation, and the rest for training. Deepvariant provides a solution to create examples with training/validation data sets, and shuffle the data. The data is then stored in google storage. Below is an example call for the training set, but we will need to run this again for the validation set.

```sh
(time seq 0 $((N_SHARDS-1)) | \
  parallel --halt 2 --line-buffer \
    sudo docker run \
      -v ${HOME}:${HOME} \
      ${DOCKER_IMAGE} \
      make_examples \
      --mode training \
      --ref "${HOME}/reference/GRCh38_no_alt_analysis_set.fasta" \
      --reads "${HOME}/deepvariant_trainingdata/sorted_aligned_train.bam" \
      --examples "${HOME}/output/training_set.with_label.tfrecord@${N_SHARDS}.gz" \
      --truth_variants "${HOME}/reference/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
      --confident_regions "${HOME}/reference/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed" \
      --task {} \
      --regions "'chr1 chr2 chr3 ... chr22'" \
      --exclude_regions "'chr10'" \
) 2>&1 | tee "${HOME}/training_set.with_label.make_examples.log"
```

## Fine tuning
To set the hyperparamaters, we can either modify dv_config.py or use command line flags. By default, the config for training the WGS model is given as such:

```
def get_wgs_config(config: ml_collections.ConfigDict):
  """Config parameters for wgs training."""

  config.train_dataset_pbtxt = '/path/to/your/train.dataset_config.pbtxt'
  config.tune_dataset_pbtxt = '/path/to/your/tune.dataset_config.pbtxt'
  config.init_checkpoint = ''
  # If set to 0, use full validation dataset.
  config.num_validation_examples = 150_000

  config.best_checkpoint_metric = 'tune/f1_weighted'
  config.batch_size = 16384
  config.num_epochs = 10

  # Optimizer hparams
  config.optimizer = 'sgd'
  config.momentum = 0.9
  config.use_ema = True
  config.ema_momentum = 0.99
  config.optimizer_weight_decay = 0.0

  config.weight_decay = 0.0001

  config.early_stopping_patience = 100
  config.learning_rate = 0.01
  config.learning_rate_num_epochs_per_decay = 2.25
  config.learning_rate_decay_rate = 0.9999
  config.warmup_steps = 0

  config.label_smoothing = 0.01
  config.backbone_dropout_rate = 0.2
```

I attempted two iterations. The results are summarized in the next section. I conversed with chatGPT in order to understand what hyper paramaters to tune, and used my best judgement for both iterations. 

In the first iteration:
- config.learning_rate = 0.001 (was 0.01)
- config.batch_size = 256 (was 16384) - to avoid any oom errors
- config.num_validation_examples = 0 (use full set)
- config.early_stopping_patience = 50 (was 250) - prevent overfitting
- config.warmup_steps = 1000 (was 0) - 

The first iteration results showed signs of overfitting (lower recall, higher precision), so I increased the weight decay and changed a few more paramaters.

In the second iteration:
- config.num_epochs = 20 (was 10) added more epochs
- config.learning_rate = 0.001
- config.batch_size = 256
- config.num_validation_examples = 0 (use full set)
- config.early_stopping_patience = 50 
- config.warmup_steps = 500
- config.learning_rate_decay_rate=0.995 (was .9999) smooth out learning
- config.weight_decay=0.00005 (was .0001) - less overfitting

Example command:

```
  ( time sudo docker run --gpus 1 \
    -v /home/rubinthomas:/home/rubinthomas \
    -w /home/rubinthomas \
    deepvariant-custom_v2:1.8.0 \
    train \
    --config=/home/rubinthomas/dv_config.py:wgs \
    --config.train_dataset_pbtxt="gs://deep-var-test/training_set.dataset_config.pbtxt" \
    --config.tune_dataset_pbtxt="gs://deep-var-test/validation_set.dataset_config.pbtxt" \
    --config.init_checkpoint="gs://deepvariant/models/DeepVariant/1.8.0/checkpoints/wgs/deepvariant.wgs.ckpt" \
    --config.num_epochs=20 \
    --config.learning_rate=0.001 \
    --config.learning_rate_decay_rate = 0.995 \
    --config.batch_size=256 \
    --config.warmup_steps=500 \
    --config.early_stopping_patience = 50 \
    --config.weight_decay=0.0005 \
    --config.optimizer='sgd' \
    --config.early_stopping_patience=250 \
    --config.num_validation_examples=20000 \
    --config.tune_every_steps=5000 \
    --experiment_dir="/home/rubinthomas/deepvariant_model/checkpoints_finetune_v4" \
    --strategy=mirrored \
) > "/home/rubinthomas/deepvariant_trainingdata/train.log" 2>&1 &
```

## Results

### Results after V1
| Type  | Filter | TRUTH.TOTAL | TRUTH.TP | TRUTH.FN | QUERY.TOTAL | QUERY.FP | QUERY.UNK | FP.gt | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score | TRUTH.TOTAL.TiTv_ratio | QUERY.TOTAL.TiTv_ratio | TRUTH.TOTAL.het_hom_ratio | QUERY.TOTAL.het_hom_ratio |
|-------|--------|-------------|----------|----------|-------------|----------|-----------|-------|---------------|-----------------|---------------|----------------|------------------------|-----------------------|-------------------------|--------------------------|
| INDEL | ALL    | 38202       | 22350    | 15852    | 39629       | 3419     | 13485     | 3147  | 0.585048      | 0.869224        | 0.340281      | 0.699371       | NaN                    | NaN                   | 1.30908                  | 1.03282                   |
| INDEL | PASS   | 38202       | 22350    | 15852    | 39629       | 3419     | 13485     | 3147  | 0.585048      | 0.869224        | 0.340281      | 0.699371       | NaN                    | NaN                   | 1.30908                  | 1.03282                   |
| SNP   | ALL    | 247102      | 215037   | 32065    | 250690      | 5260     | 30352     | 4308  | 0.870236      | 0.976128        | 0.121074      | 0.920145       | 2.19104                 | 2.11049                | 1.39093                  | 1.17069                   |
| SNP   | PASS   | 247102      | 215037   | 32065    | 250690      | 5260     | 30352     | 4308  | 0.870236      | 0.976128        | 0.121074      | 0.920145       | 2.19104                 | 2.11049                | 1.39093                  | 1.17069                   |

### Results from V2

| Type  | Filter | TRUTH.TOTAL | TRUTH.TP | TRUTH.FN | QUERY.TOTAL | QUERY.FP | QUERY.UNK | FP.gt | METRIC.Recall | METRIC.Precision | METRIC.Frac_NA | METRIC.F1_Score | TRUTH.TOTAL.TiTv_ratio | QUERY.TOTAL.TiTv_ratio | TRUTH.TOTAL.het_hom_ratio | QUERY.TOTAL.het_hom_ratio |
|-------|--------|-------------|----------|----------|-------------|----------|-----------|-------|---------------|-----------------|---------------|----------------|------------------------|-----------------------|-------------------------|--------------------------|
| INDEL | ALL    | 38202       | 23750    | 14452    | 42300       | 3050     | 15069     | 2852  | 0.621695      | 0.887995        | 0.356241      | 0.731358       | NaN                    | NaN                   | 1.30908                  | 0.892249                  |
| INDEL | PASS   | 38202       | 23750    | 14452    | 42300       | 3050     | 15069     | 2852  | 0.621695      | 0.887995        | 0.356241      | 0.731358       | NaN                    | NaN                   | 1.30908                  | 0.892249                  |
| SNP   | ALL    | 247102      | 215764   | 31338    | 253949      | 4603     | 33529     | 4045  | 0.873178      | 0.979117        | 0.13203       | 0.923118       | 2.19104                 | 2.11985                | 1.39093                  | 1.15887                   |
| SNP   | PASS   | 247102      | 215764   | 31338    | 253949      | 4603     | 33529     | 4045  | 0.873178      | 0.979117        | 0.13203       | 0.923118       | 2.19104                 | 2.11985                | 1.39093                  | 1.15887                   |

## Performance Comparison

To assess the effectiveness of the fine-tuned models (V1 and V2) against the original model performance on downsampled data, we compare key metrics: recall, precision, fraction of NA calls, and F1 score.

### Key Metrics Comparison

#### INDEL Performance

| Model           | Recall  | Precision | Frac_NA  | F1 Score  |
|----------------|---------|-----------|----------|----------|
| **Original**   | 0.731218 | 0.7994    | 0.407739 | 0.763791 |
| **V1 Fine-Tuned** | 0.585048 | 0.869224  | 0.340281 | 0.699371 |
| **V2 Fine-Tuned** | 0.621695 | 0.887995  | 0.356241 | 0.731358 |

#### SNP Performance

| Model           | Recall  | Precision | Frac_NA  | F1 Score  |
|----------------|---------|-----------|----------|----------|
| **Original**   | 0.902878 | 0.981041  | 0.139434 | 0.940338 |
| **V1 Fine-Tuned** | 0.870236 | 0.976128  | 0.121074 | 0.920145 |
| **V2 Fine-Tuned** | 0.873178 | 0.979117  | 0.132030 | 0.923118 |

### Analysis

Both **V1** and **V2** show signs of **overfitting** to the training data. While precision improves significantly over the original model, recall drops in both fine-tuned versions, particularly for **INDELs**, indicating that the model is becoming more conservative and failing to detect some true variants. 

- **V1 Fine-Tuned** for indels show a big drop in recall(0.731 → 0.585) while gaining higher precision (0.799 → 0.869).  
  - This suggests the model is filtering more aggressively, potentially missing true variants in favor of reducing false positives.
- **V2 Fine-Tuned** improves recall slightly (0.621 vs. 0.585 in V1) while further increasing precision (0.888).  


#### **Potential Causes of Overfitting**
1. **Small Validation Set**  
   - A larger validation set could lead to moregeneralization, but due to time constraints, I decided to not regenerate the data.
2. **Low Coverage Regions**  
   - Both training and test data are from low coverage regions, where distinguishing real variants from sequencing noise is more difficult.
   - The model may have over-adapted to patterns in the training set, leading to overfitting.
3. **Hyperparamaters**
   - In the future, with smaller datasets, I would like to learn more which hyperparamaters cause which effect through experimentation. The training time for this model was long with 1 gpu, and I decided to only train twice.