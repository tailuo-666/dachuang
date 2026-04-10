# Step 3 Evaluation Diagnostic: fading_memory_ha_gnn_diag_20260410_utf8

## Evaluation Input
- input source: `branch_fallback`
- input doc count: `5`

## Evaluation Result
```json
{
  "sufficient": false,
  "partial_answerable": true,
  "aspect_coverage": 0.75,
  "support_strength": 0.4125,
  "noise_ratio": 0.2,
  "covered_aspects": [
    "role of fading memory in dynamic systems",
    "similarities between fading memory and HA-GNN",
    "essential differences between fading memory and HA-GNN prediction mechanism"
  ],
  "weak_aspects": [
    "mechanism of historical access information in HA-GNN"
  ],
  "missing_aspects": [],
  "aspect_scores": {
    "role of fading memory in dynamic systems": 0.5143,
    "mechanism of historical access information in HA-GNN": 0.3857,
    "similarities between fading memory and HA-GNN": 0.45,
    "essential differences between fading memory and HA-GNN prediction mechanism": 0.3
  },
  "aspect_best_chunks": {
    "role of fading memory in dynamic systems": [
      0,
      2,
      3
    ],
    "mechanism of historical access information in HA-GNN": [
      1,
      0,
      2
    ],
    "similarities between fading memory and HA-GNN": [
      3,
      0,
      1
    ],
    "essential differences between fading memory and HA-GNN prediction mechanism": [
      1,
      3,
      0
    ]
  },
  "next_action": "retrieve_more",
  "top1_coverage": 0.5143,
  "avg_top3_coverage": 0.4929,
  "unique_sources": 2,
  "reason": "aspect_coverage=0.75, support_strength=0.41, noise_ratio=0.20, missing_aspects_count=0, missing_aspects=none, next_action=retrieve_more"
}
```

## Aspect to Best Chunks
```json
{
  "role of fading memory in dynamic systems": [
    0,
    2,
    3
  ],
  "mechanism of historical access information in HA-GNN": [
    1,
    0,
    2
  ],
  "similarities between fading memory and HA-GNN": [
    3,
    0,
    1
  ],
  "essential differences between fading memory and HA-GNN prediction mechanism": [
    1,
    3,
    0
  ]
}
```

## Scored Docs

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.5143`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2603.23814 [ pdf , ps , other ]",
  "title": "arXiv:2603.23814 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2603.23814",
  "pdf_link": "https://arxiv.org/pdf/2603.23814",
  "source_file": "arXiv2603.23814 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
[24] Y. Huo, T. Chaffey, R. Sepulchre, Kernel modelling of fading memory systems (2024). doi:10.48550/ARXIV.2403.11945. [25] R. Sepulchre, Fading memory [from the editor], IEEE Control Systems 41 (1) (2021) 4–5. doi:10.1109/mcs.2020.3033098.
```

### Chunk 2
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.3857`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2604.06469 [ pdf , ps , other ]",
  "title": "arXiv:2604.06469 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2604.06469",
  "pdf_link": "https://arxiv.org/pdf/2604.06469",
  "source_file": "arXiv2604.06469 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
Architecture of our HA-GNN model Figure 4. Architecture of our HA-GNN model ## Page 5 2.5 Training and Evaluation We use a random 20% split of our data to pretrain our GCN component. The pretraining is a 3-class classification task where the model predicts the diagnosis of a subject (CN, MCI, or AD) in a given visit. Unlike our main task, during pretraining, the input consists of a single visit instead of the entire visit history of a subject. Fig. 5 shows the architecture of the pretraining model. Figure 5. Architecture of the GCN pretraining model Once pretraining is done, we use 5-fold cros...(truncated)
```

### Chunk 3
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.5143`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2603.23814 [ pdf , ps , other ]",
  "title": "arXiv:2603.23814 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2603.23814",
  "pdf_link": "https://arxiv.org/pdf/2603.23814",
  "source_file": "arXiv2603.23814 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
N. Tsitsiklis, Worst-case identification of nonlinear fading memory systems, Automatica 31 (3) (1995) 503–508. doi:10.1016/0005-1098(94)00131-2.
```

### Chunk 4
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.4500`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2603.23814 [ pdf , ps , other ]",
  "title": "arXiv:2603.23814 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2603.23814",
  "pdf_link": "https://arxiv.org/pdf/2603.23814",
  "source_file": "arXiv2603.23814 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
W. Sandberg, Z₊ fading memory and extensions of input–output maps, International Journal of Circuit Theory and Applications 29 (4) (2001) 381–388. doi:10.1002/cta.156.
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.3000`
- origin: `local_kb`
- aspects: []
- metadata:
```json
{
  "source": "arXiv:2604.06469 [ pdf , ps , other ]",
  "title": "arXiv:2604.06469 [ pdf , ps , other ]",
  "url": "https://arxiv.org/pdf/2604.06469",
  "pdf_link": "https://arxiv.org/pdf/2604.06469",
  "source_file": "arXiv2604.06469 [ pdf , ps , other ].pdf",
  "origin": "local_kb",
  "diagnostic_branch": "bm25_en"
}
```
- content preview:
```text
Model Acc. AUC-ROC BA CN to MCI MCI to AD HA-GNN (LSTM) (0.829 \pm 0.058) (0.852 \pm 0.065) (0.771 \pm 0.114) (0.688) (0.676) HA-GNN (RNN) (0.743 \pm 0.038) (0.771 \pm 0.067) (0.651 \pm 0.114) (0.5) (0.514) HA-GNN (GRU) (0.733 \pm 0.049) (0.786 \pm 0.062) (0.704 \pm 0.071) (0.625) (0.676) w/o pretraining (0.723 \pm 0.079) (0.642 \pm 0.095) (0.594 \pm 0.061) (0.375) (0.405) Due to our novel methodology, comparisons with the existing literature are difficult. However there are a few papers that are close to our task and methods.
```