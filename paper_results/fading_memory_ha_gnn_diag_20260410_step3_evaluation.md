# Step 3 Evaluation Diagnostic: fading_memory_ha_gnn_diag_20260410

## Evaluation Input
- input source: `branch_fallback`
- input doc count: `5`

## Evaluation Result
```json
{
  "sufficient": false,
  "partial_answerable": true,
  "aspect_coverage": 0.6,
  "support_strength": 0.3617,
  "noise_ratio": 0.4,
  "covered_aspects": [
    "definition of fading memory in dynamic systems",
    "comparison of fading memory and GNN memory models",
    "similarities between dynamical system fading memory and GNN attention"
  ],
  "weak_aspects": [],
  "missing_aspects": [
    "HA-GNN historical information access mechanism",
    "differences in historical information processing between theory and GNNs"
  ],
  "aspect_scores": {
    "definition of fading memory in dynamic systems": 0.5143,
    "HA-GNN historical information access mechanism": 0.18,
    "comparison of fading memory and GNN memory models": 0.5143,
    "similarities between dynamical system fading memory and GNN attention": 0.3,
    "differences in historical information processing between theory and GNNs": 0.3
  },
  "aspect_best_chunks": {
    "definition of fading memory in dynamic systems": [
      1,
      3,
      0
    ],
    "HA-GNN historical information access mechanism": [
      0
    ],
    "comparison of fading memory and GNN memory models": [
      3,
      1,
      4
    ],
    "similarities between dynamical system fading memory and GNN attention": [
      2,
      3,
      1
    ],
    "differences in historical information processing between theory and GNNs": [
      2,
      0,
      3
    ]
  },
  "next_action": "crawl_more",
  "top1_coverage": 0.5143,
  "avg_top3_coverage": 0.4714,
  "unique_sources": 2,
  "reason": "aspect_coverage=0.60, support_strength=0.36, noise_ratio=0.40, missing_aspects_count=2, missing_aspects=HA-GNN historical information access mechanism, differences in historical information processing between theory and GNNs, next_action=crawl_more"
}
```

## Aspect to Best Chunks
```json
{
  "definition of fading memory in dynamic systems": [
    1,
    3,
    0
  ],
  "HA-GNN historical information access mechanism": [
    0
  ],
  "comparison of fading memory and GNN memory models": [
    3,
    1,
    4
  ],
  "similarities between dynamical system fading memory and GNN attention": [
    2,
    3,
    1
  ],
  "differences in historical information processing between theory and GNNs": [
    2,
    0,
    3
  ]
}
```

## Scored Docs

### Chunk 1
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.2571`
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

### Chunk 2
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

### Chunk 3
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
. Additionally, investigating the relationship between visit history length and prediction accuracy could provide insights into the minimum number of data points required for reliable prognostic predictions. Fourth, while our model achieves respectable performance metrics, the lack of built-in interpretability mechanisms limits its clinical utility. Future iterations should incorporate attention mechanisms or other interpretability tools to identify which brain regions and connections are most influential in driving predictions, as demonstrated in Kim et al.(^{19}) Such interpretability would...(truncated)
```

### Chunk 4
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
W. Sandberg, Z₊ fading memory and extensions of input–output maps, International Journal of Circuit Theory and Applications 29 (4) (2001) 381–388. doi:10.1002/cta.156.
```

### Chunk 5
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.3857`
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
M. Kang, Memristive devices and systems, Proceedings of the IEEE 64 (2) (1976) 209–223. doi:10.1109/proc.1976.10092. [32] F. Forni, R. Sepulchre, Gradient modelling of memristive systems (2025). arXiv:2504.10093. [33] C. M. Kellett, A compendium of comparison function results, Mathematics of Control, Signals, and Systems 26 (3) (2014) 339–374. doi:10.1007/s00498-014-0128-8. [34] T. I. Donchev, B. M.
```