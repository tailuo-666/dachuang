# Step 3 Evaluation Diagnostic: fading_memory_ha_gnn_diag_20260410_fixed

## Evaluation Input
- input source: `main_retrieval`
- input doc count: `5`

## Evaluation Result
```json
{
  "sufficient": true,
  "partial_answerable": true,
  "aspect_coverage": 1.0,
  "support_strength": 0.6341,
  "noise_ratio": 0.0,
  "covered_aspects": [
    "definition of fading memory in dynamical systems",
    "function of fading memory in dynamical systems",
    "mechanism of historical access usage in HA-GNN",
    "similarities between fading memory and HA-GNN memory",
    "differences between fading memory and HA-GNN prediction framework"
  ],
  "weak_aspects": [],
  "missing_aspects": [],
  "aspect_scores": {
    "definition of fading memory in dynamical systems": 0.8586,
    "function of fading memory in dynamical systems": 0.8586,
    "mechanism of historical access usage in HA-GNN": 0.4707,
    "similarities between fading memory and HA-GNN memory": 0.5475,
    "differences between fading memory and HA-GNN prediction framework": 0.435
  },
  "aspect_best_chunks": {
    "definition of fading memory in dynamical systems": [
      2,
      0,
      1
    ],
    "function of fading memory in dynamical systems": [
      2,
      0,
      1
    ],
    "mechanism of historical access usage in HA-GNN": [
      3,
      2,
      0
    ],
    "similarities between fading memory and HA-GNN memory": [
      0,
      2,
      3
    ],
    "differences between fading memory and HA-GNN prediction framework": [
      0,
      2,
      3
    ]
  },
  "next_action": "answer",
  "top1_coverage": 0.8586,
  "avg_top3_coverage": 0.7352,
  "unique_sources": 2,
  "reason": "aspect_coverage=1.00, support_strength=0.63, noise_ratio=0.00, missing_aspects_count=0, missing_aspects=none, next_action=answer"
}
```

## Aspect to Best Chunks
```json
{
  "definition of fading memory in dynamical systems": [
    2,
    0,
    1
  ],
  "function of fading memory in dynamical systems": [
    2,
    0,
    1
  ],
  "mechanism of historical access usage in HA-GNN": [
    3,
    2,
    0
  ],
  "similarities between fading memory and HA-GNN memory": [
    0,
    2,
    3
  ],
  "differences between fading memory and HA-GNN prediction framework": [
    0,
    2,
    3
  ]
}
```

## Scored Docs

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.7404`
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
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 2,
      "dense_zh": 6,
      "dense_en": 2
    },
    "rrf_score": 0.0...(truncated)
```
- content preview:
```text
[20] J. Dambre, D. Verstraeten, B. Schrauwen, S. Massar, Information processing capacity of dynamical systems, Sci. Rep. 2 (1) (2012) 514. doi:10.1038/srep00514. [21] J.-P. Ortega, F. Rossmannek, Echoes of the past: A unified perspective on fading memory and echo states (2025). doi:10.48550/ARXIV.2508.19145. [22] L. Gonon, J.-P. Ortega, Fading memory echo state networks are universal, Neural Networks 138 (2021) 10–13. doi:10.1016/j.neunet.2021.01.025.
```

### Chunk 2
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.6065`
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
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 1,
      "dense_zh": 12,
      "dense_en": 7
    },
    "rrf_score": 0....(truncated)
```
- content preview:
```text
[24] Y. Huo, T. Chaffey, R. Sepulchre, Kernel modelling of fading memory systems (2024). doi:10.48550/ARXIV.2403.11945. [25] R. Sepulchre, Fading memory [from the editor], IEEE Control Systems 41 (1) (2021) 4–5. doi:10.1109/mcs.2020.3033098.
```

### Chunk 3
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.8586`
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
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 10,
      "dense_zh": 15,
      "dense_en": 4
    },
    "rrf_score": 0...(truncated)
```
- content preview:
```text
## Page 1 State-space fading memory Gustave Bainier(^{a,*}), Antoine Chaillet(^{b}), Rodolphe Sepulchre(^{c,d}), Alessio Franci(^{a,e,*}) (^{a})Dept. of Electrical Engineering and Computer Science, University of Liège, 10 allée de la Découverte, Liège, 4000, Belgium (^{b})Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, Gif-sur-Yvette, 91190, France (^{c})KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics, Kasteel Park Arenberg 10, Leuven, 3001, Belgium (^{d})Department of Engine...(truncated)
```

### Chunk 4
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.5350`
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
  "retrieval_debug": {
    "branch_hits": [
      "dense_zh",
      "dense_en"
    ],
    "branch_ranks": {
      "dense_zh": 1,
      "dense_en": 1
    },
    "rrf_score": 0.032787
  },
  "rrf_score": 0.032787
}
```
- content preview:
```text
. Recurrent neural networks (RNNs) such as long short-term memory (LSTM)(^{9}) and gated recurrent units (GRU)(^{10}) have shown potential in modeling temporal sequences, yet their application to longitudinal neuroimaging data with irregular sampling remains difficult. In this study, we tackle these challenges by introducing a history-aware graph neural network (HA-GNN) model, which integrates rs-fMRI-derived functional connectivity graphs with longitudinal visit information. Our strategy combines a GNN for spatial brain network representation with an RNN for temporal representation, enabling...(truncated)
```

### Chunk 5
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.5888`
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
  "retrieval_debug": {
    "branch_hits": [
      "bm25_en",
      "dense_en"
    ],
    "branch_ranks": {
      "bm25_en": 4,
      "dense_en": 3
    },
    "rrf_score": 0.031498
  },
  "rrf_score": 0.031498
}
```
- content preview:
```text
N. Tsitsiklis, Worst-case identification of nonlinear fading memory systems, Automatica 31 (3) (1995) 503–508. doi:10.1016/0005-1098(94)00131-2.
```