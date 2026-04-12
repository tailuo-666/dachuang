# Step 2 Retrieval Diagnostic: fading_memory_ha_gnn_diag_20260410_rerun

## Retrieval Inputs
```json
{
  "retrieval_query_zh": "动态系统 衰减记忆 HA-GNN 历史信息 预测 时序依赖",
  "retrieval_query_en": "Fading Memory Dynamic Systems HA-GNN Historical Access Prediction Mechanism Temporal Dependencies",
  "keywords_en": [
    "Fading Memory",
    "HA-GNN",
    "Dynamic Systems",
    "Historical Access",
    "Prediction Mechanism"
  ],
  "bm25_query": "Fading Memory Dynamic Systems HA-GNN Historical Access Prediction Mechanism Temporal Dependencies Fading Memory HA-GNN Dynamic Systems Historical Access Prediction Mechanism"
}
```

## Initialization
- initialized: `True`
- initialize error type: ``
- initialize error message: ``

## Main Retrieval
- ok: `True`
- error type: ``
- error message: ``
- doc count: `5`
```json
{
  "branch_counts": {
    "bm25_en": 15,
    "dense_zh": 15,
    "dense_en": 15
  },
  "rrf_pool_count": 20,
  "returned_count": 5,
  "bm25_query": "Fading Memory Dynamic Systems HA-GNN Historical Access Prediction Mechanism Temporal Dependencies Fading Memory HA-GNN Dynamic Systems Historical Access Prediction Mechanism",
  "retrieval_query_zh": "动态系统 衰减记忆 HA-GNN 历史信息 预测 时序依赖",
  "retrieval_query_en": "Fading Memory Dynamic Systems HA-GNN Historical Access Prediction Mechanism Temporal Dependencies"
}
```

## Main Retrieval Chunks

### Chunk 1
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `0.1278`
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
      "bm25_en"
    ],
    "branch_ranks": {
      "bm25_en": 15
    },
    "rrf_score": 0.127794
  },
  "rrf_score": 0.127794
}
```
- content preview:
```text
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad...(truncated)
```

### Chunk 2
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0325`
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
      "dense_en": 2
    },
    "rrf_score": 0.032522
  },
  "rrf_score": 0.032522
}
```
- content preview:
```text
. Recurrent neural networks (RNNs) such as long short-term memory (LSTM)(^{9}) and gated recurrent units (GRU)(^{10}) have shown potential in modeling temporal sequences, yet their application to longitudinal neuroimaging data with irregular sampling remains difficult. In this study, we tackle these challenges by introducing a history-aware graph neural network (HA-GNN) model, which integrates rs-fMRI-derived functional connectivity graphs with longitudinal visit information. Our strategy combines a GNN for spatial brain network representation with an RNN for temporal representation, enabling...(truncated)
```

### Chunk 3
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0323`
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
      "dense_zh": 3,
      "dense_en": 1
    },
    "rrf_score": 0.032266
  },
  "rrf_score": 0.032266
}
```
- content preview:
```text
. However, most Send correspondence to M.M.: moghaddami@oakland.edu ## Page 2 existing GNN methods focus on cross-sectional data and do not fully utilize the longitudinal aspects of disease progression.(^{8}) Longitudinal prediction poses additional challenges, including irregular time intervals between clinical visits, missing imaging data, and varying lengths of visit history across subjects.(^{8}) These factors complicate the modeling of temporal processes and restrict the applicability of standard sequential or graph-based models
```

### Chunk 4
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0315`
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
      "dense_zh": 2,
      "dense_en": 5
    },
    "rrf_score": 0.031514
  },
  "rrf_score": 0.031514
}
```
- content preview:
```text
6. CONCLUSIONS In this study, we establish the effectiveness of using resting-state functional MRI and our proposed history-aware graph neural network model for predicting the progression of cognitive impairment. Using focal loss and pretraining, our model demonstrates the ability to predict converter subjects despite data imbalance, missing visits, and irregular visit distances. The results indicate that our model can effectively capture the temporal dynamics of cognitive impairment progression. Our approach offers a promising method for early detection and monitoring of Alzheimer's disease w...(truncated)
```

### Chunk 5
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `0.0300`
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
      "dense_zh": 11,
      "dense_en": 3
    },
    "rrf_score": 0.029958
  },
  "rrf_score": 0.029958
}
```
- content preview:
```text
. Additionally, investigating the relationship between visit history length and prediction accuracy could provide insights into the minimum number of data points required for reliable prognostic predictions. Fourth, while our model achieves respectable performance metrics, the lack of built-in interpretability mechanisms limits its clinical utility. Future iterations should incorporate attention mechanisms or other interpretability tools to identify which brain regions and connections are most influential in driving predictions, as demonstrated in Kim et al.(^{19}) Such interpretability would...(truncated)
```

## Plan Reference
```json
{
  "original_query": "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？",
  "normalized_query_zh": "动态系统中衰减记忆性质的作用与HA-GNN历史预测机制的本质差异分析",
  "retrieval_query_zh": "动态系统 衰减记忆 HA-GNN 历史信息 预测 时序依赖",
  "retrieval_query_en": "Fading Memory Dynamic Systems HA-GNN Historical Access Prediction Mechanism Temporal Dependencies",
  "crawler_query_en": "Fading Memory HA-GNN Dynamic Systems",
  "keywords_zh": [
    "衰减记忆",
    "HA-GNN",
    "动态系统",
    "历史信息",
    "预测机制"
  ],
  "keywords_en": [
    "Fading Memory",
    "HA-GNN",
    "Dynamic Systems",
    "Historical Access",
    "Prediction Mechanism"
  ],
  "required_aspects": [
    "definition of Fading Memory in dynamic systems",
    "role of Fading Memory in dynamic systems",
    "HA-GNN historical access mechanism",
    "prediction mechanism of HA-GNN",
    "differences between Fading Memory and HA-GNN temporal representations"
  ]
}
```