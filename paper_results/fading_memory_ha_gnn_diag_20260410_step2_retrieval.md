# Step 2 Retrieval Diagnostic: fading_memory_ha_gnn_diag_20260410

## Retrieval Inputs
```json
{
  "retrieval_query_zh": "动态系统 衰减记忆 HA-GNN 历史访问 预测机制 对比",
  "retrieval_query_en": "comparison of fading memory in dynamic systems and HA-GNN history access prediction mechanisms",
  "keywords_en": [
    "HA-GNN",
    "dynamic systems",
    "fading memory",
    "historical information",
    "prediction mechanism"
  ],
  "bm25_query": "comparison of fading memory in dynamic systems and HA-GNN history access prediction mechanisms HA-GNN dynamic systems fading memory historical information prediction mechanism"
}
```

## Initialization
- initialized: `True`
- initialize error type: ``
- initialize error message: ``

## Main Retrieval
- ok: `False`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`
```json
{}
```

## Branch Fallback

### bm25_en
- ok: `True`
- query: `comparison of fading memory in dynamic systems and HA-GNN history access prediction mechanisms HA-GNN dynamic systems fading memory historical information prediction mechanism`
- error type: ``
- error message: ``
- doc count: `15`

### Chunk 1
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
Architecture of our HA-GNN model Figure 4. Architecture of our HA-GNN model ## Page 5 2.5 Training and Evaluation We use a random 20% split of our data to pretrain our GCN component. The pretraining is a 3-class classification task where the model predicts the diagnosis of a subject (CN, MCI, or AD) in a given visit. Unlike our main task, during pretraining, the input consists of a single visit instead of the entire visit history of a subject. Fig. 5 shows the architecture of the pretraining model. Figure 5. Architecture of the GCN pretraining model Once pretraining is done, we use 5-fold cros...(truncated)
```

### Chunk 2
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
[24] Y. Huo, T. Chaffey, R. Sepulchre, Kernel modelling of fading memory systems (2024). doi:10.48550/ARXIV.2403.11945. [25] R. Sepulchre, Fading memory [from the editor], IEEE Control Systems 41 (1) (2021) 4–5. doi:10.1109/mcs.2020.3033098.
```

### Chunk 3
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
. Additionally, investigating the relationship between visit history length and prediction accuracy could provide insights into the minimum number of data points required for reliable prognostic predictions. Fourth, while our model achieves respectable performance metrics, the lack of built-in interpretability mechanisms limits its clinical utility. Future iterations should incorporate attention mechanisms or other interpretability tools to identify which brain regions and connections are most influential in driving predictions, as demonstrated in Kim et al.(^{19}) Such interpretability would...(truncated)
```

### Chunk 4
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
W. Sandberg, Z₊ fading memory and extensions of input–output maps, International Journal of Circuit Theory and Applications 29 (4) (2001) 381–388. doi:10.1002/cta.156.
```

### Chunk 5
- source: `arXiv:2603.23814 [ pdf , ps , other ]`
- title: `arXiv:2603.23814 [ pdf , ps , other ]`
- score: `null`
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
  "origin": "local_kb"
}
```
- content preview:
```text
M. Kang, Memristive devices and systems, Proceedings of the IEEE 64 (2) (1976) 209–223. doi:10.1109/proc.1976.10092. [32] F. Forni, R. Sepulchre, Gradient modelling of memristive systems (2025). arXiv:2504.10093. [33] C. M. Kellett, A compendium of comparison function results, Mathematics of Control, Signals, and Systems 26 (3) (2014) 339–374. doi:10.1007/s00498-014-0128-8. [34] T. I. Donchev, B. M.
```

### dense_zh
- ok: `False`
- query: `动态系统 衰减记忆 HA-GNN 历史访问 预测机制 对比`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`

### dense_en
- ok: `False`
- query: `comparison of fading memory in dynamic systems and HA-GNN history access prediction mechanisms`
- error type: `TypeError`
- error message: `'VLLMOpenAIEmbeddings' object is not callable`
- doc count: `0`

## Fallback Evaluation Input
- input source: `branch_fallback`
- branch order: `bm25_en, dense_zh, dense_en`
- doc count: `5`

### Chunk 1
- source: `arXiv:2604.06469 [ pdf , ps , other ]`
- title: `arXiv:2604.06469 [ pdf , ps , other ]`
- score: `null`
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
- score: `null`
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
- score: `null`
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
- score: `null`
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
- score: `null`
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

## Plan Reference
```json
{
  "original_query": "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？",
  "normalized_query_zh": "动态系统中衰减记忆性质的作用与HA-GNN模型历史访问预测机制的相似性及本质区别分析",
  "retrieval_query_zh": "动态系统 衰减记忆 HA-GNN 历史访问 预测机制 对比",
  "retrieval_query_en": "comparison of fading memory in dynamic systems and HA-GNN history access prediction mechanisms",
  "crawler_query_en": "fading memory dynamic systems HA-GNN history attention",
  "keywords_zh": [
    "HA-GNN",
    "动态系统",
    "衰减记忆",
    "历史访问信息",
    "预测机制"
  ],
  "keywords_en": [
    "HA-GNN",
    "dynamic systems",
    "fading memory",
    "historical information",
    "prediction mechanism"
  ],
  "required_aspects": [
    "definition of fading memory in dynamic systems",
    "HA-GNN historical information access mechanism",
    "comparison of fading memory and GNN memory models",
    "similarities between dynamical system fading memory and GNN attention",
    "differences in historical information processing between theory and GNNs"
  ]
}
```