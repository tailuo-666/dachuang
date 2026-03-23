import re
from typing import List, Tuple, Dict, Optional
import spacy
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate

class QueryDecomposerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self._nlp = None
       
        # 优化：按意图分类的关键词库，权重不同
        self.INTENT_PATTERNS = {
            "comparison": {
                "keywords": ["区别", "差异", "不同", "vs", "对比", "优劣", "哪个好", "比较"],
                "weight": 1.2
            },
            "multi_hop": {
                "keywords": ["先.*再.*", "流程", "步骤", "如何实现", "原理", "怎么做", "如何部署", "如何.*"],
                "weight": 1.0
            },
            "aggregation": {
                "keywords": ["总结", "归纳", "共同点", "总和"],
                "weight": 0.5
            }
        }
       
        # 实体类型权重配置
        self.ENTITY_TYPE_WEIGHTS = {
            "ORG": 1.0,      # 组织机构
            "PERSON": 0.9,   # 人名
            "GPE": 0.8,      # 地理政治实体（地名）
            "PRODUCT": 0.7,  # 产品
            "EVENT": 0.6,    # 事件
            "WORK_OF_ART": 0.5,  # 艺术作品
            "LAW": 0.5,      # 法律
            "LANGUAGE": 0.4, # 语言
            "DATE": 0.3,     # 日期
            "TIME": 0.3,     # 时间
            "PERCENT": 0.2,  # 百分比
            "MONEY": 0.2,    # 货币
            "QUANTITY": 0.2, # 数量
            "CARDINAL": 0.1, # 基数
            "ORDINAL": 0.1   # 序数
        }
       
        # 词性权重配置
        self.POS_WEIGHTS = {
            "NOUN": 0.8,     # 名词
            "PROPN": 0.9,    # 专有名词
            "VERB": 0.6,     # 动词
            "ADJ": 0.4,      # 形容词
            "ADV": 0.3,      # 副词
        }

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("zh_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy中文模型未安装。请运行: python -m spacy download zh_core_web_sm"
                )
        return self._nlp
       
    # 停用词过滤
    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        使用完整的中文停用词库过滤
        """
        # 扩展的中文停用词库
        stop_words = {
            "什么", "怎么", "如何", "的", "是", "在", "了", "和", "有", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "那", "里", "吗", "啊", "呢", "吧", "呀", "哦", "嗯", "哈", "嘿", "哎",
            "可以", "能够", "应该", "需要", "想要", "希望", "觉得", "认为", "以为", "知道", "明白",
            "了解", "清楚", "确定", "肯定", "当然", "确实", "真的", "非常", "特别", "比较", "相当",
            "已经", "正在", "将要", "刚才", "马上", "立刻", "现在", "今天", "明天", "昨天", "以前",
            "以后", "之前", "之后", "同时", "然后", "接着", "最后", "首先", "其次", "再次", "总之",
            "比如", "例如", "像", "如同", "类似", "包括", "包含", "涉及", "关于", "对于", "至于",
            "因为", "所以", "由于", "因此", "但是", "然而", "不过", "虽然", "尽管", "如果", "假如",
            "只要", "只有", "除非", "无论", "不管", "即使", "哪怕", "或者", "还是", "还是", "以及",
            "并且", "而且", "此外", "另外", "除了", "除了", "之外", "之间", "之中", "之上", "之下",
            "之中", "之间", "之中", "之间", "之中", "之间", "之中", "之间", "之中", "之间"
        }
        return [token for token in tokens if token not in stop_words]

    # 实体标准化处理
    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """
        实体标准化处理：去重、过滤空值、去除单字
        """
        seen = set()
        normalized = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and entity not in seen:
                seen.add(entity)
                normalized.append(entity)
        return normalized
   
    # 实体类型权重获取
    def _get_entity_weight(self, entity_type: str) -> float:
        """
        根据实体类型获取权重
        """
        return self.ENTITY_TYPE_WEIGHTS.get(entity_type, 0.3)
   
    @lru_cache(maxsize=256)
    def _extract_entities_nlp(self, query: str) -> List[str]:
        """
        使用spaCy进行智能实体提取，包括NER和关键词提取
        使用LRU缓存优化性能
        """
        # 使用spaCy进行实体识别
        doc = self.nlp(query)
       
        # 1. 提取命名实体（NER），把上一步的doc存进named_entities
        named_entities = []
        for ent in doc.ents:
            named_entities.append(ent.text)
       
        # 2. 基于词性标注提取关键词
        keywords = []
        for token in doc:
            # 提取名词、专有名词和动词
            if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                # 过滤停用词和单字
                if len(token.text) > 1 and not token.is_stop:
                    keywords.append(token.text)
       
        # 3. 合并并标准化实体
        all_entities = named_entities + keywords
        filtered_entities = self._filter_stopwords(all_entities)
        normalized_entities = self._normalize_entities(filtered_entities)
       
        return normalized_entities
   
    @lru_cache(maxsize=256)
    def _calculate_complexity_score(self, query: str, entities_tuple: tuple) -> float:
        """
        计算复杂性评分，考虑实体类型、词性、关键词和句法结构
        使用LRU缓存优化性能，entities_tuple用于缓存兼容性
        """
        score = 0.0
        doc = self.nlp(query)
        entities = list(entities_tuple)
       
        # 1. 基础实体数量评估 - 区分是否为真正需要分解的多实体查询
        if len(entities) >= 2:
            # 检查是否存在真正的对比或关系关键词
            has_comparison = any(kw in query for intent_data in self.INTENT_PATTERNS.values() 
                                for kw in intent_data["keywords"] if intent_data.get("weight", 0) >= 0.8)
            # 检查是否为属性查询模式，如"北京人口"、"苹果市值"等
            has_attribute_pattern = any([
                re.search(r'(.*)(人口|市值|股价|GDP|房价|面积|长度|重量|容量|销量|收入|利润|价格|成本)', query),
                re.search(r'(.*)(创始人|作者|发明者|开发者|制作者)', query),
                re.search(r'(.*)(历史|发展|起源|背景|简介|定义|概念)', query)
            ])
            if has_comparison:
                score += 1.0  # 对比类查询，需要分解
            elif has_attribute_pattern:
                score += 0.1  # 属性查询，不需要分解
            else:
                # 多个实体但不是对比类也不是属性查询，可能是并列查询
                score += 0.3  # 中等加分，但仍需LLM判断
        elif len(entities) == 0:
            score -= 0.2
       
        # 2. 命名实体类型加权
        entity_types_found = set()
        for ent in doc.ents:
            entity_weight = self._get_entity_weight(ent.label_)
            score += entity_weight * 0.3
            entity_types_found.add(ent.label_)
       
        # 3. 多种实体类型组合加分
        if len(entity_types_found) >= 2:
            score += 0.2
       
        # 4. 词性组合加权
        pos_tags = [token.pos_ for token in doc]
        if "NOUN" in pos_tags and "VERB" in pos_tags:
            score += 0.15
        if "PROPN" in pos_tags:
            score += 0.1
       
        # 5. 关键词加权 - 更精确的意图识别
        for intent, data in self.INTENT_PATTERNS.items():
            for kw in data["keywords"]:
                if re.search(kw, query) if '.' in kw else kw in query:
                    score += data["weight"]
                    # 命中一次即可，避免重复加分
                    break
       
        # 6. 句法结构加权 - 区分真正需要分解的连接词
        # 专门针对对比类连接词
        if re.search(r'.*(区别|差异|不同|对比|比较|vs|哪个好).*和.*', query):
            score += 0.6  # 明确的对比意图
        elif re.search(r'.+和.+', query):
            # 非对比性的"和"连接，如"北京和上海的人口" vs "北京和上海的区别"
            if any(comp_word in query for comp_word in ["区别", "差异", "不同", "对比", "比较", "vs"]):
                score += 0.5  # 有对比意图
            else:
                score += 0.1  # 单纯的并列关系
       
        if re.search(r'.+与.+', query):
            if any(comp_word in query for comp_word in ["区别", "差异", "不同", "对比", "比较", "vs"]):
                score += 0.5  # 有对比意图
            else:
                score += 0.1  # 单纯的并列关系
       
        if re.search(r'.+或者.+', query):
            score += 0.25
       
        # 7. 复杂句式检测
        if re.search(r'.+先.+再.+', query):
            score += 1.0  # 明确的多步骤意图
        # 更精确地检测"如何"和"怎么"，排除"怎么样"这类简单疑问词
        has_how_question = re.search(r'.+如何.+', query) or (re.search(r'.*怎么.*', query) and '怎么样' not in query)
        if has_how_question:
            # 检查是否是简单询问，如"如何部署" vs "如何部署和配置"
            if any(word in query for word in ["流程", "步骤", "方法", "怎么做"]):
                score += 0.8  # 明确的多步骤意图
            elif re.search(r'.+如何.+(和|与).+', query) or re.search(r'.*怎么.*(和|与).*', query):  # 如"如何部署和配置"或"怎么部署和配置"
                score += 0.7  # 包含多个操作
            elif re.search(r'.*怎么.*让.*', query) or re.search(r'.+如何.+让.+', query):  # 如"怎么让RAG查得更准"
                score += 0.8  # 表达改善方法的意图，通常是复杂查询
            else:
                # 对于简单"如何"或"怎么"查询，检查是否涉及多个实体或复杂主题
                if len(entities) <= 2:
                    score += 0.6  # "如何"或"怎么"类查询通常需要分解，即使实体不多
                else:
                    score += 0.6  # 涉及多个实体的"如何"查询，可能需要分解
        # 处理"怎么样"、"是什么"等简单疑问句
        elif '怎么样' in query or '是什么' in query or '是什么样的' in query:
            # 判断是否为简单查询
            if len(entities) <= 1:
                score -= 0.3  # 简单的"怎么样"或"是什么"查询，降低分解倾向
       
        return score

    def route_query(self, query: str) -> Tuple[bool, str, List[str]]:
        """
        主入口：决定处理策略
        Returns: (needs_decomposition, reason, sub_questions)
        """

        # 1. 实体提取、标准化和去重
        entities = self._extract_entities_nlp(query)
        # 将entities转换为tuple以支持缓存
        score = self._calculate_complexity_score(query, tuple(entities))
       
        print(f"🔍 Query: {query} | Score: {score:.2f} | Entities: {entities}")

        # 阈值策略 - 更加保守以减少不必要的分解
        HIGH_THRESHOLD = 0.75  # 确信需要分解
        LOW_THRESHOLD = 0.45  # 确信不需要

        if score >= HIGH_THRESHOLD:
            print("🚀 触发规则分解")
            return True, "High Complexity Score", self._generate_subquestions_llm(query)
       
        elif score <= LOW_THRESHOLD:
            print("🛑 简单查询，直连检索")
            return False, "Simple Query", [query]
       
        else:
            print("⚖️ 模糊区间，调用 LLM 轻量判断")
            is_complex = self._llm_light_check(query)
            if is_complex:
                return True, "LLM Judged Complex", self._generate_subquestions_llm(query)
            else:
                return False, "LLM Judged Simple", [query]

    def _llm_light_check(self, query: str) -> bool:
        """
        优化点3：Few-shot Prompting 提升准确率
        """
        prompt = f"""## 任务目标
        精准判断用户查询是否包含**多重意图**或需要**多步推理**，从而决定是否进行查询分解。

        ---

        ## 判断标准（满足任一即回答"是"）

        **1. 多实体对比类**
        - 明确对比动词：区别、差异、对比、比较、vs、哪个好、优劣
        - 隐含对比：A和B的[属性]（如"北京和上海人口"）

        **2. 多步骤操作类**
        - 流程性动词：如何、怎么、步骤、流程、教程、指南
        - 时序连接词：先...再...、然后、接着、最后  

        **3. 聚合分析类**
        - 总结、归纳、共同点、差异点、综合分析

        **4. 条件组合类**
        - 或、或者、还是（如"Python还是Java适合后端"）  

        ---

        ## 示例库

        ### ❌ 不需要分解（回答"否"）
        - "今天天气怎么样" → 单一事实查询
        - "Python的创始人是谁" → 单实体单属性
        - "北京人口" → 单实体属性查询
        - "什么是深度学习" → 定义性问题
        - "推荐一部科幻电影" → 主观推荐
        - "ChatGPT最新版本" → 单实体最新信息    

        ### ✅ 需要分解（回答"是"）
        - "苹果和微软的市值对比" → 双实体对比
        - "如何部署DeepSeek模型" → 多步骤流程
        - "Transformer和CNN的区别" → 技术对比
        - "北京和上海哪个适合定居" → 条件对比
        - "总结2024年AI领域突破" → 聚合总结
        - "先安装Python再配置环境变量" → 时序步骤

        ### ⚠️ 边界案例（需仔细判断）
        - "DeepSeek和ChatGPT" → **是**（缺属性但隐含对比）
        - "Python数据分析和机器学习" → **是**（多个主题）
        - "如何学习深度学习" → **否**（单主题无多步骤词）
        - "MySQL索引优化技巧" → **否**（单点技术）
        - "北京上海广州人口排名" → **是**（三实体聚合）

        ---

        ## 当前查询
        用户输入："{query}"

        ## 思考步骤
        1. 识别实体数量（≥2个且非属性查询？）
        2. 检测关键词（是否命中对比/流程/聚合词？） 
        3. 判断句式结构（是否含"和/与/或/先...再"？）
        4. 排除简单模式（单实体+属性/定义/推荐）

        ## 输出要求
        仅输出 **是** 或 **否**，不要解释。"""
        # 调用 llm.invoke ...
        # return True if "是" in result else False
        return True # Mock

    def _generate_subquestions_llm(self, query: str) -> List[str]:
        """子问题生成函数"""
        print(f"[Sub-question] 子问题生成: {query}")
    
        try:
            subquestion_template = """用户原始查询：{query}
            任务：将其拆分为2-5个独立子问题，需满足：
            1. 每个子问题只对应1个信息点，无重叠；
            2. 保留原始查询的核心上下文（如时间、主体、场景），不丢失关键约束；
            3. 子问题直接可用于检索（无需额外补充信息）；
            4. 不生成冗余子问题（如"对比AB"不拆"什么是A""什么是B"）。
            输出格式：按1.、2.、3.…编号列出子问题，无其他内容。"""  
        
            subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)
            response = self.llm.invoke(subquestion_prompt.format_messages(query=query))
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
            # 解析子问题列表
            sub_questions = []
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.\s*', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        sub_questions.append(question)
        
            print(f"  [Success] 生成了 {len(sub_questions)} 个子问题:")
            for i, q in enumerate(sub_questions, 1):
                print(f"    {i}. {q}")
            
            return sub_questions
        except Exception as e:
            print(f"  [Warning] 子问题生成出错: {e}")
            # 返回默认子问题列表
            return [f"关于 {query} 的相关信息"]

# 使用建议
# agent = QueryDecomposerAgent(llm_client)
# needs_decomp, reason, qs = agent.route_query("对比一下DeepSeek和ChatGPT的推理成本")