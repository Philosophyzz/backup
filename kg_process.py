from flask import Flask, request, jsonify
import json
import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate, Settings

# 初始化Flask应用
app = Flask(__name__)

# 初始化 LLM 模型
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.1,
        "do_sample": True,
    },
    tokenizer_name=r'D:\models\Qwen2-7B-Instruct',
    model_name=r'D:\models\Qwen2-7B-Instruct',
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},
)
Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
    model_name=r"D:\models\bge-large-zh-v1.5"
)

# 加载llama-index的存储上下文和索引
storage_context = StorageContext.from_defaults(persist_dir=r"D:\sxr\elearnPJ\data\elearn_index")
index = load_index_from_storage(storage_context)

# 定义系统提示词 (system prompt)
system_prompt = PromptTemplate(
    template="""你是一个精通物理生物的检索专家，请根据给定的知识图谱，返回最相关的文档内容
    注意：回答中只需包括检索内容，不许包含任何多余的话"""
)

# 构建查询引擎
retriever = index.as_retriever(retriever_mode='embedding')

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    system_prompt=system_prompt  # 加入系统提示词
)

# TODO：效果不是很好，该怎么利用KG呢
@app.route('/process_knowledge_graph', methods=['POST'])
def process_knowledge_graph():
    try:
        data = request.get_json()
        print(f"Received data: {data}")  # 调试打印请求数据
        
        # 获取用户传递的知识图谱和提问
        knowledge_graph_content = data.get('knowledge_graph', [])
        question = data.get('question', '')

        # 校验提问和知识图谱
        if not knowledge_graph_content or not isinstance(knowledge_graph_content, list):
            return jsonify({"error": "Knowledge graph content not provided or in wrong format"}), 400

        if not question:
            return jsonify({"error": "Question not provided"}), 400

        related_content = []
        torch.cuda.empty_cache()

        # 遍历实体并过滤掉无效或空的实体
        for entity in knowledge_graph_content:
            if not entity.strip():  # 跳过空字符串
                continue
            
            content = "请根据给定的知识图谱，返回最相关的文档内容："
            response = query_engine.query(content + entity)  # 根据知识图谱中的实体进行查询
            if response:  # 确保返回的结果有效
                related_content.append(str(response))  # 收集相关的nodes

        if not related_content:
            return jsonify({"error": "No relevant nodes found"}), 404

        system_prompt = PromptTemplate(
            template="""你是一个精通物理生物的检索专家，请根据给定的相关内容，回答用户的提问，回答不要换行
            注意：
            1. 索引内容可能会出现有文字没图片的情况，此时为了让用户看懂回答，请不要回答图片内容，比如 “图22所示” 这种表述不要出现，请用文字表述清楚，最后在进行一遍检查后，确定文字表述清晰，在进行回答
            2. 如索引内容不足以支持回答，请回答内容不足，无法回答。"""
        )

        # print(f"Related nodes: {related_content}")
        final_answer = generate_answer_from_related_content(related_content, question)

        return jsonify({
            "related_content": related_content,
            "final_answer": final_answer
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # 打印错误信息
        return jsonify({"error": str(e)}), 500


def generate_answer_from_related_content(related_content, question):
    """
    调用对话模型生成回答，接收相关的文档内容和用户的提问作为输入
    """
    nodes_combined = " ".join(related_content)
    
    # 在生成的 prompt 中包含用户的提问
    prompt = f"你是一个精通物理生物的教育专家，用户提问：'{question}'，请根据以下相关内容进行回答：{nodes_combined}"

    # 调试打印 prompt
    print(f"Prompt: {prompt}")

    try:
        response = query_engine.query(prompt)
        # if response is None or not response.strip():
        #     raise ValueError("Generated response is empty or None")
        print(f"Answer:" + str(response))
        return str(response)
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Error generating response"


# 启动Flask应用
if __name__ == '__main__':
    app.run(debug=False)
