import requests
import json

url = 'http://127.0.0.1:5000/process_knowledge_graph'
data = {
    "knowledge_graph": ["细胞", "进行", "减数分裂"],
    "question": "请解释细胞如何进行减数分裂？"
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
