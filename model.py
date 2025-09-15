from llama_cpp import Llama
import json
# 加载gguf模型（路径为相对路径）
model_path = "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)  # n_gpu_layers根据显卡设置
# 输入数据
input_data = {    
"human": {        
"sex": "男",        
"age": 45,        
"height": 175,        
"weight": 65,        
"blood_pressure": "120/80",        
"heart_rate":87,        
"blood_glucose":6.4,        
"history": ["asthma"],        
"work_place": "室内"    
},    
"env": {        
"aqi": 150,        
"temperature": 25,        
"humidity": 60,        
"wind_speed":1.6,        
"noise_intensity": 40,        
"light_intensity":20171,        
"pullutant_level": "moderate",    
}
}
# 构造提示（转义{ 和 } 为 {{ 和 }}）
prompt = f"""
你是一个健康专家，基于以下人体和气象数据，生成健康评分（0-100）、运动建议（类型、强度、时长、地点）、饮食建议（热量、具体推荐）和风险评估（疾病风险概率）。输出为JSON格式，并附上推理原因。严格遵循约束：
- 如果有哮喘且AQI>100，运动必须室内且强度低。
- 如果血压>140/90，限制运动强度，避免高强度。
- BMI>30（肥胖）时，饮食推荐低脂。
- 提供解释（reason字段）。
数据：
{json.dumps(input_data, indent=2)}
"""
# 推理
output = llm(prompt, max_tokens=512, temperature=0.7, top_p=0.9)
response = output["choices"][0]["text"]
# 提取JSON输出
import re
json_match = re.search(r'\{.*\}', response, re.DOTALL)
if json_match:
    output_json = json.loads(json_match.group(0))    
    print(json.dumps(output_json, indent=2, ensure_ascii=False))
else:    
    print("未找到有效JSON输出，请调整提示。")