from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from modelscope.utils.constant import Tasks
import gradio

_model_path = "./cache/qwen/Qwen2-0___5B-Instruct"

_model = AutoModelForCausalLM.from_pretrained(_model_path,device_map="cuda")
_tokenizer = AutoTokenizer.from_pretrained(_model_path)

_pp = pipeline(task=Tasks.text_generation,model=_model,tokenizer=_tokenizer)

gradio.Interface.from_pipeline(_pp).launch(server_name="0.0.0.0",server_port=60001, share=True)