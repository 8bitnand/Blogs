from search import SemanticSearch, GoogleSearch, Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
import yaml
import torch
import nltk

def load_configs(config_file: str) -> dict:
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    return configs


class RAGModel:
    def __init__(self, configs) -> None:
        self.configs = configs
        self.device = configs["model"]["device"]
        model_url = configs["model"]["genration_model"]
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            torch_dtype=torch.float16,
            # quantization_config=quantization_config,
            low_cpu_mem_usage=False,
            attn_implementation="sdpa",
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_url,
        )

    def create_prompt(self, query, topk_items: list[str]):

        context =  "\n-".join(c for c in topk_items)

        base_prompt = f"""You are an alternate to goole search. Your job is to answer the user query in as detailed manner as possible.
        you have access to the internet and other relevent data related to the user's question.
        Give time for yourself to read the context and user query and extract relevent data and then answer the query.
        make sure your answers is as detailed as posssbile. 
        Do not return thinking process, just return the answer.
        Give the output structured as a Wikipedia article.
        Now use the following context items to answer the user query
        context: {context}
        user query : {query} 
        """

        dialog_template = [{"role": "user", "content": base_prompt}]

        prompt = self.tokenizer.apply_chat_template(
            conversation=dialog_template, tokenize=False, add_feneration_prompt=True
        )
        return prompt

    def answer_query(self, query: str, topk_items: list[str]):

        prompt = self.create_prompt(query, topk_items)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**input_ids, temperature=0.7, max_new_tokens=512, do_sample=True)
        text = self.tokenizer.decode(output[0])
        text = text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")


        return text

if __name__ == "__main__":
    configs = load_configs(config_file="rag.configs.yml")
    query = "The height of burj khalifa is 1000 meters and it was built in 2023. What is the height of burgj khalifa"
    # g = GoogleSearch(query)
    # data = g.all_page_data
    # d = Document(data, 512)
    # doc_chunks = d.doc()
    # s = SemanticSearch(doc_chunks, "all-mpnet-base-v2", "mps")
    # topk, u = s.semantic_search(query=query, k=32)
    r = RAGModel(configs)
    output = r.answer_query(query=query, topk_items=[""])
    print(output)
