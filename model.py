from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, GenerationConfig
import torch


class LLM:
    def __init__(self) -> None:
        self.model_id = "Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self.gen_cfg = self.__init_generation_config()

    def __init_generation_config(self) -> GenerationConfig:
        gen_cfg = GenerationConfig.from_pretrained(self.model_id)
        gen_cfg.max_new_tokens = 250
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        gen_cfg.begin_suppress_tokensrepetition_penalty = 5
        gen_cfg.no_repeat_ngram_size = 3
        gen_cfg.do_sample = True
        gen_cfg.top_k = 90
        gen_cfg.num_beams = 3
        return gen_cfg

    def embed_message(self, message: str, history: list[list[str]]) -> str:
        # System
        prompt = """
        <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
                You are a helpful assistant named Pharia. You give engaging, well-structured answers to user inquiries.<|eot_id|>
        """

        # Chat history
        for dialog in history[-4:]:
            prompt += f"""
            <|start_header_id|>user<|end_header_id|>
                {dialog[0]}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
                {dialog[1]}
            """
        # New message
        prompt += f"""
            <|start_header_id|>user<|end_header_id|>
                {message}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
        return prompt

    def generate(self, message: str, history: list[list[str]]) -> str:
        prompt = self.embed_message(message, history)
        input_ids = self.tokenizer(
            prompt, return_token_type_ids=False, return_tensors="pt"
        ).input_ids.to(self.device)
        outputs = self.model.generate(input_ids, generation_config=self.gen_cfg)
        return self.tokenizer.decode(
            outputs[0][len(input_ids[0]) :], skip_special_tokens=True
        )
