import gradio as gr
from model import LLM

llm = LLM()


def echo(message, history):
    return llm.generate(message, history)


def main() -> None:
    demo = gr.ChatInterface(
        fn=echo,
        title="Aleph Alpha Pharia",
        multimodal=False,
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
