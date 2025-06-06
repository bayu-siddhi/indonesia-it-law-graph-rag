import chainlit as cl


GRAPH_RAG_DESC = """
### ⚙️ Graph-RAG

> **Graph-RAG** adalah sistem tanya-jawab cerdas yang menggabungkan kekuatan **Large Language Model (LLM)** dan **Graph Database** untuk menghasilkan jawaban yang lebih **akurat**, **kontekstual**, dan **terhubung**. Dengan memanfaatkan **LLM** dan **graf pengetahuan**, Graph-RAG mampu memahami pertanyaan kompleks serta menelusuri hubungan antar data secara mendalam.

🎯 **Coba sekarang!** Ajukan pertanyaanmu dan temukan jawaban cerdas dari Graph-RAG.
""".strip()

GRAPH_RAG_SETTINGS = [
    # TOOD: Tentukan Setting nya
    cl.input_widget.Select(
        id="llm_model",
        label="LLM Model",
        tooltip="Model LLM yang akan digunakan sebagai landasan Graph-RAG",
        initial_index=1,
        values=[
            "gemini-2.0-flash",
            "llama3.1:8b-instruct-q4_0"
        ]
    )
]

GRAPH_RAG_STARTERS = [
    # TOOD: Tentukan daftar Starter nya
    cl.Starter(
        label="Morning routine ideation",
        message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
        icon="/public/idea.svg",
        ),

    cl.Starter(
        label="Explain superconductors",
        message="Explain superconductors like I'm five years old.",
        icon="/public/learn.svg",
        ),
    cl.Starter(
        label="Python script for daily email reports",
        message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
        icon="/public/terminal.svg",
        ),
    cl.Starter(
        label="Text inviting friend to wedding",
        message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
        icon="/public/write.svg",
        )
]