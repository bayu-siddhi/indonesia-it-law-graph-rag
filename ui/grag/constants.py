"""Defines settings and starters for the Graph-RAG chat application using Chainlit"""

import chainlit as cl


GRAPH_RAG_SETTINGS = [
    cl.input_widget.Select(
        id="llm_model_name", label="LLM", initial_index=0,
        values=[
            "claude-3-5-haiku-20241022",
            "llama3.1:8b-instruct-q4_K_M",
        ],
        description=(
            "Large Language Model (LLM) yang digunakan sebagai "
            "landasan sistem Graph-RAG."
        )
    ),
    cl.input_widget.Switch(
        id="stream", label="Streaming", initial=True,
        description=(
            "Apakah akan melakukan streaming respons dari LLM. "
            "Jika diaktifkan, respons akan ditampilkan secara bertahap."
        ),
    ),
]

GRAPH_RAG_STARTERS = [
    cl.Starter(
        label="Perluas Wawasan Hukum",
        message="Apa yang dapat saya lakukan jika saya merasa bahwa data pribadi saya telah disalahgunakan oleh penyelenggara sistem elektronik?",
        icon="/public/assets/idea.svg",
    ),
    cl.Starter(
        label="Baca Konten Peraturan",
        message="Apa isi pasal 25 sampai dengan 30 di UU Nomor 11 Tahun 2008?",
        icon="/public/assets/learn.svg",
    ),
    cl.Starter(
        label="Jelajahi Data Peraturan",
        message="Apa pasal yang paling banyak dirujuk di UU Nomor 11 Tahun 2008?",
        icon="/public/assets/terminal.svg",
    ),
]
