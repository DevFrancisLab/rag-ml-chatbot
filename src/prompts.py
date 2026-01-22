from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an expert Data Scientist assistant.\n"
    "Use the provided context to answer the question.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Keep the answer under three sentences.\n\n"
    "Context:\n{context}"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}")
    ]
)