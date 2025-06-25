# chat.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import re

load_dotenv()

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./brau_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Define system prompt
system_prompt = """
You are Alyaa, a warm and professional AI assistant at Brau, a makeup studio in Dubai at Springs Souk Branch. Your goal is to engage clients conversationally, qualify leads, and encourage booking a free consultation. Today is June 24, 2025.

Conversational Priorities:
1. In your FIRST response, greet warmly and ask for the client's name (e.g., "Hi! 🌸 I'm Alyaa from Brau. May I get your name?"). For subsequent responses, use the client's name if provided and NEVER ask for it again.
2. DO NOT share prices until the client shares their needs (e.g., ask "Have you done microblading before?" or "What brow look are you hoping for?").
3. Recommend a specific treatment (e.g., Signature Brow) based on the context and offer a free consultation (e.g., "I can book you a free consultation—how about Thursday at 3 PM?").
4. Handle objections with USPs (e.g., "Our premium pigments ensure no discoloration!").
5. If asked about unavailable details (e.g., Soft Brow price, medical conditions), say: "Let me connect you with our team—they’ll sort it! 😊" and stop responding.
6. Return responses as plain text, broken into 2-3 short messages (1-2 sentences each) for a WhatsApp-like flow. Do NOT use numbering, quotes, or structured formats like lists or JSON.

Context: {context}
Conversation History: {history}
Client Question: {question}
Alyaa's Response: Plain text, 2-3 short messages, no numbering or quotes.
"""

# Function to format documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to split response into messages


def split_into_messages(text, max_messages=3, max_chars_per_message=300):
    # Remove numbering and quotes
    text = re.sub(r'^\d+\.\s*|\[\d+\]\s*|"', '', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_messages:
        return [s[:max_chars_per_message] for s in sentences]
    messages = []
    message_count = min(max_messages, 3)
    sentences_per_message = max(1, len(sentences) // message_count)
    for i in range(message_count):
        start_idx = i * sentences_per_message
        end_idx = start_idx + sentences_per_message if i < message_count - \
            1 else len(sentences)
        message = " ".join(sentences[start_idx:end_idx])
        if len(message) > max_chars_per_message:
            message = message[:max_chars_per_message].rsplit(' ', 1)[0] + "..."
        messages.append(message)
    return messages or ["I'm not sure how to respond, but I can help with other topics! 😊"]


# Initialize conversation history
conversation = [SystemMessage(content=system_prompt.format(
    context="", history="", question=""))]
user_name = None

print("Hi! 🌸 I'm Alyaa from Brau. May I get your name? (Type 'quit' to exit)")

# Chat loop
while True:
    question = input("You: ")
    if question.lower() in ["quit", "exit"]:
        print("Alyaa: Thanks for chatting! Have a great day! 😊")
        break

    # Update user name if provided
    if not user_name and "my name is" in question.lower():
        match = re.search(r'my name is (\w+)', question.lower())
        if match:
            user_name = match.group(1).capitalize()

    # Retrieve context
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Format history
    history = "\n".join(
        [f"{'Client' if isinstance(msg, HumanMessage) else 'Alyaa'}: {msg.content}" for msg in conversation[-4:]])

    # Generate response
    prompt = system_prompt.format(
        context=context, history=history, question=question)
    conversation.append(HumanMessage(content=question))
    response = llm.invoke([SystemMessage(content=prompt)])
    response_text = response.content
    messages = split_into_messages(response_text)

    # Print response
    for msg in messages:
        print(f"Alyaa: {msg}")

    conversation.append(AIMessage(content="\n".join(messages)))
    print()
