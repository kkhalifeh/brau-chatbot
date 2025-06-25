# test_local.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import time
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='chat.log', filemode='a')
logger = logging.getLogger(__name__)

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./brau_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Define prompt
prompt_template = PromptTemplate.from_template("""
You are Alyaa, a warm and professional AI assistant at Brau, a makeup studio in Dubai at Springs Souk Branch. Your goal is to engage clients conversationally, qualify leads, and encourage booking a free consultation. Today is June 24, 2025.

Conversational Flow:
1. Intro: In your FIRST response, greet warmly and ask for the client's name (e.g., "Hi! 🌸 I'm Alyaa from Brau. May I get your name?"). Use the name in subsequent responses and NEVER ask again.
2. Consultative Discovery: Ask ONE open-ended question to understand needs (e.g., "Have you done microblading before?" or "What brow look are you hoping for?"). DO NOT share prices until needs are shared.
3. Pitch One Solution: Recommend a treatment (e.g., Signature Brow) based on needs and context, share ONE USP, and offer a consultation (e.g., "Our Signature Brow is perfect for you! Want a free consultation Thursday at 3 PM?").
4. Selling/Closing: Handle objections with ONE USP (e.g., "Our premium pigments ensure no discoloration!") and use FOMO (e.g., "Slots are filling fast!"). If the client agrees to book, say: "Great! I'll pass you to our team to book!" and stop responding.
5. For unavailable details (e.g., Soft Brow price, medical conditions), say: "Let me connect you with our team—they'll sort it! 😊" and stop responding.
6. If relevant, request a brow photo: "Could you share a photo of your brows? Our artists can suggest the best treatment! 😊"

Guidelines:
- Use a warm, feminine tone with emojis only in greetings or closings (e.g., 🌸, 😊).
- Break responses into 2-3 short messages (1-2 sentences each) in plain text, no numbering or quotes.
- Use the client's name if provided.
- Avoid repeating greetings or questions already answered.

Context: {context}
Conversation History: {history}
Client Question: {question}
Alyaa's Response: Plain text, 2-3 short messages, no numbering or quotes.
""")


def format_docs(docs):
    """Format retrieved documents into context string"""
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error formatting docs: {str(e)}")
        return ""


def split_into_messages(content, max_length=300):
    """Split response into multiple messages"""
    try:
        if not isinstance(content, str):
            content = str(content)
        content = re.sub(r'^\d+\.\s*|\[\d+\]\s*|"', '', content.strip())
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            return [s[:max_length] for s in sentences]
        messages = []
        for i in range(0, len(sentences), 2):
            message = " ".join(sentences[i:i+2])[:max_length]
            messages.append(message)
            if len(messages) >= 3:
                break
        return messages or ["I'm not sure how to respond, but I can help with other topics! 😊"]
    except Exception as e:
        logger.error(f"Error splitting messages: {str(e)}")
        return ["I'm having a little trouble. How can I help you with your brows? 😊"]


def test_local_chat():
    """Main chat function"""
    conversation_history = []
    user_name = None
    print("Hi! 🌸 I'm Alyaa from Brau. May I get your name? (Type 'quit' to exit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            print("Alyaa: Thanks for chatting! Have a great day! 😊")
            break

        # Extract user name if provided
        if not user_name:
            match = re.search(r'\b(?:my name is)?\s*(\w+)\b',
                              question.lower(), re.IGNORECASE)
            if match and match.group(1).isalpha():
                user_name = match.group(1).capitalize()

        # Format conversation history
        history = "\n".join([f"{'Client' if i % 2 == 0 else 'Alyaa'}: {msg}"
                             for i, msg in enumerate(conversation_history[-4:])])

        print("\nAlyaa is typing...\n")
        try:
            # Retrieve relevant context
            docs = retriever.invoke(question)
            context = format_docs(docs)

            # Format prompt
            formatted_prompt = prompt_template.format(
                context=context,
                history=history,
                question=question
            )

            # Get LLM response
            llm_response = llm.invoke(formatted_prompt)
            response_text = llm_response.content if hasattr(
                llm_response, 'content') else str(llm_response)

            # Split into messages
            messages = split_into_messages(response_text)

            # Display messages with typing delays
            for i, message in enumerate(messages):
                time.sleep(1)
                print(f"Alyaa: {message}")
                if i < len(messages) - 1:
                    time.sleep(0.5)
                    print("\nAlyaa is typing...\n")

            # Update conversation history
            conversation_history.append(question)
            conversation_history.append("\n".join(messages))

            # Check for transfer triggers
            if "Let me connect you with our team" in "\n".join(messages):
                print("\nAlyaa: Transferring to team. Goodbye! 😊")
                break

            print()

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            print(f"\nError: {str(e)}")
            print("Alyaa: I'm having a little trouble—let's try again! 😊")


if __name__ == "__main__":
    test_local_chat()
