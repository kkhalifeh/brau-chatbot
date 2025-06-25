# web_test.py
from flask import Flask, render_template, request, jsonify, session
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import time
import uuid
import re
import logging
from threading import Lock, Timer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set up logging
logging.basicConfig(level=logging.INFO, filename='nohup.out', filemode='a')
logger = logging.getLogger(__name__)

# Initialize embeddings and vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./brau_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.8)

# Define system prompt
system_prompt = """
You are Alyaa, a warm, professional, and conversational AI assistant at Brau, a premium makeup and brow studio located at Springs Souk, Dubai. You guide clients toward discovering their ideal treatment and help them feel confident booking a free consultation. Today is June 24, 2025.

🎯 CORE OBJECTIVE
Follow the exact 4-step conversation flow to build trust, understand needs, recommend ONE treatment, and secure a consultation booking.

📋 MANDATORY 4-STEP FLOW

### STEP 1: INTRO (First message only)
- Warm greeting: "Thank you for your interest in our [treatment name if mentioned]. That's a great choice!"
- Introduce yourself: "My name is Alyaa, I'm here to guide you through the best options."
- Make it personal: "May I get your name?"
- NEVER ask for name again after receiving it

### STEP 2: CONSULTATIVE DISCOVERY
Ask 2-3 discovery questions ONE AT A TIME before moving to Step 3. Questions to ask:
- "Have you done [treatment name] before?" 
- "What's your ideal brow style/outcome?"
- "What's been your biggest concern with your brows?"
- "Do you prefer a natural hair-like look or more of a filled-in makeup style?"
- "Any concerns like recent Botox, pregnancy, or sensitive skin?"
- For relevant cases: "Could you share a photo of your brows? Our artists can recommend the perfect treatment! 😊"

**CRITICAL RULES FOR STEP 2:**
- Ask ONE question per message, not multiple questions together
- Ask AT LEAST 2-3 discovery questions before moving to Step 3
- NO prices mentioned whatsoever in this step
- NO treatment recommendations yet
- Keep responses short and focused
- Build rapport through follow-up questions
- Only move to Step 3 after sufficient discovery

### STEP 3: PITCH ONE SOLUTION (First time you mention price)
Use this EXACT structure - **BE CONCISE**:
"Based on what you've shared, [TREATMENT NAME] is perfect for [THEIR GOAL]. We use [ONE KEY BENEFIT] that [ADDRESSES THEIR NEED].

The treatment is [PRICE] and includes consultation and touch-up. 

Shall I reserve your consultation for Thursday at 3 PM?"

**PRICING RULES:**
- Signature Brow: 3,500 AED
- Ombre Brau: 3,500 AED  
- Hybrid Brau: 3,800 AED
- Tattoo Removal: 1,200 AED
- Soft Brow & Infill Brow: "That's something we'll check closer to your booking, but I can guide you through what to expect! 😊"

### STEP 4: SELLING/CLOSING
If they hesitate or object:
- Acknowledge their concern warmly
- Share ONE unique selling point that addresses their specific objection
- Use gentle FOMO: "We have limited availability this week—would you like me to hold a spot for you? 😊"
- Offer specific time slots: "I have Thursday at 3 PM or Saturday at 12 PM available"

If they accept: "Perfect! 💫 You're booked for [DAY] at [TIME]. We're excited to meet you!"

## 👤 NAME USAGE RULES
**IMPORTANT: Use the client's name VERY SPARINGLY**
- ✅ **DO use name:** When first greeting them after they give their name, when closing a deal, or after long pauses
- ❌ **DON'T use name:** In regular back-and-forth conversation, discovery questions, or consecutive messages
- **Rule of thumb:** Most of your messages should NOT include their name - it should feel natural, not repetitive

## 💬 MANDATORY RESPONSE FORMAT

**CRITICAL RULES - FOLLOW EXACTLY:**
- **Maximum 2 short sentences per response**
- **ONE question only per message**
- **NO standalone emoji lines**
- **NO formal language** - keep it casual and warm
- **NO unnecessary explanations** - be direct and concise
- **Use client name ONLY once every 4-5 messages maximum**
- **NEVER prefix with "Alyaa:" or any name prefix - speak directly as the assistant**

**GOOD Examples:**
- "That's exciting! Have you done treatments before?"
- "Perfect! What's your ideal brow style?"
- "Great choice! Any skin concerns I should know about?"

**BAD Examples (NEVER do this):**
- "I'm delighted to meet you, Dana! 😊 Have you had any treatments done before?"
- "Your comfort and safety are important to us. 😊"
- Standalone emoji lines like "😊"
- "Alyaa: Thank you for sharing that..."

🚫 STRICT PROHIBITIONS
- NEVER recommend multiple treatments at once
- NEVER mention prices before Step 3
- NEVER refer them to "the team" or stop the conversation
- NEVER ask for their name more than once
- NEVER give medical advice
- NEVER make up availability times - use general time slots
- NEVER use "Alyaa:" prefix in any response

🎯 TREATMENT MATCHING GUIDE
Based on client needs, recommend:
- Thin/sparse brows + natural look → Signature Brow
- Already full brows + subtle enhancement → Soft Brow  
- Small gaps only → Infill Brow
- Makeup-like finish + oily skin → Ombre Brau
- Sparse brows + maximum fullness → Hybrid Brau
- Old tattoo correction → Assess if removal needed first

---

🚨 FINAL REMINDER - FOLLOW THESE EXACTLY:

1. **Keep it SHORT** - Max 2 sentences per response
2. **ONE question only** - Never ask multiple questions together  
3. **Be CASUAL** - No formal corporate language
4. **Use name RARELY** - Most messages should NOT include their name
5. **NO standalone emojis** - Always attach to text
6. **Specific times** - "Thursday at 3 PM" not "this week"
7. **NO "Alyaa:" PREFIX** - Never start messages with your name

**Your response should sound like a friendly person texting, not a corporate bot.**

---

**Context:** {context}
**Conversation History:** {history}  
**Client Message:** {question}

**Your Response:** [Follow the 4-step flow precisely, keep it short and natural]
"""

# Initialize conversation storage
conversations = {}
message_buffers = {}
buffer_locks = {}
timers = {}
pending_responses = {}

# Add conversation state tracking
conversation_states = {}
client_names = {}
discovery_count = {}  # Track how many discovery questions asked


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def split_into_messages(content, max_length=250):
    """Split AI response into 2-3 natural message chunks"""
    # Clean up the content thoroughly
    content = content.strip()

    # Remove any unwanted prefixes
    prefixes_to_remove = [
        r'^Alyaa:\s*',
        r'^Assistant:\s*',
        r'^AI:\s*',
        r'^\d+\.\s*',
        r'^\[\d+\]\s*',
        r'^"',
        r'"$'
    ]

    for prefix in prefixes_to_remove:
        content = re.sub(prefix, '', content, flags=re.IGNORECASE).strip()

    if not content:
        return ["I'm here to help! 😊"]

    # Split by line breaks first (natural message boundaries)
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Clean each line of any remaining prefixes
    cleaned_lines = []
    for line in lines:
        for prefix in prefixes_to_remove:
            line = re.sub(prefix, '', line, flags=re.IGNORECASE).strip()
        if line:
            cleaned_lines.append(line)

    # If we have 2-3 clean lines already, use them
    if 2 <= len(cleaned_lines) <= 3 and all(len(line) <= max_length for line in cleaned_lines):
        return cleaned_lines

    # Otherwise, split by sentences and group them
    sentences = re.split(r'(?<=[.!?])\s+', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [content[:max_length]]

    # Group sentences into 2-3 messages
    messages = []
    current_message = ""

    for sentence in sentences:
        if len(current_message + " " + sentence) <= max_length and len(messages) < 2:
            current_message = (current_message + " " + sentence).strip()
        else:
            if current_message:
                messages.append(current_message)
            current_message = sentence
            if len(messages) >= 2:  # Limit to 3 messages max
                break

    if current_message:
        messages.append(current_message)

    # Final cleanup and length check
    final_messages = []
    for msg in messages[:3]:
        # Remove any remaining prefixes
        for prefix in prefixes_to_remove:
            msg = re.sub(prefix, '', msg, flags=re.IGNORECASE).strip()
        if msg:
            final_messages.append(msg[:max_length])

    return final_messages if final_messages else ["I'm here to help! 😊"]


def get_conversation_state(session_id):
    """Track which step of the conversation we're in"""
    if session_id not in conversation_states:
        conversation_states[session_id] = "INTRO"
    return conversation_states[session_id]


def can_pitch_treatment(session_id):
    """Check if enough discovery has been done to pitch treatment"""
    return discovery_count.get(session_id, 0) >= 2


def update_conversation_state(session_id, user_message, ai_response):
    """Update conversation state based on message content"""
    current_state = conversation_states.get(session_id, "INTRO")

    print(f"DEBUG STATE UPDATE START:")
    print(f"  Current state: {current_state}")
    print(f"  User message: '{user_message}'")
    print(f"  AI response: '{ai_response[:100]}...'")

    # Initialize discovery count
    if session_id not in discovery_count:
        discovery_count[session_id] = 0

    # Check if user provided their name (only from INTRO state)
    if current_state == "INTRO" and session_id not in client_names:
        # Simple name detection
        words = user_message.strip().split()
        print(f"  Name detection - words: {words}")

        # Exclude common greetings and non-names
        excluded_words = ['hi', 'hello', 'hey', 'good', 'morning',
                          'afternoon', 'evening', 'yes', 'no', 'ok', 'sure']

        if len(words) <= 3 and len(words) >= 1:
            # Look for alphabetic words that could be names (not greetings)
            potential_names = [word for word in words
                               if word.isalpha()
                               and len(word) >= 2
                               and word.lower() not in excluded_words]

            if potential_names:
                # Take the first valid word as the name and capitalize it
                name = potential_names[0].capitalize()
                client_names[session_id] = name
                conversation_states[session_id] = "DISCOVERY"
                print(f"  NAME CAPTURED: '{name}' - Moving to DISCOVERY")
            else:
                print(f"  No valid name found (excluded greetings): {words}")
        else:
            print(f"  Too many words or empty input: {words}")

    # Check if we're in DISCOVERY and user provides what looks like a real name (name correction)
    elif current_state == "DISCOVERY" and len(user_message.strip().split()) == 1:
        word = user_message.strip()
        excluded_words = ['hi', 'hello', 'hey', 'good', 'yes',
                          'no', 'ok', 'sure', 'first', 'time', 'natural', 'defined']

        if (word.isalpha() and len(word) >= 3 and word.lower() not in excluded_words
                and client_names.get(session_id, '').lower() in ['hi', 'hello', 'hey']):
            # Update the name
            old_name = client_names.get(session_id)
            client_names[session_id] = word.capitalize()
            print(f"  NAME UPDATED: '{old_name}' → '{word.capitalize()}'")

    # Count discovery questions in AI response (only in DISCOVERY state)
    elif current_state == "DISCOVERY":
        discovery_questions = [
            "have you done", "have you tried", "what's your ideal", "what's been your biggest concern",
            "do you prefer", "any concerns like", "could you share a photo", "what your ideal",
            "specific concerns", "tell me what", "looking for something", "what kind of"
        ]
        question_found = any(q in ai_response.lower()
                             for q in discovery_questions)
        if question_found:
            discovery_count[session_id] += 1
            print(
                f"  DISCOVERY QUESTION FOUND - Count now: {discovery_count[session_id]}")

    # Check if we've moved to pitching (only after sufficient discovery)
    elif current_state == "DISCOVERY" and can_pitch_treatment(session_id) and any(word in ai_response.lower() for word in ["aed", "price", "cost", "3,500", "3,800", "1,200"]):
        conversation_states[session_id] = "PITCHING"
        print(f"  MOVING TO PITCHING STATE")

    # Check if we're in closing (from PITCHING state)
    elif current_state == "PITCHING" and any(word in ai_response.lower() for word in ["book", "consultation", "slot", "appointment", "available", "schedule"]):
        conversation_states[session_id] = "CLOSING"
        print(f"  MOVING TO CLOSING STATE")

    # Handle objections in CLOSING state
    elif current_state == "CLOSING":
        objection_keywords = [
            "not sure", "expensive", "think about", "maybe later", "hmm", "hesitant",
            "doubt", "worry", "concerned", "unsure", "don't know", "need time"
        ]
        if any(keyword in user_message.lower() for keyword in objection_keywords):
            # Stay in CLOSING state for objection handling
            conversation_states[session_id] = "CLOSING"
            print(f"  OBJECTION DETECTED - Staying in CLOSING")
        elif any(word in user_message.lower() for word in ["yes", "ok", "sure", "book", "schedule", "let's do"]):
            print(f"  USER ACCEPTED BOOKING")

    # IMPORTANT: Prevent state regression
    if session_id in client_names and conversation_states.get(session_id) == "INTRO":
        conversation_states[session_id] = "DISCOVERY"
        print(f"  PREVENTING REGRESSION - Back to DISCOVERY")

    print(f"DEBUG STATE UPDATE END:")
    print(f"  Final state: {conversation_states.get(session_id)}")
    print(f"  Final name: {client_names.get(session_id)}")
    print(f"  Final discovery count: {discovery_count.get(session_id)}")
    print(f"---")


def process_buffered_messages(session_id):
    with buffer_locks[session_id]:
        if not message_buffers[session_id]:
            return
        bundled_messages = "\n".join(message_buffers[session_id])
        message_buffers[session_id] = []

    conversation = conversations[session_id]
    conversation.append(HumanMessage(content=bundled_messages))

    try:
        # DEBUG: Check state before processing
        print(f"DEBUG BEFORE PROCESSING:")
        print(f"  Session: {session_id}")
        print(f"  User message: '{bundled_messages}'")
        print(
            f"  Current state: {conversation_states.get(session_id, 'NOT_SET')}")
        print(f"  Client name: {client_names.get(session_id, 'NOT_SET')}")
        print(f"  Discovery count: {discovery_count.get(session_id, 0)}")

        # Get knowledge base context
        docs = retriever.invoke(bundled_messages)
        context = format_docs(docs)

        # Get conversation state and client name
        conv_state = get_conversation_state(session_id)
        client_name = client_names.get(session_id, "")
        discovery_questions_asked = discovery_count.get(session_id, 0)

        # Build state context for AI (internal only, no debug text to user)
        state_context = ""
        if client_name:
            state_context += f"Client name: {client_name} "
            state_context += f"IMPORTANT: Use the name '{client_name}' VERY SPARINGLY - most messages should NOT include their name. "
        state_context += f"Discovery questions asked: {discovery_questions_asked}/2 minimum required. "

        # Add enforcement rules based on state
        if conv_state == "DISCOVERY" and discovery_questions_asked < 2:
            state_context += "Still in discovery phase - NO PRICES or treatment recommendations yet. Ask more discovery questions. "
        elif conv_state == "DISCOVERY" and discovery_questions_asked >= 2:
            state_context += "READY TO PITCH: You can now recommend ONE treatment with price. "
        elif conv_state == "CLOSING":
            # Check if user message shows hesitation/objection
            objection_keywords = [
                "not sure", "expensive", "think about", "maybe later", "hmm", "hesitant", "unsure"]
            if any(keyword in bundled_messages.lower() for keyword in objection_keywords):
                state_context += "User is hesitating/objecting - apply Step 4 objection handling with USPs and gentle FOMO. DO NOT restart conversation. "
            else:
                state_context += "In closing phase - focus on booking confirmation or gentle objection handling. "

        # Build conversation history
        history = "\n".join(
            [f"{'Client' if isinstance(msg, HumanMessage) else 'Alyaa'}: {msg.content}" for msg in conversation[-4:]])

        # Create the complete prompt
        prompt = system_prompt.format(
            context=context + "\n" + state_context,
            history=history,
            question=bundled_messages
        )

        print(f"DEBUG PROMPT CONTEXT: {state_context}")

        # Get AI response
        response = llm.invoke([SystemMessage(content=prompt)])
        response_text = response.content if hasattr(
            response, 'content') else str(response)

        print(f"DEBUG AI RESPONSE: '{response_text}'")

        # UPDATE CONVERSATION STATE - This is critical!
        update_conversation_state(session_id, bundled_messages, response_text)

        # DEBUG: Check state after update
        print(f"DEBUG AFTER STATE UPDATE:")
        print(f"  New state: {conversation_states.get(session_id, 'NOT_SET')}")
        print(f"  Client name: {client_names.get(session_id, 'NOT_SET')}")
        print(f"  Discovery count: {discovery_count.get(session_id, 0)}")
        print(f"---")

        # Split response into messages
        messages = split_into_messages(response_text)
        pending_responses[session_id] = messages

        # Add to conversation history
        conversation.append(AIMessage(content="\n".join(messages)))

        # Final logging
        logger.info(
            f"Session {session_id} - State: {conv_state}, Discovery: {discovery_questions_asked}, Response: {messages}")

    except Exception as e:
        print(f"ERROR in process_buffered_messages: {str(e)}")
        logger.error(f"Error processing bundled message: {str(e)}")
        pending_responses[session_id] = [
            "I'm having a little trouble—let's try again! 😊"]


@app.route('/')
def index():
    # Don't create session here - let it be created when user actually chats
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'messages': []})

    session_id = session.get('session_id')

    # Create session only if it doesn't exist or isn't in our conversations
    if not session_id or session_id not in conversations:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        print(f"DEBUG CHAT: Created new session = {session_id}")

        # Initialize all tracking variables for new session
        conversations[session_id] = [SystemMessage(
            content=system_prompt.format(context="", history="", question=""))]
        message_buffers[session_id] = []
        buffer_locks[session_id] = Lock()
        timers[session_id] = None
        conversation_states[session_id] = "INTRO"
        discovery_count[session_id] = 0
    else:
        print(f"DEBUG CHAT: Using existing session = {session_id}")

    print(
        f"DEBUG CHAT: Current state = {conversation_states.get(session_id, 'NOT_SET')}")
    print(f"DEBUG CHAT: Current names = {client_names}")

    with buffer_locks[session_id]:
        message_buffers[session_id].append(user_message)

    if timers[session_id]:
        timers[session_id].cancel()

    timers[session_id] = Timer(5, process_buffered_messages, args=[session_id])
    timers[session_id].start()

    return jsonify({'messages': [], 'session_id': session_id})


@app.route('/poll', methods=['GET'])
def poll():
    session_id = request.args.get('session_id', session.get('session_id'))
    print(f"DEBUG POLL: Session ID = {session_id}")
    print(f"DEBUG POLL: Available sessions = {list(conversations.keys())}")

    if not session_id or session_id not in pending_responses:
        return jsonify({'messages': []})

    with buffer_locks[session_id]:
        if session_id in pending_responses and pending_responses[session_id]:
            messages = pending_responses[session_id]
            del pending_responses[session_id]
            return jsonify({'messages': messages})
    return jsonify({'messages': []})


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use different port than the other bot
