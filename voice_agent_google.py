import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, AgentSession, Agent, RoomInputOptions, AutoSubscribe
from livekit.plugins import silero, llama_index, noise_cancellation
from custom_plugins import google_tts  # Your Google TTS wrapper
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
logger = logging.getLogger("voice-assistant")

# === Set up LLM + Embeddings ===
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="gemma", request_timeout=120.0, base_url="http://127.0.0.1:11435")
Settings.llm = llm
Settings.embed_model = embed_model

# === Create or load LlamaIndex ===
PERSIST_DIR = "./chat-engine-storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# === Prewarm ===
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# === Main entrypoint ===
async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

    session = AgentSession(
        stt=silero.STT(),
        llm=llama_index.LLM(chat_engine=chat_engine),
        tts=google_tts.TTS(),
        vad=ctx.proc.userdata["vad"],
        turn_detection=silero.TurnDetector(),  # or another plugin if installed
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="You are a funny, witty assistant. Respond with short and concise answers. Avoid punctuation that is hard to pronounce or emojis."
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await session.say("Hey there! How can I help you today?", allow_interruptions=True)

# === Run the agent ===
if __name__ == "__main__":
    print("Starting voice agent...")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
