import os
import json
import re
import shutil
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings, 
    load_index_from_storage,
    Document
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import gradio as gr
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SimpleNodeParser
import fitz
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
import os
# Rimuovi: from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import os
from google import genai





# from llama_index.core.query_engine import VectorIndexQueryEngine

# ======================================================
# 1. CONFIGURAZIONE LLM + EMBEDDINGS
# ======================================================

# --- CONFIGURAZIONE ---
MIA_CHIAVE = "xxx"

# 2. SEGUIAMO LA DOCUMENTAZIONE: Impostiamo la variabile d'ambiente corretta
client = genai.Client(api_key=MIA_CHIAVE)
os.environ["GOOGLE_API_KEY"] = MIA_CHIAVE

print("--- AVVIO ADAI IN CORSO ---")

try:
    # Verifica modelli (Nuovo comando 2026)
    print("Verifica connessione...")
    for m in client.models.list():
        if "gemini" in m.name:
            print(f"-> Modello pronto: {m.name}")
    
    # Impostazione per LlamaIndex (Lettura PDF)
    Settings.llm = Gemini(
        model="models/gemini-3.1-flash-lite-preview",
        api_key=MIA_CHIAVE
    )
    print("✓ Connessione riuscita. Caricamento documenti...")

except Exception as e:
    print(f"ERRORE: {e}")

# Modello multilingua per capire l'italiano correttamente
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

USERS_FILE = "users_mock.json"
UPDATES_FILE = "history/knowledge_updates.json"
PERSIST_DIR = "./storage"
DOCS_DIR = "docs"
os.makedirs("history", exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ======================================================
# 2. FUNZIONI DI SUPPORTO E PULIZIA
# ======================================================

def pulisci_testo(testo):
    # Rimuove sequenze di puntini (indici) e compatta gli spazi
    testo = re.sub(r'\.{3,}', ' ', testo)
    testo = re.sub(r'\s+', ' ', testo)
    return testo.strip()

def carica_utenti():
    if not os.path.exists(USERS_FILE):
        mock_data = {
            "RSSMRA80A01H501W": {"password": "pass", "ruolo": "paziente", "nome": "Mario Rossi"},
            "admin@policlinico.it": {"password": "root", "ruolo": "staff", "nome": "Admin Sistema"}
        }
        with open(USERS_FILE, "w") as f:
            json.dump(mock_data, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def anonimizza_input(testo):
    testo = re.sub(r'[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]', '[CF-OSCURATO]', testo)
    testo = re.sub(r'\b\d{10}\b|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b', '[TEL-OSCURATO]', testo)
    return testo

# ======================================================
# 3. MOTORE DI INDICIZZAZIONE (NORMALIZZATO)
# ======================================================

# Cartella dei PDF
DOCS_DIR = "docs"
TXT_OUTPUT = "estratti.txt"

with open(TXT_OUTPUT, "w", encoding="utf-8") as f_out:
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIR, file)
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # pulizia base
                        text_clean = re.sub(r'\.{3,}', ' ', text)
                        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
                        # scrivo su file
                        f_out.write(f"--- {file} Page {i+1} ---\n")
                        f_out.write(text_clean + "\n\n")

print(f"✅ Testo estratto salvato in {TXT_OUTPUT}")

# --- DEBUG: Verifica se 'bancomat' è nel TXT ---
with open(TXT_OUTPUT, "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = list(re.finditer(r".{0,30}ban.{0,30}", text.lower()))
if matches:
    print(f"✅ Trovate {len(matches)} occorrenze simili a 'bancomat' nel TXT:")
    for m in matches[:5]:  # stampiamo le prime 5 occorrenze
        print("Simile trovato:", repr(m.group()))
else:
    print("❌ Nessuna occorrenza di 'bancomat' trovata nel TXT")

def inizializza_indice():
    os.makedirs(PERSIST_DIR, exist_ok=True)  # CREA CARTELLA STORAGE

    if not os.listdir(PERSIST_DIR):
        print("🔨 LOG: Creazione nuovo indice da estratti.txt...")

        # Leggo il TXT e pulisco
        with open(TXT_OUTPUT, "r", encoding="utf-8") as f:
            document_text = pulisci_testo(f.read())

        # Creo documento unico
        doc = Document(text=document_text, metadata={"source": TXT_OUTPUT})

        # Splitting in finestre di 3 frasi sovrapposte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            keep_separator=True,
            # separator=" "  # chunk basato su spazi, mai tagliare parola a metà
        )

        chunks = text_splitter.split_text(document_text)

# Creo Document
        documents = [Document(text=c, metadata={"source": TXT_OUTPUT}) for c in chunks]
        print(f"🔍 Documenti generati con chunking word-safe: {len(documents)}")

        # print(f"🔍 Documenti generati: {len(documents)}")

        # Creo indice vettoriale
        print("📦 Creo indice vettoriale...")
        index = VectorStoreIndex(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    else:
        print("📂 Carico indice esistente...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index

# ===============================
# Ricarichiamo l'indice
# ===============================
index = inizializza_indice()

# Dopo aver creato l'indice
print("🔎 PRIMI CHUNK DELL'INDICE")
for i, doc in enumerate(index.storage_context.docstore.docs.values()):
    content = doc.get_content()
    if "bancomat" in content.lower():
        print(f"✅ Chunk {i} contiene 'bancomat':\n{content}\n")
        break
else:
    print("❌ Nessun chunk contiene 'bancomat' nel testo caricato.")
# Query di test con top_k più alto
# query_engine = index.as_query_engine(similarity_top_k=50)
# response = query_engine.query("dove si trova il bancomat?")
# print(response)

# Il Chat Engine ora deve "sostituire" la frase trovata con l'intero paragrafo circostante
chat_engine = index.as_chat_engine(
    chat_mode="context",
    similarity_top_k=25,
    system_prompt=(
        "Sei ADAi, un assistente virtuale che risponde basandosi esclusivamente sui documenti forniti. "
        "REGOLE DI RISPOSTA:\n"
        "1. Identifica il servizio richiesto e l'edificio associato nel testo.\n"
        "2. Se un servizio è chiaramente indicato in un edificio, non confonderlo con altri edifici citati nei frammenti.\n"
        "3. Se il documento non specifica la posizione esatta (via/mappa) di un edificio, ammetti di non averla e suggerisci di consultare la mappa ufficiale.\n"
        "4. Non inventare mai associazioni tra servizi e numeri di edificio se non sono esplicitamente scritte nel testo recuperato."
    )
)

# ======================================================
# 4. LOGICHE AI
# ======================================================

def risposta_chat_completa(messaggio):
    print(f"💬 LOG: Domanda utente: {messaggio}")
    messaggio_safe = anonimizza_input(messaggio)
    response = chat_engine.chat(messaggio_safe)

    print("🔎 --- DEBUG FRAMMENTI RECUPERATI ---")
    if hasattr(response, "source_nodes"):
        for i, node in enumerate(response.source_nodes[:5]):
            print(f"[{i+1}] SCORE {node.score:.4f} - {node.node.metadata.get('file_name')}")
            print(node.node.get_content())
            print("------------------------------------")

    print(f"🤖 RISPOSTA: {str(response)}\n")
    return str(response)

def navigatore_ai(richiesta, posizione):
    prompt = f"Sei ADAi. Utente a: {posizione}. Destinazione: {richiesta}. Fornisci percorso in 8 passi."
    risposta = Settings.llm.complete(prompt).text.strip()
    return "maps/Mappa-AOU-legenda-lato.jpg", risposta

# ======================================================
# 5. INTERFACCIA GRADIO
# ======================================================

def login_handler(username, password):
    utenti = carica_utenti()
    if username in utenti and utenti[username]["password"] == password:
        is_staff = (utenti[username]["role"] == "staff")
        return gr.update(visible=False), gr.update(visible=True), f"👤 **{utenti[username]['nome']}**", ""
    return gr.update(), gr.update(), "", "❌ Credenziali errate."

def vai_ospite():
    return gr.update(visible=False), gr.update(visible=True), "👤 **OSPITE**", ""

with gr.Blocks(title="ADAI - Policlinico") as demo:

    with gr.Column(visible=True) as login_view:
        gr.Markdown("# 🏥 Accesso Portale ADAi")
        user_id = gr.Textbox(label="Identificativo")
        user_pw = gr.Textbox(label="Password", type="password")
        btn_login = gr.Button("Entra", variant="primary")
        btn_ospite = gr.Button("Continua come Ospite")
        login_error = gr.Markdown("")

    with gr.Column(visible=False) as main_view:
        with gr.Row():
            status_text = gr.Markdown()
            btn_logout = gr.Button("Logout", size="sm")

        with gr.Tabs():
            with gr.TabItem("💬 Assistente"):
                chatbot = gr.Chatbot(label="Chat ADAi")
                msg_input = gr.Textbox(placeholder="Chiedi orari, edifici, servizi...")

                def gestisci_chat(msg, history):
                    if not msg: return "", history
                    risposta = risposta_chat_completa(msg)
                    history.append({"role": "user", "content": msg})
                    history.append({"role": "assistant", "content": risposta})
                    return "", history

                msg_input.submit(gestisci_chat, [msg_input, chatbot], [msg_input, chatbot])

            with gr.TabItem("📍 Navigazione"):
                pos = gr.Dropdown(["Ingresso", "CUP", "Pronto Soccorso"], label="Posizione attuale")
                dest = gr.Textbox(label="Dove vuoi andare?")
                btn_nav = gr.Button("Vai")
                mappa_img = gr.Image()
                guida_testo = gr.Markdown()
                btn_nav.click(navigatore_ai, [dest, pos], [mappa_img, guida_testo])

    btn_login.click(login_handler, [user_id, user_pw], [login_view, main_view, status_text, login_error])
    btn_ospite.click(vai_ospite, None, [login_view, main_view, status_text, login_error])
    btn_logout.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [login_view, main_view])

demo.launch(theme=gr.themes.Soft())