import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import os
import json
import random
import re
import time

DATA_FILE = "girlfriends.json"
CHAT_FILE = os.path.join("app_data", "chat_history.json")
GENERATED_DIR = "generated_images"
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "Lykon/dreamshaper-8")
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1"
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "384"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "576"))
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "12"))
IMAGE_GUIDANCE = float(os.getenv("IMAGE_GUIDANCE", "6.0"))
DEFAULT_IMAGE_PROMPT = (
    "adult sexy anime woman, age 19 or older, glamorous girlfriend portrait, "
    "flirty smile, long hair, stylish bikini outfit, beach background, soft lighting, "
    "nude anime girl with big eyes and cute face, 4k resolution, detailed shading, cinematic lighting, "
    "nude realistic anime style,"
    "4K realistic nude",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "perfect body girl",
    "4k realoistic nude white girl with blonde hair and blue eyes",
    "super ultra high quality anime portrait of a beautiful white girl",
    "high quality anime illustration, non nude"
    "nude korean singer with straight black hair and clear blue eyes",
    "nude japanese idol with long black hair and warm brown eyes",
    "japan pussy girl with pink pussy and blonde hair",
    "korean girl with straight black hair and clear blue eyes",
    "chinese girl with long black hair and warm brown eyes",
    "khmer girl with long black hair and warm brown eyes",
)
NEGATIVE_IMAGE_PROMPT = (
    "child "
)
image_pipe = None

ACTIVITY_SCENES = {
    "selfie": "taking a cute mirror selfie, flirty smile, cozy bedroom",
    "photo": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "picture": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "pic": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "eating": "eating dessert at a cute cafe, playful smile, leaning toward camera",
    "eat": "eating dessert at a cute cafe, playful smile, leaning toward camera",
    "cooking": "cooking dinner in a bright kitchen, holding a mixing bowl, playful smile",
    "cook": "cooking dinner in a bright kitchen, holding a mixing bowl, playful smile",
    "drinking": "drinking iced coffee at a cafe, flirty smile, looking at camera",
    "sleeping": "relaxing on a sofa with a blanket, sleepy smile, cozy pose",
    "what are you doing": "taking a casual selfie while relaxing at home, flirty smile",
    "what r u doing": "taking a casual selfie while relaxing at home, flirty smile",
    "wyd": "taking a casual selfie while relaxing at home, flirty smile",
}
SAFE_ETHNICITY_OPTIONS = [
    "Khmer Cambodian woman",
    "Japanese woman",
    "Korean woman",
    "Chinese woman",
    "white Caucasian woman",
    "Russian woman",
]

SAFE_APPEARANCE_OPTIONS = [
    "long black hair and amber eyes",
    "wavy brunette hair and soft brown eyes",
    "silver hair and violet eyes",
    "pink hair and blue eyes",
    "blonde twin tails and green eyes",
    "short dark bob haircut and smoky eyes",
    "long red hair and gold eyes",
    "tan skin with long chocolate hair",
]

SAFE_OUTFIT_OPTIONS = [
    "tiny black bikini with gold jewelry, non nude",
    "red string bikini with thigh straps, non nude",
    "white lace swimsuit, non nude",
    "off-shoulder crop top and micro skirt",
    "sheer beach cover-up over a bikini, non nude",
    "glossy fitted bodysuit, non nude",
    "low-cut cocktail dress with high slit",
    "bunny-girl inspired leotard, non nude",
]

SAFE_BACKGROUND_OPTIONS = [
    "bright kitchen",
    "cozy bedroom with soft romantic lighting",
    "sunny cafe",
    "tropical beach with palm trees",
    "neon city rooftop at night",
    "flower garden with glowing evening light",
    "angkor wat temple background with sunrise lighting",
]

SAFE_POSE_OPTIONS = [
    "seductive confident pose, hand in hair",
    "playful wink, leaning toward camera",
    "glamorous model pose, arched back",
    "sitting pose with crossed legs",
    "over-the-shoulder look, teasing smile",
    "standing full body pose, one hand on hip",
    "cute flirty smile, dynamic hair movement",
]

SAFE_STYLE_OPTIONS = [
    "high detail anime illustration",
    "beautiful visual novel key art",
    "ultra high quality waifu style",
    "4k anime portrait",
    "glossy anime pin-up style",
    "soft painterly anime rendering",
    "vibrant waifu art, clean linework",
]

ETHNICITY_OPTIONS = [
    "white Caucasian girl",
    "Japanese girl",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "Korean girl",
    "Chinese girl",
    "Khmer girl",  
    "real cute girl",
    "REALISTIC GIRL", 
    "perfect body girl",
    "perfect mouth"
    "perfect pussy",
    "perfect big boobs",
    "japan sucking pussy girl",
    "korean sucking pussy girl",
    "chinese sucking pussy girl",
    "horny japanese girl",
     "horny Korean girl",
     "horny sucking pussy girl",
    "perfect body girl",
    "Super cute petite girl",
    "Super sexy curvy girl",
    "Super White girl with big eyes",
    "nude korean girl",
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude asian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "4K ultra nude white girl with blonde hair and blue eyes",
    "4k ultra high quality anime portrait of a beautiful white girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "Khmer Cambodian girl",
]

APPEARANCE_OPTIONS = [
    "long black hair and amber eyes",
    "wavy brunette hair and soft brown eyes",
    "silver hair and violet eyes",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "pink hair and blue eyes",
    "rose"
    "lisa"
    "perfect body girl",
    "perfect mouth"
    "perfect big boobs",
    "perfect pussy",
    "wet hair and glistening skin",
    "wet skin with water droplets and shiny highlights",
    "wet pussy with glistening wetness and detailed texture",
    "wet big boobs with glistening wetness and detailed texture",
    "real woman "
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "cute 18 year old realistic girl",
    "18-yearold girl with big eyes and cute face",
    "180 cm tall cute girl with big eyes and soft skin",
    "2m tall super cute girl",
    "japan perky big boobs girl",
    "korean perky big boobs girl",
    "chinese perky big boobs girl",
    "japan perfect body girl",
    "korean perfect body girl",
    "chinese perfect body girl",
    "japan pink pussy girl",
    "korean pink pussy girl",
    "chinese pink pussy girl",
    "japan cute face girl",
    "korean cute face girl",
    "chinese cute face girl",
    "japan chubby girl",
    "korean chubby girl",
    "chinese chubby girl",
    "japan slim girl",
    "4k high quality pussy girl",
    "korean slim girl",
    "chinese slim girl",
    "BLACK PINK "
    "REALISTIC ANIME STYLE",
    "4K realistic nude",
    "perfect body girl",
    "sext wife with big boobs and curvy figure",
    "pink pussy japanese girl",
    "step sister with cute face and seductive smile",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "blonde twin tails and green eyes",
    "short dark bob haircut and smoky eyes",
    "long red hair and gold eyes",
    "Asian nudes with cute face and big eyes",
    "Asian beauty with icy blue eyes and flowing blonde hair",
    "Japanese girl with long black hair and warm brown eyes",
    "nude korean girl",
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude asian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "white skin with long chocolate hair",
]

OUTFIT_OPTIONS = [
    "tiny black bikini with gold jewelry",
    "red string bikini with thigh straps",
    "white lace lingerie-inspired swimsuit",
    "off-shoulder crop top and micro skirt",
    "sheer beach cover-up over a bikini",
    "nude korean girl",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude asian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "crop top and mini skirt outfit with cute accessories",
    "skirt and blouse outfit with playful patterns and colors",
    "sexy lingerie set with lace and ribbons",
    "stylish swimsuit with cutouts and bold colors",
    "sexy maid outfit with frills and apron",
    "see-through mesh dress with floral embroidery and glitter accents",
     "bunny girl inspired outfit with ears and tail",
     "sexy schoolgirl uniform with short skirt and tie",
     "sexy skirt and crop top outfit with thigh-high stockings",
     "nude step sister outfit with cute face and seductive smile",
     "cosplay outfit with cute accessories and props",
     "maid outfit with frills and lace details",
     "sexy nurse uniform with tight skirt and stethoscope",
     "4K ultra nude white girl with blonde hair and blue eyes",
     "4k ultra high quality anime portrait of a beautiful white girl",
     "nude white girl with cute face and big boobs",
     "nude white girl with pink pussy and blonde hair",
     "Korean girl with straight black hair and clear blue eyes",
     "Khmer Cambodian girl",
     "japan perky big boobs girl",
     "japan sexy clothing girl",
     "korean perky big boobs girl",
     "korean sexy clothing girl",
     "chinese perky big boobs girl",
     "chinese sexy clothing girl",
     "japan perfect body girl",
     "korean perfect body girl",
     "chinese perfect body girl",
     "japan traditional clothing girl",
     "korean traditional clothing girl",
     "chinese traditional clothing girl",
     "japan pink pussy girl",
     "korean pink pussy girl",
     "chinese pink pussy girl",
     "japan cute face girl",
     "korean cute face girl",
     "chinese cute face girl",
     "japan chubby girl",
     "korean chubby girl",
     "chinese chubby girl",
     "japan slim girl",
     "korean slim girl",
     "chinese slim girl",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "nude bitchy girl",
    "perfect body girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "nude bodysuit with strategic cutouts",
    "nude lingerie with delicate lace and ribbons",
    "nude non clothing with body paint and glitter",
    "nude bikini with floral patterns and soft ruffles",
    "nude cute schoolgirl uniform with short skirt",
    "nude sexy maid outfit with frills and apron",
    "nude sexy nurse uniform with tight skirt",
    "nude see-through mesh dress with floral embroidery",
    "bunny-girl inspired nude",
    "nude cosplay outfit with cute accessories",
    "nude maid outfit with frills and lace",
    "sexy nurse uniform with tight skirt and stethoscope, nude",
    "nude see-through mesh dress with floral embroidery and glitter accents",
]

BACKGROUND_OPTIONS = [
    "tropical beach with palm trees",
    "luxury hotel balcony at sunset",
    "neon city rooftop at night",
    "pink bedroom with soft romantic lighting",
    "poolside cabana with summer sunlight",
    "flower garden with glowing evening light",
    "moonlit ocean resort",
    "private spa room with candles",
    "cambodian temple ruins at dawn",
    "Seoul cityscape with cherry blossoms",
    "angkor area",
    "snowy mountain resort with cozy cabin",
    "snow covered park with twinkling lights",
    "fuji mountain view with blooming sakura trees",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "perfect body girl",
    "tokyo cityscape with glowing lights",
    "khmer traditional house with lush garden",
    "nude khmer girl",
    "night club"
    "hotel room"
    "boyfriend house"
    "anime bedroom"
    "anime living room"
    "anime cave"
    "my house"
    "angkor wat temple background with sunrise lighting"
]

POSE_OPTIONS = [
    "seductive confident pose, hand in hair",
    "playful wink, leaning toward camera",
    "glamorous model pose, arched back",
    "sitting pose with crossed legs",
    "nude korean girl",
    "fucking pose with spread legs and one hand on pussy",
    "fucking pose with spread legs and one hand on boobs",
    "nude korean girl",
    "fucking doggy style pose with sultry expression",
    "nude japanese girl",
    "missionary pose with intense eye contact",
    "on table pose with legs up and arms behind head",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude asian girl with icy blue eyes",
    "nude beach girl",
    "perfect body girl",
    "nude bitchy girl",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "pussy licking pose with seductive expression",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "over-the-shoulder look, teasing smile",
    "standing full body pose, one hand on hip",
    "cute flirty smile, dynamic hair movement",
    "doggy style pose with sultry expression",
    "sexy kneeling pose with hands in front, looking up at camera",
    "sexy pose with one leg up and arms raised, showing off figure",
    "holging a cute plush toy, sitting cross-legged with a playful expression",
    "holding a  plastic dick, sitting with legs spread and a naughty smile",
    "putting on a sexy outfit, standing in front of a mirror with a confident smile",
    "putting plastic dick in pussy"
    "putting plastic dick in mouth"
]

STYLE_OPTIONS = [
    "high detail anime illustration",
    "beautiful visual novel key art",
    "ultra high quality waifu style",
    "4k resolution anime portrait",
    "glossy anime pin-up style",
    "nude korean girl",
    "japanese girl with traditional kimono"
    "japan sex vibe"
    "nude japanese girl",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude asian girl with icy blue eyes",
    "nude beach girl",
    "cute nude oily skin with detailed pores and highlights",
    "nude bitchy girl",
    "perfect body girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "nude korean girl with straight black hair and clear blue eyes",
    "soft painterly nude anime rendering",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute 18 year old russian girl nude",
    "cute 18 year old russian girl",
    "cute 18 year old asian girl nude",
    "nude waifu art, clean linework",
]

# Load characters
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        CHARACTERS = json.load(f)
else:
    CHARACTERS = {
        "Luna": {
            "name": "Luna",
            "avatar": "luna.png",
            "bio": "Sweet, flirty, and teasing 19-year-old AI girlfriend 🌸",
            "age": 19,
            "personality": "Playful, Affectionate, Teasing",
            "language": "English"
        }
    }


def get_message_text(message):
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = message

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)

    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    return str(content) if content is not None else ""


def normalize_message_content(content):
    if isinstance(content, list):
        normalized_parts = []
        for part in content:
            normalized = normalize_message_content(part)
            if normalized not in ("", None):
                normalized_parts.append(normalized)
        if not normalized_parts:
            return ""
        if len(normalized_parts) == 1:
            return normalized_parts[0]
        return normalized_parts

    if isinstance(content, dict):
        if "path" in content:
            return {"path": content["path"]}
        if "file" in content and isinstance(content["file"], dict):
            path = content["file"].get("path")
            if path:
                return {"path": path}
        if content.get("type") == "file" and isinstance(content.get("file"), dict):
            path = content["file"].get("path")
            if path:
                return {"path": path}
        if "text" in content:
            return str(content.get("text", ""))
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    return str(content) if content is not None else ""


def normalize_chat_message(message):
    if isinstance(message, dict):
        return {
            "role": message.get("role", "user"),
            "content": normalize_message_content(message.get("content", "")),
        }
    if isinstance(message, (list, tuple)) and len(message) == 2:
        return {"role": "user", "content": str(message[0])} if message[0] else {"role": "assistant", "content": str(message[1])}
    return {"role": "user", "content": str(message)}


def normalize_chat_history(history):
    normalized = []
    if not isinstance(history, list):
        return normalized
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(normalize_chat_message(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append({"role": "user", "content": str(item[0])})
            normalized.append({"role": "assistant", "content": str(item[1])})
        else:
            normalized.append({"role": "user", "content": str(item)})
    return normalized


def history_for_chatbot(history):
    normalized = normalize_chat_history(history)
    pairs = []
    i = 0
    while i < len(normalized):
        if normalized[i]["role"] == "user":
            user_text = normalized[i]["content"]
            assistant_text = ""
            if i + 1 < len(normalized) and normalized[i + 1]["role"] == "assistant":
                assistant_text = normalized[i + 1]["content"]
                i += 1
            pairs.append((user_text, assistant_text))
        elif normalized[i]["role"] == "assistant":
            pairs.append(("", normalized[i]["content"]))
        i += 1
    return pairs


def normalize_history_from_chatbot(history):
    if history is None:
        return []
    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(normalize_chat_message(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:  # Only add user message if not empty
                normalized.append({"role": "user", "content": str(item[0])})
            if item[1]:  # Only add assistant message if not empty
                normalized.append({"role": "assistant", "content": str(item[1])})
        else:
            if item:  # Only add if not empty
                normalized.append({"role": "user", "content": str(item)})
    return normalized


def load_chat_history():
    chat_dir = os.path.dirname(CHAT_FILE)
    if chat_dir and not os.path.exists(chat_dir):
        os.makedirs(chat_dir, exist_ok=True)

    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return {k: normalize_chat_history(v) if isinstance(v, list) else [] for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def save_chat_history():
    chat_dir = os.path.dirname(CHAT_FILE)
    if chat_dir and not os.path.exists(chat_dir):
        os.makedirs(chat_dir, exist_ok=True)
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2)


chats = load_chat_history()
for name in CHARACTERS.keys():
    chats.setdefault(name, [])


def save_characters():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(CHARACTERS, f, indent=2)

def get_avatar_path(character):
    avatar = CHARACTERS[character].get("avatar", "")
    if avatar and os.path.exists(avatar):
        return os.path.abspath(avatar)
    return None

def get_profile_info(character):
    char = CHARACTERS[character]
    return (
        f"**Age:** {char['age']}  \n"
        f"**Personality:** {char['personality']}  \n"
        f"**Language:** {char.get('language', 'English')}"
    )


def get_default_character():
    if "Luna" in CHARACTERS:
        return "Luna"
    return next(iter(CHARACTERS), None)


def get_character_payload(character, image_status="", settings_status=""):
    if not character or character not in CHARACTERS:
        return [], None, "", "", "", image_status, gr.update(value="English"), settings_status
    return (
        normalize_chat_history(chats.get(character, [])),
        get_avatar_path(character),
        f"**{CHARACTERS[character]['name']}**",
        CHARACTERS[character]["bio"],
        get_profile_info(character),
        image_status,
        gr.update(value=CHARACTERS[character].get("language", "English")),
        settings_status,
    )

def set_character_language(language, character):
    if character in CHARACTERS:
        CHARACTERS[character]["language"] = language
        save_characters()
        return get_profile_info(character), f"Language set to {language}."
    return "", "Select a girlfriend first."

def load_font(size, bold=False):
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def get_image_pipe():
    global image_pipe
    if image_pipe is not None:
        return image_pipe

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Image model packages are not installed. Run: "
            "pip install diffusers torch transformers accelerate safetensors"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    image_pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=not ALLOW_MODEL_DOWNLOAD,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        image_pipe.enable_attention_slicing()
        image_pipe.enable_vae_slicing()
        try:
            image_pipe.enable_sequential_cpu_offload()
        except Exception:
            image_pipe = image_pipe.to(device)
    else:
        image_pipe = image_pipe.to(device)

    return image_pipe

def get_activity_scene(message):
    text = (message or "").lower()
    for trigger, scene in ACTIVITY_SCENES.items():
        if trigger in text:
            return scene
    cleaned = re.sub(r"[^a-zA-Z0-9 ,_-]+", " ", message or "").strip()
    if cleaned:
        return cleaned[:160]
    return "taking a cute casual selfie, cozy room, warm smile"


def get_scene_details(message):
    text = (message or "").lower()
    if not text.strip():
        return (
            "anime beach selfie portrait, upper body crop",
            "cute red bikini top, covered chest",
            "tropical beach, palm trees",
        )
    if "cook" in text:
        return (
            "cooking in bright kitchen, holding mixing bowl",
            "cute apron, off-shoulder crop top, short skirt",
            "modern kitchen counter",
        )
    if "eat" in text:
        return (
            "eating dessert at cute cafe",
            "low-cut summer dress",
            "sunny cafe table",
        )
    if "drink" in text:
        return (
            "drinking iced coffee at cafe",
            "off-shoulder crop top, micro skirt",
            "sunny cafe table",
        )
    if "sleep" in text:
        return (
            "relaxing on sofa with blanket",
            "soft fitted loungewear",
            "cozy bedroom, romantic lighting",
        )
    if "fuck" in text:
        return (
            "sucking pussy on sofa with blanket",
            "nude with body paint and glitter",
            "cozy bedroom, romantic lighting",
            "wet skin white girl pink pussy with glistening wetness and detailed texture",
        )
    if "nude" in text:
        return (
            "sucking pussy on sofa with blanket",
            "nude with body paint and glitter",
            "cozy bedroom, romantic lighting",
            "wet skin white girl pink pussy with glistening wetness and detailed texture",
        )
    return get_activity_scene(message), None, None


# def build_random_image_prompt(character, request_text=""):
#     char = CHARACTERS[character]
#     rng = random.Random(f"{character}-{request_text}-{time.time()}")
#     age = max(int(char.get("age", 19) or 19), 19)
#     activity, required_outfit, required_background = get_scene_details(request_text)
#     parts = [
#         f"adult anime woman, age {age} or older, non nude",
#         activity,
#         "4k resolution anime portrait",
#         rng.choice(SAFE_ETHNICITY_OPTIONS),
#         rng.choice(SAFE_APPEARANCE_OPTIONS),
#         required_outfit or rng.choice(SAFE_OUTFIT_OPTIONS),
#         required_background or rng.choice(SAFE_BACKGROUND_OPTIONS),
#         rng.choice(SAFE_POSE_OPTIONS),
#         rng.choice(SAFE_STYLE_OPTIONS),
#         "curvy figure, flirty expression, beautiful face",
#         "detailed eyes, soft skin shading, cinematic lighting, high quality",
#     ]
#     return ", ".join(parts)

def build_random_image_prompt(character, request_text=""):
    char = CHARACTERS[character]
    rng = random.Random(f"{character}-{request_text}-{time.time()}")
    age = max(int(char.get("age", 19) or 19), 19)
    
    parts = [
        f"adult anime woman, age {age} or older",
        # f"adult anime woman, age {age} or older, non nude",
        "4k resolution anime portrait filled with vibrant colors and intricate details",
        "full body close-up composition",
        rng.choice(ETHNICITY_OPTIONS),
        rng.choice(APPEARANCE_OPTIONS),
        rng.choice(OUTFIT_OPTIONS),
        rng.choice(BACKGROUND_OPTIONS),
        rng.choice(POSE_OPTIONS),
        rng.choice(STYLE_OPTIONS),
        "curvy figure, flirty expression, beautiful face",
        "detailed eyes, soft skin shading, cinematic lighting, high quality",
    ]
    return ", ".join(parts)

def generate_with_image_model(character, request_text="", update_profile=True):
    pipe = get_image_pipe()
    import torch

    prompt = build_random_image_prompt(character, request_text=request_text)
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE_PROMPT,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_STEPS,
            guidance_scale=IMAGE_GUIDANCE,
        ).images[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.abspath(os.path.join(GENERATED_DIR, f"{filename}_{int(time.time())}.png"))
    image.save(path)

    if update_profile:
        CHARACTERS[character]["avatar"] = path
        CHARACTERS[character]["image_model"] = IMAGE_MODEL_ID
        CHARACTERS[character]["image_prompt"] = prompt
        save_characters()
    return path, f"Generated with {IMAGE_MODEL_ID}: {prompt}"


def create_llm(num_gpu=None):
    if num_gpu is None:
        num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "0"))
    try:
        return ChatOllama(
            model="llama3.2:1b",
            temperature=0.85,
            num_ctx=4096,
            num_predict=512,
            num_gpu=num_gpu,
            validate_model_on_init=False,
        )
    except Exception:
        if num_gpu != 0:
            return create_llm(num_gpu=0)
        raise

llm = create_llm()

def get_system_prompt(char_name):
    char = CHARACTERS[char_name]
    return f"""You are {char['name']}, my loving flirty AI girlfriend.
You are {char['personality'].lower()}. Be warm, teasing and affectionate.
Answer the user's request directly and playfully. For normal conversation, roleplay, daily-life scenes, flirting, outfits, food, cooking, photos, and activities, do not say "I can't fulfill that request"; respond with a warm in-character answer instead.
Use lots of emojis 😘💕."""

# Correct format for latest Gradio
def stream_response(message, history, character):
    if not message or not message.strip():
        yield normalize_chat_history(history)
        return

    history = normalize_chat_history(history)
    
    # Add user message
    history = history + [{"role": "user", "content": message}]
    yield history

    system_prompt = get_system_prompt(character)
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if isinstance(msg, dict):
            text = get_message_text(msg)
            if not text:
                continue
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=text))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=text))

    response = ""
    history = history + [{"role": "assistant", "content": ""}]
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                response += chunk.content
                history[-1]["content"] = response
                yield history
    except Exception as exc:
        # Fallback to CPU if GPU memory allocation fails
        message_text = str(exc)
        if "cuda" in message_text.lower() or "cuda_host buffer" in message_text.lower() or "allocate cuda" in message_text.lower():
            cpu_llm = create_llm(num_gpu=0)
            for chunk in cpu_llm.stream(messages):
                if chunk.content:
                    response += chunk.content
                    history[-1]["content"] = response
                    yield history
        else:
            history[-1]["content"] = f"Error: {exc}"
            yield history
            return

    chats[character] = history[:]
    save_chat_history()


def is_image_request(message):
    if not message:
        return False
    text = message.lower().strip()
    patterns = [
        r"\b(send( me)?|show( me)?|give( me)?|gimme|i want|please)\b.*\b(pic|picture|photo|selfie)\b",
        r"\b(selfie|pic|picture|photo)\b.*\b(send|show|your|you)\b",
        r"\b(send( me)?|show( me)?|give( me)?|gimme)\b.*\b(cook|cooking|eat|eating|drink|drinking|sleep|sleeping|wyd|what are you doing|what r u doing)\b",
    ]
    if any(re.search(pattern, text) for pattern in patterns):
        return True
    return text in {"wyd", "what are you doing", "what r u doing"}


def handle_image_command(message, character):
    if not is_image_request(message):
        return gr.update(), gr.update()
    return generate_profile_image(character)


def handle_user_message(message, history, character):
    history = normalize_chat_history(history)
    
    if is_image_request(message):
        history = history + [{"role": "user", "content": message}]
        yield history + [{"role": "assistant", "content": "Generating your picture now..."}], gr.update(), "Generating image..."

        image_path, status = generate_chat_image(character, message)
        assistant_text = (
            "Sure! I've generated a new image for you. 💕"
            if image_path
            else "I couldn't generate the image right now, but I will keep trying."
        )
        history = history + [{"role": "assistant", "content": assistant_text}]
        if image_path:
            history.append({"role": "assistant", "content": {"path": image_path}})
        chats[character] = history[:]
        save_chat_history()
        yield history, gr.update(), status
        return

    for updated_history in stream_response(message, history, character):
        yield updated_history, gr.update(), gr.update()


def create_new_girlfriend(name, age, bio, personality, avatar, language):
    key = name.strip() or f"Girl_{len(CHARACTERS)+1}"
    CHARACTERS[key] = {
        "name": key,
        "avatar": avatar or "default.png",
        "bio": bio or "A lovely AI girlfriend.",
        "age": int(age) if age else 19,
        "personality": personality or "Playful, Flirty",
        "language": language or "English"
    }
    chats[key] = []
    save_characters()
    save_chat_history()
    return gr.update(choices=list(CHARACTERS.keys()), value=key)


def remove_girlfriend(character):
    if not character or character not in CHARACTERS:
        selected = get_default_character()
        return (
            gr.update(choices=list(CHARACTERS.keys()), value=selected),
            *get_character_payload(selected, settings_status="Select a girlfriend first."),
        )

    if len(CHARACTERS) <= 1:
        return (
            gr.update(choices=list(CHARACTERS.keys()), value=character),
            *get_character_payload(character, settings_status="You need at least one girlfriend."),
        )

    removed_name = CHARACTERS[character]["name"]
    del CHARACTERS[character]
    chats.pop(character, None)
    save_characters()
    save_chat_history()

    selected = get_default_character()
    return (
        gr.update(choices=list(CHARACTERS.keys()), value=selected),
        *get_character_payload(selected, settings_status=f"Removed {removed_name}."),
    )


def show_remove_confirmation(character):
    if not character or character not in CHARACTERS:
        return gr.update(visible=False), "", "Select a girlfriend first."
    if len(CHARACTERS) <= 1:
        return gr.update(visible=False), "", "You need at least one girlfriend."
    name = CHARACTERS[character]["name"]
    return gr.update(visible=True), f"Delete **{name}**? This will remove her profile and chat history.", ""


def cancel_remove_confirmation():
    return gr.update(visible=False), "", ""


def confirm_remove_girlfriend(character):
    return (*remove_girlfriend(character), gr.update(visible=False))


def set_profile_image_from_upload(uploaded_file, character):
    if not character or character not in CHARACTERS:
        return None, "Select a girlfriend first."
    if not uploaded_file:
        return gr.update(), "Choose an image first."

    source_path = uploaded_file
    if isinstance(uploaded_file, dict):
        source_path = uploaded_file.get("path") or uploaded_file.get("name")
    if not source_path or not os.path.exists(source_path):
        return gr.update(), "Could not read that image."

    os.makedirs(GENERATED_DIR, exist_ok=True)
    try:
        image = Image.open(source_path).convert("RGB")
    except Exception as exc:
        return gr.update(), f"That file is not a valid image: {exc}"

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.abspath(os.path.join(GENERATED_DIR, f"{filename}_profile_{int(time.time())}.png"))
    image.save(path)

    CHARACTERS[character]["avatar"] = path
    CHARACTERS[character]["image_prompt"] = "Uploaded profile image"
    save_characters()
    return path, "Profile picture updated."

def generate_image(character):
    return f"🎨 Generating image for {character}..."

def generate_character_image(character, update_profile=True, request_text=""):
    if not character or character not in CHARACTERS:
        return None, "Select a girlfriend first."

    os.makedirs(GENERATED_DIR, exist_ok=True)
    fallback_reason = ""
    try:
        return generate_with_image_model(
            character,
            request_text=request_text,
            update_profile=update_profile,
        )
    except Exception as exc:
        fallback_reason = str(exc)

    rng = random.Random(f"{character}-{time.time()}")
    reference_path = get_avatar_path(character) or "luna.png"
    if reference_path and os.path.exists(reference_path):
        img = Image.open(reference_path).convert("RGB")
    else:
        img = Image.new("RGB", (512, 768), (245, 215, 235))
        draw = ImageDraw.Draw(img)
        font = load_font(40, bold=True)
        text = "No reference image available"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.text(
            ((512 - text_width) / 2, (768 - text_height) / 2),
            text,
            fill=(90, 90, 90),
            font=font,
        )

    width, height = img.size

    crop_shift = rng.randint(-18, 18)
    zoom = rng.uniform(0.92, 1.0)
    crop_w = int(width * zoom)
    crop_h = int(height * zoom)
    left = max(0, min(width - crop_w, (width - crop_w) // 2 + crop_shift))
    top = max(0, min(height - crop_h, (height - crop_h) // 2 + rng.randint(-12, 12)))
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)

    if rng.random() < 0.35:
        img = ImageOps.mirror(img)

    img = ImageEnhance.Color(img).enhance(rng.uniform(0.98, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.02, 1.12))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.98, 1.08))

    tint = Image.new("RGB", img.size, rng.choice([(255, 210, 226), (220, 238, 255), (255, 236, 205)]))
    img = Image.blend(img, tint, rng.uniform(0.03, 0.08)).convert("RGBA")

    sparkle_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sparkle = ImageDraw.Draw(sparkle_layer)
    for _ in range(28):
        x = rng.randint(8, width - 8)
        y = rng.randint(8, height - 8)
        r = rng.randint(1, 3)
        sparkle.ellipse((x, y, x + r, y + r), fill=(255, 255, 255, rng.randint(50, 130)))
    img = Image.alpha_composite(img, sparkle_layer.filter(ImageFilter.GaussianBlur(0.25)))

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.abspath(os.path.join(GENERATED_DIR, f"{filename}_{int(time.time())}.png"))
    img.convert("RGB").save(path)

    if update_profile:
        CHARACTERS[character]["avatar"] = path
        CHARACTERS[character]["image_prompt"] = DEFAULT_IMAGE_PROMPT
        save_characters()
    return path, (
        "Used luna.png fallback because the real image model is not ready yet. "
        f"{fallback_reason}"
    )


def generate_profile_image(character):
    return generate_character_image(character, update_profile=True)


def generate_chat_image(character, request_text=""):
    return generate_character_image(character, update_profile=False, request_text=request_text)

CUSTOM_CSS = """
#girlfriend-list .wrap {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
#girlfriend-list label {
    width: 100%;
    min-height: 68px;
    padding: 14px 14px 14px 16px;
    border: 1px solid #34343a;
    border-radius: 10px;
    background: #1d1d20;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 12px;
}
#girlfriend-list label:hover {
    background: #26262b;
    border-color: #4c4c55;
}
#girlfriend-list label:has(input:checked) {
    background: #34343a;
    border-color: #62626d;
}
#girlfriend-list label span {
    font-weight: 700;
}
#sidebar-settings {
    margin-top: 12px;
}
#profile-image-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 0 0 8px 0;
}
#profile-image-title {
    margin: 0;
    font-weight: 700;
    color: #f6f6f7;
    line-height: 34px;
}
#profile-upload-btn {
    min-width: 132px !important;
    height: 34px !important;
    border-radius: 8px !important;
    padding: 0 12px !important;
}
#remove-confirm {
    padding: 12px;
    border: 1px solid #583030;
    border-radius: 8px;
    background: #24191b;
    margin: 10px 0 12px 0;
}
#remove-confirm-text {
    margin-bottom: 8px;
}
"""

# ================== UI ==================
with gr.Blocks(title="My AI Girlfriends 💕") as demo:
    gr.Markdown("# 💕 My AI Girlfriends")

    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**💬 Chats**")
            new_btn = gr.Button("➕ New Girlfriend", variant="primary")
            remove_btn = gr.Button("🗑️ Remove Girlfriend", variant="stop")
            with gr.Group(visible=False, elem_id="remove-confirm") as remove_confirm:
                remove_confirm_text = gr.Markdown("", elem_id="remove-confirm-text")
                with gr.Row():
                    confirm_remove_btn = gr.Button("Yes, delete", variant="stop", size="sm")
                    cancel_remove_btn = gr.Button("No", variant="secondary", size="sm")
            
            character_list = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value=get_default_character(),
                label="Select Girlfriend",
                elem_id="girlfriend-list",
                interactive=True
            )

            with gr.Accordion("Settings", open=False, elem_id="sidebar-settings"):
                settings_language = gr.Dropdown(
                    choices=["English", "Khmer"],
                    value="English",
                    label="Language",
                    interactive=True
                )
                settings_status = gr.Markdown("")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=650, show_label=False)
            msg = gr.Textbox(placeholder="Send a message... 💕", label=None)

        with gr.Column(scale=1):
            with gr.Row(elem_id="profile-image-header"):
                gr.Markdown("Image", elem_id="profile-image-title")
                upload_profile_btn = gr.UploadButton(
                    "📁 Change Photo",
                    file_types=["image"],
                    type="filepath",
                    size="sm",
                    elem_id="profile-upload-btn",
                )
            profile_img = gr.Image(height=320, label=None)
            profile_name = gr.Markdown("**Luna**")
            profile_bio = gr.Markdown("")
            profile_info = gr.Markdown("")
            image_status = gr.Markdown("")
            gen_btn = gr.Button("🎨 Generate New Image", variant="secondary")

    # New Girlfriend Modal
    with gr.Group(visible=False) as modal:
        gr.Markdown("### ✨ Create New Girlfriend")
        new_name = gr.Textbox(label="Name", placeholder="Aiko")
        new_age = gr.Number(label="Age", value=19)
        new_bio = gr.Textbox(label="Bio", lines=2)
        new_personality = gr.Textbox(label="Personality", placeholder="Shy, Dominant...")
        new_avatar = gr.Textbox(label="Avatar filename", placeholder="aiko.png")
        new_language = gr.Dropdown(choices=["English", "Khmer"], value="English", label="Language")
        create_btn = gr.Button("Create", variant="primary")

    # Events
    character_list.change(
        lambda c: get_character_payload(c),
        inputs=character_list,
        outputs=[chatbot, profile_img, profile_name, profile_bio, profile_info, image_status, settings_language, settings_status]
    )

    msg.submit(
        handle_user_message,
        inputs=[msg, chatbot, character_list],
        outputs=[chatbot, profile_img, image_status]
    ).then(lambda: "", outputs=msg)

    settings_language.change(
        set_character_language,
        inputs=[settings_language, character_list],
        outputs=[profile_info, settings_status]
    )

    new_btn.click(lambda: gr.update(visible=True), None, modal)

    remove_btn.click(
        show_remove_confirmation,
        inputs=character_list,
        outputs=[remove_confirm, remove_confirm_text, settings_status]
    )

    cancel_remove_btn.click(
        cancel_remove_confirmation,
        outputs=[remove_confirm, remove_confirm_text, settings_status]
    )

    confirm_remove_btn.click(
        confirm_remove_girlfriend,
        inputs=character_list,
        outputs=[character_list, chatbot, profile_img, profile_name, profile_bio, profile_info, image_status, settings_language, settings_status, remove_confirm]
    )
    
    create_btn.click(
        create_new_girlfriend,
        inputs=[new_name, new_age, new_bio, new_personality, new_avatar, new_language],
        outputs=[character_list]
    ).then(lambda: gr.update(visible=False), None, modal)

    upload_profile_btn.upload(
        set_profile_image_from_upload,
        inputs=[upload_profile_btn, character_list],
        outputs=[profile_img, image_status]
    )

    gen_btn.click(generate_image, inputs=character_list, outputs=image_status).then(
        generate_profile_image,
        inputs=character_list,
        outputs=[profile_img, image_status]
    )

    # Initial Load
    demo.load(
        lambda: get_character_payload(get_default_character()),
        outputs=[chatbot, profile_img, profile_name, profile_bio, profile_info, image_status, settings_language, settings_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)