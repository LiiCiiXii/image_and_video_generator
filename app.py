import os
import json
import random
import re
import time
import traceback
from pathlib import Path

ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1"
if not ALLOW_MODEL_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = os.getenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

import gradio as gr
import imageio
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "videos"
DATA_FILE = BASE_DIR / "girlfriends.json"
GENERATED_DIR = BASE_DIR / "generated_images"
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "Lykon/dreamshaper-8")
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "512"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "768"))
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "22"))
IMAGE_GUIDANCE = float(os.getenv("IMAGE_GUIDANCE", "7.5"))
LTX_IMAGE_WIDTH = int(os.getenv("LTX_IMAGE_WIDTH", "384"))
LTX_IMAGE_HEIGHT = int(os.getenv("LTX_IMAGE_HEIGHT", "512"))
LTX_IMAGE_FRAMES = int(os.getenv("LTX_IMAGE_FRAMES", "9"))
LTX_IMAGE_STEPS = int(os.getenv("LTX_IMAGE_STEPS", "8"))
LTX_IMAGE_GUIDANCE = float(os.getenv("LTX_IMAGE_GUIDANCE", "3.0"))
VIDEO_FPS = 24
FRAME_MULTIPLE = 8
MAX_FRAMES_PER_CLIP = 161
FASTEST_FRAMES_PER_CLIP = 49
BALANCED_FRAMES_PER_CLIP = 81
MAX_DURATION_SECONDS = 180
MAX_SINGLE_PASS_SECONDS = 3
PERFORMANCE_PROFILE = os.getenv("PERFORMANCE_PROFILE", "m2max").lower()
LOW_MEMORY_MODE = os.getenv("LOW_MEMORY_MODE", "0") == "1"
IMAGE_MPS_DTYPE = os.getenv("IMAGE_MPS_DTYPE", "float16").lower()

torch.set_grad_enabled(False)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
cpu_count = os.cpu_count()
if cpu_count is not None:
    torch.set_num_threads(cpu_count)

pipe = None
image_pipe = None
image_pipe_dtype = None

APP_CSS = """
.field-random-button {
    align-self: flex-end !important;
    flex: 0 0 88px !important;
    min-width: 76px !important;
    max-width: 88px !important;
}
.field-random-button button {
    min-height: 40px !important;
    height: 40px !important;
    padding: 0 10px !important;
    font-size: 14px !important;
    line-height: 1 !important;
}
"""


def get_server_port():
    configured_port = os.environ.get("GRADIO_SERVER_PORT")
    if configured_port:
        return int(configured_port)
    return None


DEFAULT_IMAGE_PROMPT = (
    "masterpiece anime character art, full body head-to-toe, feet visible, clean line art"
)
REALISTIC_IMAGE_PROMPT = (
    "RAW DSLR photo, 85mm lens, f/1.8, of a real adult age 19 or older, "
    "natural skin texture with visible pores and slight imperfections, "
    "soft golden hour side-lighting, depth of field, sharp focus on eyes, "
    "highly detailed, film grain, unedited, cinematic color grading, "
    "high dynamic range, "
)
NEGATIVE_IMAGE_PROMPT = (
    "bad anatomy, deformed face"
    "extra limbs, missing limbs, fused fingers, blurry, low quality, noisy, watermark, text, "
    "cropped head, headshot, bust shot, cut off legs, cut off feet, "
    " armor, metal armor, gold armor, cape, helmet"
)
VIDEO_NEGATIVE_PROMPT = (
    "blurry, flicker, jitter, shaky camera, warped face, bad anatomy, extra limbs, "
    "duplicate person, inconsistent character, text, watermark, close-up, cropped, "
    "out of frame, cutaway shot, anime, cartoon, cgi, plastic skin, doll face"
)

ETHNICITY_OPTIONS = [
    "Khmer Cambodian woman",
    "Japanese woman",
    "Korean woman",
    "Chinese woman",
    "white Caucasian woman",
    "Russian woman",
    "Scandinavian woman",
    "German woman",
    "French woman",
    "British woman",
    "Irish woman",
    "Ukrainian woman",
    "Polish woman",
    "Dutch woman",
    "Italian woman",
    "Mongolian woman",
    "Kazakhstani woman",
    "Uzbekistani woman",
    "Tajikistani woman",
]

APPEARANCE_OPTIONS = [
    "tall and graceful build, long slender legs, radiant white skin, flowing raven-black hair, and deep expressive eyes",
    "stately height, model-like long legs, dewy porcelain-white skin, wavy chestnut hair, and warm brown eyes",
    "statuesque girl, endless long legs, glowing fair skin, soft pastel pink-to-blue gradient hair, and bright sapphire eyes",
    "elegant tall stature, toned long legs, pristine white skin, vibrant cherry-red hair, and intense golden-amber eyes",
    "supermodel frame, long legs, glowing skin with a natural sheen, sleek silver-white hair, and striking violet eyes",
    "tall and athletic build, long legs, luminous white skin, voluminous copper-red hair, and deep mahogany eyes",
    "dramatic tall silhouette, long elegant legs, radiant skin, silky midnight-blue hair, and icy blue eyes",
    "long layered hair with a vibrant blue-to-pink ombre, glowing porcelain skin, and sharp violet eyes",
    "short neon pink bob with electric blue highlights, fair glowing skin, fit frame, and dramatic smoky eyeliner",
    "peekaboo style hair: jet-black on top with hidden bright pink under-layers, radiant glowing skin, and dark amber eyes",
    "soft wavy hair with an iridescent pink and lavender mix, silky white skin, and shimmering silver-grey eyes",
    "long vibrant magenta hair with deep purple highlights, glowing olive-fair skin, and striking copper eyes",
    "fiery sunset-orange hair, radiant white skin, fit and toned frame, and intense emerald-green eyes",
    "blunt-cut blonde hair with pastel pink streaks, dewy skin, tall model frame, and heterochromia (one blue, one gold eye)",
    "long silky midnight-black hair, flawless ivory-white skin, elegant posture, and soft, soulful eyes",
    "deep burgundy waves, luminous glowing skin, sophisticated silhouette, and deep, warm copper-toned eyes",
    "shimmering platinum blonde hair, porcelain skin with a soft natural glow, tall frame, and clear ice-blue eyes",
    "natural chestnut brown hair, clear radiant white skin, tall build, and soft hazel eyes",
    "vibrant fire-engine red hair, dewy porcelain skin, long graceful legs, and bright, youthful green eyes",
    "classic honey-blonde hair, glowing sun-kissed skin, athletic height, and sparkling sky-blue eyes",
    "sleek dark chocolate hair, pale glowing skin, model-like stature, and deep, dark brown eyes",
    "long platinum hair with soft white highlights, ethereal white skin, model tall frame, and glowing cyan eyes",
    "messy top-knot hair, pastel pink and sky-blue highlights, glowing skin, and soft, friendly gaze",
    "voluminous red-gold curls, luminous white skin, statuesque height, and piercing amber eyes",
    "sleek raven-black hair, dewy skin with a light-from-within glow, tall build, and sophisticated gaze",
    "wavy ash-blonde hair, radiant porcelain skin, tall model legs, and deep, mesmerizing lilac eyes"
]
OUTFIT_OPTIONS = [
    "white button-up shirt with rolled sleeves, dark slim-fit jeans",
    "oversized heather-grey hoodie, distressed light-wash denim",
    "textured wool sweater, tailored chinos, clean white sneakers",
    "flannel shirt over a plain white tee, comfortable black joggers",
    "minimalist crewneck sweatshirt, relaxed-fit denim, casual vibe",
    "tiny black bikini with gold jewelry",
    "red string bikini with thigh straps",
    "white lace swimsuit with cutouts and frills",
    "sheer beach cover-up over a bikini",
    "metallic bronze high-cut bikini with gold chains",
    "velvet emerald green monokini with deep plunge",
    "see-through lace bodysuit with intricate floral embroidery",
    "satin silk lingerie set with garter belt and stockings",
    "delicate floral lace bralette and matching high-waisted briefs",
    "monokini with side cutouts and metallic hardware accents",
    "glossy fitted black latex bodysuit with silver zipper details",
    "low-cut black velvet cocktail dress with an extreme thigh-high slit",
    "sheer mesh bodycon dress with strategically placed embellishments",
    "leather corset top paired with a micro-mini leather skirt",
    "plunging V-neck silk slip dress with delicate lace trim",
    "cut-out bodycon dress with crisscross lace-up side panels",
    "backless sequined mini dress with a plunging neckline",
    "high-slit silk gown in deep crimson with an open back",
    "suede bodycon dress with elaborate lace-up front detail",
    "bunny-girl inspired leotard with fishnet stockings and satin collar",
    "cyberpunk-style cropped tactical top with buckles and sheer paneling",
    "high-collared cheongsam-inspired mini dress with extreme side slits",
    "sheer organza bralette with a high-waisted transparent skirt",
    "tight-fitting vinyl catsuit with a daring front zipper",
    "crocheted string bikini with dangling bead accents",
    "off-shoulder corset crop top and ultra-short pleated skirt",
    "futuristic holographic bodysuit with metallic trim",
    "gothic-inspired lace corset with ruffled layered skirt"
]

BACKGROUND_OPTIONS = [
    "bright, modern kitchen with marble countertops and soft morning sunlight",
    "cozy, cluttered bedroom with fairy lights and warm, romantic mood lighting",
    "luxurious walk-in closet with velvet seating and golden ambient lighting",
    "sun-drenched bathroom with marble textures and soft-focus greenery",
    "spacious home library with floor-to-ceiling mahogany bookshelves and soft lamp glow",
    "airy living room with linen curtains billowing in the breeze, soft afternoon sun",
    "sunny cafe with floor-to-ceiling windows and blurred street bokeh",
    "neon-drenched city rooftop at night with vibrant pink and blue city lights",
    "sleek, minimalist art gallery with white walls and sharp architectural shadows",
    "industrial loft apartment with exposed brick and moody, cinematic lighting",
    "bustling neon-lit shopping district with rain-slicked pavement and vibrant reflections",
    "chic hotel balcony overlooking a dense, glowing cityscape at twilight",
    "tropical beach at sunset with golden hour light and silhouettes of palm trees",
    "serene flower garden at golden hour with soft glowing bokeh background",
    "misty mountain forest with shafts of sunlight filtering through the trees",
    "secluded cliff-side viewpoint overlooking a calm turquoise ocean",
    "enchanted lavender field under a soft twilight sky, dreamy atmosphere",
    "luxurious resort pool deck with blue water reflections and manicured palms",
    "majestic Angkor Wat temple silhouette during a vibrant purple-orange sunrise",
    "traditional stone pagoda exterior with intricate patterns and soft dappled sunlight",
    "bustling, rain-slicked Tokyo street at night with glowing signs and reflections",
    "high-end vintage study with mahogany shelves, soft firelight, and dust motes",
    "quiet, traditional tea house interior with sliding shoji screens and soft shadows",
    "grand ancient stone courtyard with moss-covered pillars and dramatic directional light"
]

POSE_OPTIONS = [
    "seductive confident pose, running fingers through hair, direct gaze",
    "glamorous high-fashion pose, arched back, one hand resting on thigh",
    "standing full-body power pose, one hand on hip, confident stance",
    "fierce editorial pose, looking down slightly with a sharp, piercing stare",
    "dynamic movement pose, spinning around with hair caught in the breeze",
    "hands in pockets, casual and confident stance, relaxed shoulder posture",
    "stretching arms gracefully above head, elongated frame, looking at camera",
    "leaning against a wall, crossing one leg over the other, relaxed and cool",
    "playful wink, leaning forward toward the camera with a bright smile",
    "over-the-shoulder look, teasing smile, looking back at the camera",
    "cute flirty expression, head tilted, one finger near lips",
    "candid laugh, covering mouth slightly with a playful, natural gaze",
    "winking and pouting, playful hand gestures framing the face",
    "holding a strand of hair, playful sideways glance, inviting smile",
    "blowing a kiss toward the camera, one hand extended playfully",
    "relaxed sitting pose, legs crossed elegantly, leaning back on hands",
    "reclining on a surface, looking up at the camera with a sultry expression",
    "sprawled gracefully, natural posture, one hand touching the floor",
    "kneeling pose, leaning slightly forward, soft and inviting expression",
    "soft, dreamy pose, arms wrapped around self, looking away from camera",
    "curled up on a chair, resting chin on knees, soft and comfortable",
    "lying on stomach, propped up by elbows, looking directly into the lens",
    "resting head gently against a window, pensive and serene expression",
    "walking naturally toward the camera, caught in a candid moment",
    "mid-laugh, turning to look back at the camera, natural hair movement",
    "brushing hair behind ear, looking down with a shy, graceful smile",
    "adjusting jacket lapel, cool and focused gaze, slight smirk",
    "standing in a park, soft breeze blowing through hair, tranquil expression"
]

STYLE_OPTIONS = [
    "high detail anime illustration, sharp lines, cinematic lighting",
    "ultra high quality, crisp vector-style art, vibrant color palette",
    "4k anime portrait, intricate shading, masterpiece quality",
    "vibrant waifu art, clean linework, professional digital finish",
    "glossy anime pin-up style, high contrast, smooth highlights",
    "detailed character design sheet, clean background, sharp focus",
    "photorealistic anime render, ray-traced reflections, high-end production",
    "soft painterly anime rendering, ethereal atmosphere, brushwork textures",
    "beautiful visual novel key art, high-end production, detailed background",
    "cel-shaded masterpiece, vibrant saturated colors, professional anime studio style",
    "watercolor-infused anime style, dreamy lighting, delicate soft edges",
    "digital oil painting style, thick impasto textures, vivid anime character",
    "artistic concept art, loose brushstrokes, atmospheric anime aesthetic",
    "manga-inspired sketch with subtle digital coloring, expressive line weights",
    "thick-lined bold anime style, high fashion aesthetic, poster art quality",
    "dramatic low-key lighting, neon color accents, edgy modern anime style",
    "soft-focus romantic anime portrait, lens flare, gentle glowing highlights",
    "retro 90s aesthetic anime style, film grain, muted nostalgic color grading",
    "highly detailed CG anime style, volumetric lighting, rich environmental depth",
    "dusk-lit anime portrait, deep blue shadows, warm golden rim lighting",
    "glitch-art inspired anime, chromatic aberration, vibrant high-contrast style",
    "minimalist anime aesthetic, flat design elements, high-contrast silhouette",
    "hyper-stylized aesthetic, bioluminescent accents, ethereal glow, cinematic wide shot"
]

DEFAULT_IMAGE_PROMPT_OPTIONS = [
    DEFAULT_IMAGE_PROMPT,
    "masterpiece anime character art, full body composition, head-to-toe visible, clean line art",
    "full body anime character illustration, feet visible, expressive face, crisp cel shading",
    "cute anime character design, full body standing composition, clean manga line art",
]

REALISTIC_IMAGE_PROMPT_OPTIONS = [
    REALISTIC_IMAGE_PROMPT,
    "RAW DSLR photo of a real adult age 19 or older, photorealistic full body, natural skin texture",
    "realistic full body portrait photo of a cute adult person age 19 or older, natural face, real skin pores",
    "professional full body lifestyle photo, real adult human age 19 or older, natural daylight, sharp realistic detail",
]

REALISTIC_ETHNICITY_OPTIONS = [
    "Khmer Cambodian woman",
    "Japanese woman",
    "Korean woman",
    "Chinese woman",
    "Mongolian woman",
    "Kazakhstani woman",
    "Uzbekistani woman",
    "Tajikistani woman",
    "white Caucasian woman",
    "Russian woman",
    "Scandinavian woman",
    "German woman",
    "French woman",
    "British woman",
    "Irish woman",
    "Ukrainian woman",
    "Polish woman",
    "Dutch woman",
    "Italian woman",
    "Greek woman",
    "Swiss woman",
]

MALE_ETHNICITY_OPTIONS = [
    "Khmer Cambodian man",
    "Japanese man",
    "Korean man",
    "Chinese man",
    "Mongolian man",
    "Kazakhstani man",
    "Uzbekistani man",
    "Tajikistani man",
    "Vietnamese man",
    "white Caucasian man",
    "Russian man",
    "Scandinavian man",
    "German man",
    "French man",
    "British man",
    "Irish man",
    "Ukrainian man",
    "Polish man",
    "Dutch man",
    "Italian man",
    "Greek man",
    "Swiss man",
    "Swedish man",
    "Norwegian man"
]

REALISTIC_APPEARANCE_OPTIONS = [
    "long natural black hair, warm brown eyes, cute baby face, soft skin texture",
    "soft wavy brunette hair, brown eyes, sweet youthful face, freckles on nose",
    "straight dark bob haircut, gentle eyes, cute small smile, natural glow",
    "long auburn hair, hazel eyes, soft round cheeks, youthful radiant complexion",
    "natural blonde hair, blue eyes, sweet cute smile, peach fuzz skin texture",
    "long dark brown hair, amber eyes, fresh youthful face, natural skin pores",
    "wavy strawberry blonde hair, grey-green eyes, soft features, photorealistic skin",
    "long silky hair with a pink-to-blue gradient, porcelain skin, youthful face, glowing highlights",
    "sleek raven-black hair, deep emerald eyes, porcelain skin, soft natural sheen",
    "voluminous red-gold curls, bright hazel eyes, freckled skin, youthful energetic expression",
    "short textured black hair, piercing brown eyes, defined jawline, youthful look",
    "tousled dark brown hair, warm hazel eyes, friendly smile, natural skin texture",
    "neat undercut hairstyle, deep brown eyes, sharp facial features, clean-cut look",
    "shaggy blonde hair, sky-blue eyes, youthful grin, realistic skin detail",
    "natural chestnut hair, soft brown eyes, subtle stubble, youthful masculine aesthetic",
    "short straight black hair, dark almond eyes, clear skin, athletic build",
    "messy ash-brown hair, cool grey eyes, confident expression, photorealistic skin",
    "brushed-back dark hair, intense amber eyes, ruggedly handsome features, natural glow",
    "short silver-toned hair, sharp icy-blue eyes, defined facial structure, clean pores",
    "wavy mocha-brown hair, deep mahogany eyes, friendly youthful gaze, sun-kissed skin texture"
]

REALISTIC_OUTFIT_OPTIONS = [
    "cute white summer dress, minimalist silver necklace, natural fabric texture",
    "pastel cardigan over a simple white top, navy pleated skirt, soft knit detail",
    "casual light blue blouse, high-waisted denim skirt, realistic denim weave",
    "soft pink knitted sweater, slim-fit blue jeans, natural makeup look",
    "simple floral print sundress, delicate fabric, sunny-day vibe",
    "oversized pastel hoodie, short pleated skirt, casual cute student style",
    "cream-colored ribbed tank top, high-waisted linen shorts, braided hair",
    "white lace-trimmed blouse, soft beige skirt, delicate gold accessories",
    "off-the-shoulder floral blouse, high-waisted white pants, breezy summer style",
    "oversized denim jacket, black slip dress, layered gold jewelry, trendy aesthetic",
    "clean white button-up shirt, casual fit, rolled sleeves, cotton texture",
    "oversized graphic crewneck sweater, dark denim jeans, relaxed fit",
    "smart-casual polo shirt, khaki chinos, classic student aesthetic",
    "lightweight denim jacket, plain white t-shirt, black joggers",
    "hoodie with minimalist design, distressed denim, youthful urban look",
    "button-down flannel shirt over a basic tee, casual and approachable",
    "fitted sweater vest over a collared shirt, modern collegiate style",
    "breathable linen shirt, comfortable trousers, clean and natural look",
    "leather bomber jacket, grey crewneck sweater, dark slim-fit jeans",
    "layered long-sleeve tee under a short-sleeve shirt, relaxed urban streetwear"
]

MALE_APPEARANCE_OPTIONS = [
    "short textured black hair, warm amber eyes, clear youthful skin, defined jawline",
    "messy tousled brunette hair, soft brown eyes, natural skin texture with slight pores",
    "neat short dark hair, deep brown eyes, friendly expression, clean-cut aesthetic",
    "shaggy auburn hair, gold-flecked eyes, freckled fair skin, youthful features",
    "natural chestnut brown hair, warm hazel eyes, soft natural skin, approachable smile",
    "wavy dark brown hair, dark coffee eyes, athletic build, radiant natural skin",
    "classic short side-part black hair, deep brown eyes, sharp facial features, clean look",
    "short silver hair, piercing blue eyes, pale porcelain skin, sharp facial structure",
    "asymmetrical black hair, cool grey eyes, subtle youthful charm, photorealistic skin",
    "short ash-blonde hair, light blue eyes, athletic facial features, natural tan skin",
    "wavy dark espresso hair, dark almond eyes, clear skin, focused gaze",
    "swept-back dark brown hair, emerald green eyes, defined cheekbones, natural complexion",
    "short mahogany red hair, striking gold eyes, youthful glow, authentic skin pores",
    "bleached-tip textured hair, dark brown eyes, confident smirk, realistic skin detail",
    "undercut hairstyle with silver-blue highlights, pale skin, piercing heterochromia eyes",
    "messy raven-black shaggy hair, soft violet eyes, artistic youthful aesthetic",
    "long layered chestnut hair, warm hazel eyes, rugged but youthful skin texture"
]

MALE_OUTFIT_OPTIONS = [
    "white button-up shirt with rolled sleeves, dark slim-fit jeans",
    "oversized heather-grey hoodie, distressed light-wash denim",
    "textured wool sweater, tailored chinos, clean white sneakers",
    "flannel shirt over a plain white tee, comfortable black joggers",
    "minimalist crewneck sweatshirt, relaxed-fit denim, casual vibe",
    "collegiate letterman jacket, grey hoodie, relaxed fit dark jeans",
    "striped long-sleeve tee, cargo pants, chunky sneakers",
    "stylish black bomber jacket, black fitted pants, high-top leather boots",
    "modern techwear-inspired jacket with multiple pockets, dark tactical trousers",
    "leather biker jacket, slim-fit dark denim, metal accents",
    "oversized denim jacket with shearling collar, charcoal fitted jeans",
    "sleek longline trench coat, turtleneck sweater, slim dress trousers",
    "distressed black denim jacket, graphic print tee, leather Chelsea boots",
    "layered oversized knit sweater, ripped black skinny jeans, combat boots",
    "simple fitted navy suit, crisp white shirt, no tie, polished leather shoes",
    "charcoal blazer over a black high-neck shirt, tailored trousers",
    "classic waistcoat and dress shirt combo, professional student aesthetic",
    "smart-casual beige linen blazer, white shirt, light-toned dress pants",
    "tailored grey overcoat, navy turtleneck, dark slim-fit slacks",
    "fitted velvet blazer, silk dress shirt, black formal trousers"
]
REALISTIC_MALE_APPEARANCE_OPTIONS = [
    "short natural black hair, warm brown eyes, cute youthful face, clear skin, natural skin pores",
    "soft brunette hair, brown eyes, gentle smile, fresh-faced appearance, slight natural glow",
    "straight dark hair, natural face, cute small smile, soft facial features, realistic skin texture",
    "short dark brown hair, amber eyes, fresh youthful face, clean-cut aesthetic",
    "textured tousled hair, soft hazel eyes, friendly expression, natural skin tone",
    "neat side-parted hair, deep brown eyes, youthful face, light natural freckles",
    "shaggy dark hair, clear blue eyes, soft round features, natural skin detail",
    "short wavy chestnut hair, warm amber eyes, approachable youthful look, authentic skin texture",
    "effortless black hair, kind brown eyes, natural skin with subtle imperfections, clean look",
    "styled dark brown hair, gentle expression, youthful radiant skin, high-resolution skin texture",
    "textured ash-blonde hair, piercing light-blue eyes, sharp jawline, stubble, photorealistic skin pores",
    "undercut hairstyle, dark espresso hair, intense gaze, defined cheekbones, natural skin grain",
    "messy silver-grey hair, cool grey eyes, rugged facial features, authentic skin texture",
    "slicked-back raven hair, deep mahogany eyes, confident expression, subtle skin imperfections",
    "faded side hair, tousled top, striking gold-flecked eyes, masculine facial structure, realistic skin",
    "long dark hair tied back, sharp facial features, deep amber eyes, detailed natural skin",
    "short copper-toned hair, bright green eyes, strong jaw, natural sun-kissed skin texture",
    "spiky dark brown hair, cool hazel eyes, intense and focused gaze, authentic pore detail",
    "clean-cut dark hair, thoughtful gaze, refined facial structure, high-definition skin texture",
    "natural textured black hair, piercing dark eyes, slight stubble, masculine and mature look"
]

REALISTIC_MALE_OUTFIT_OPTIONS = [
    "casual white linen shirt with sleeves rolled up, classic blue jeans, natural fabric texture",
    "soft knitted beige sweater, fitted charcoal pants, minimalist aesthetic",
    "clean black bomber jacket, simple grey cotton t-shirt, relaxed denim jeans",
    "oversized pastel hoodie, slim-fit dark blue jeans, casual student vibe",
    "soft cotton polo shirt in sage green, khaki trousers, clean-cut look",
    "lightweight denim jacket over a white crewneck tee, black fitted jeans",
    "relaxed-fit grey sweatshirt, navy chino pants, authentic clothing textures",
    "white oxford button-down shirt, dark indigo jeans, smart-casual style",
    "textured navy blue cardigan, white t-shirt, slim-fit trousers",
    "modern minimalist hoodie in earthy tones, comfortable tapered pants, natural look",
    "oversized black graphic hoodie, distressed light-wash denim, tactical fabric detail",
    "high-collar techwear jacket, black utility cargo pants, nylon material texture",
    "distressed denim trucker jacket, white layering tee, charcoal skinny jeans",
    "layered oversized knit sweater, ripped black slim jeans, wool fabric detail",
    "bomber jacket with flight patches, dark tapered jeans, authentic bomber nylon texture",
    "tailored charcoal blazer, black turtleneck, slim-fit wool dress pants",
    "fitted navy blue overcoat, white dress shirt, dark grey slacks, cashmere texture",
    "smart-casual beige linen blazer, white shirt, light-toned chinos",
    "structured suede bomber jacket, black crewneck knit, dark indigo denim",
    "classic waistcoat layered over a crisp button-down, charcoal trousers, linen blend texture"
]

REALISTIC_MALE_ETHNICITY_OPTIONS = [
    "Khmer Cambodian man",
    "Japanese man",
    "Korean man",
    "Chinese man",
    "Mongolian man",
    "Vietnamese man",
    "Kazakhstani man",
    "Uzbekistani man",
    "white Caucasian man",
    "Russian man",
    "Scandinavian man",
    "German man",
    "French man",
    "British man",
    "Irish man",
    "Ukrainian man",
    "Polish man",
    "Dutch man",
    "Italian man",
    "Greek man",
    "Swiss man"
]

COUPLE_OUTFIT_OPTIONS = [
    "matching casual outfits: light-wash denim jackets over white tees and black jeans",
    "coordinated date outfits: soft cream-colored knit sweaters and charcoal trousers",
    "simple stylish everyday outfits: matching grey hoodies and black slim-fit denim",
    "cozy aesthetic: earth-toned oversized sweaters and soft-touch cotton pants",
    "laid-back weekend vibes: matching striped long-sleeve tees and relaxed cargo pants",
    "coordinated formal: navy blue blazer for him, navy silk midi dress for her",
    "smart-casual harmony: white button-up shirts and beige linen bottoms",
    "monochromatic chic: both wearing high-end charcoal coats and black turtlenecks",
    "summer vibe: white linen shirts for him, white floral sundress for her",
    "evening elegance: him in a charcoal suit, her in a velvet dark-red evening gown",
    "urban street style: matching black leather jackets and distressed denim",
    "minimalist monochrome: both in clean white t-shirts and olive cargo pants",
    "tech-wear aesthetic: matching dark tactical vests and black joggers",
    "retro collegiate: matching varsity jackets with vintage denim jeans",
    "grunge aesthetic: both in oversized flannel shirts and ripped black jeans",
    "modern aesthetic: him in an olive bomber jacket, her in a matching olive trench coat"
]

REALISTIC_BACKGROUND_OPTIONS = [
    "sunlit city street with blurred urban architecture and soft bokeh",
    "cozy modern cafe interior with warm ambient lighting and soft-focus coffee tables",
    "cute shopping street during golden hour with gentle lens flare and soft sunlight",
    "minimalist art gallery hallway with clean lines and directional natural light",
    "rooftop terrace overlooking a city skyline at dusk with shimmering bokeh city lights",
    "chic industrial loft with exposed brick, large windows, and cool morning light",
    "vibrant neon-lit street at night with rain-slicked pavement and colorful reflections",
    "bright, airy bedroom with sheer white curtains diffusing soft daylight",
    "modern sunroom filled with lush indoor plants and dappled, natural lighting",
    "spacious home library with mahogany shelving and cozy, warm atmospheric light",
    "bright kitchen with marble accents and soft, early morning sun rays",
    "cozy reading nook by a large window with soft, warm golden hour glow",
    "elegant living room with velvet seating and fireplace embers illuminating the room",
    "tropical beach at golden hour with soft orange sky and tranquil turquoise water",
    "flower garden in soft evening light with vibrant blossoms and ethereal depth of field",
    "misty forest clearing with shafts of sunlight filtering through lush greenery",
    "secluded cliffside lookout with panoramic ocean views and vibrant twilight colors",
    "charming cobblestone alleyway in a historic district with soft, overcast natural lighting",
    "serene lakeside park during sunset with long shadows and soft, golden light",
    "mountain vista at high altitude with clear blue sky and soft, bright natural light"
]

REALISTIC_POSE_OPTIONS = [
    "standing relaxed with a shy cute smile, hands lightly clasped in front",
    "walking naturally toward the camera, caught in a candid moment with a gentle smile",
    "standing full body casual pose, relaxed posture with weight on one leg",
    "casual candid pose with a natural sweet expression, looking slightly off-camera",
    "natural walking pose, looking over shoulder with a friendly, subtle grin",
    "leaning back against a wall, relaxed posture, arms crossed loosely",
    "candid shot of laughing, covering mouth slightly, genuine eye crinkles",
    "soft smile looking directly at the camera, head tilted slightly, inviting expression",
    "gentle peace sign pose, cute expression, relaxed shoulder posture",
    "standing facing the camera, hands in pockets, casual and confident stance",
    "softly waving hello, candid natural smile, youthful and approachable",
    "sitting on a chair with hands resting on knees, relaxed and comfortable posture",
    "looking down with a shy smile, hair tucking behind ear, intimate and natural",
    "standing in a park, soft breeze blowing through hair, tranquil expression",
    "adjusting wristwatch, focused natural look, slight downward gaze",
    "holding a coffee cup with both hands, warm cozy expression, gentle smile",
    "full-body portrait, hands placed gracefully at sides, relaxed shoulder alignment",
    "seated on steps, one knee up, looking toward the camera with a calm gaze",
    "leaning against a doorway, one arm raised to frame the face, soft natural posture",
    "walking away from camera, turning head back for a candid profile shot",
    "resting hands gently on a table surface, relaxed finger placement, soft focus",
    "standing with one hand on hip, natural and confident stance, soft gaze"
]

REALISTIC_STYLE_OPTIONS = [
    "RAW photo, 50mm lens, f/2.8, natural daylight, sharp focus on eyes",
    "professional DSLR photo, 85mm lens, realistic color grading, creamy bokeh",
    "full body fashion photography, 35mm lens, sharp focus, high-end editorial lighting",
    "cinematic medium shot, 50mm, f/1.8, shallow depth of field, natural falloff",
    "studio portrait, Rembrandt lighting, softbox diffusion, sharp focus on facial features",
    "candid lifestyle photography, natural skin pores, unedited look, natural indoor lighting",
    "documentary street photo, 35mm lens, realistic lighting, authentic atmospheric depth",
    "phone camera photo, realistic imperfect skin, flash-on aesthetic, raw candid capture",
    "vintage film photography style, 35mm, subtle film grain, natural color palette",
    "candid journalistic style, fast shutter speed, natural motion blur, authentic vibe",
    "wide angle environmental portrait, 24mm lens, deep depth of field, natural environment",
    "macro-detail shot, focus on skin texture and fabric weave, soft natural lighting",
    "golden hour outdoor portrait, 85mm, warm natural light, soft lens flare",
    "overcast natural light, flat lighting, realistic color rendition, high-resolution detail",
    "backlit cinematic portrait, natural rim lighting, lens flare, high dynamic range (HDR)",
    "twilight environmental portrait, ambient city light, soft cool shadows, realistic atmosphere"
]

PEOPLE_COUNT_OPTIONS = ["1 person", "2 people", "Random"]
SINGLE_GENDER_OPTIONS = ["Only girl", "Only woman", "Only boy", "Only man"]
GROUP_GENDER_OPTIONS = ["Only girl", "Only woman", "Only boy", "Only man", "Girl and boy", "Woman and man", "Random"]
GENDER_MIX_OPTIONS = GROUP_GENDER_OPTIONS
ACTION_OPTIONS = ["Walking", "Running", "Sitting", "Standing", "Hugging", "Kissing", "Random"]

if DATA_FILE.exists():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        CHARACTERS = json.load(file)
else:
    CHARACTERS = {
        "Luna": {
            "name": "Luna",
            "avatar": "luna.png",
            "bio": "Sweet, flirty, and teasing 18-year-old AI girlfriend",
            "age": 18,
            "personality": "Playful, Affectionate, Teasing",
            "language": "English",
        }
    }


def save_characters():
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(CHARACTERS, file, indent=2)


def get_image_characters():
    return list(CHARACTERS.keys())


def get_default_image_character():
    if "Luna" in CHARACTERS:
        return "Luna"
    return next(iter(CHARACTERS), None)


def get_avatar_path(character):
    avatar = CHARACTERS[character].get("avatar", "")
    avatar_path = Path(avatar)
    if avatar and avatar_path.exists():
        return str(avatar_path.resolve())
    local_avatar_path = BASE_DIR / avatar
    if avatar and local_avatar_path.exists():
        return str(local_avatar_path.resolve())
    return None


def load_font(size, bold=False):
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def options_to_text(options):
    return "\n".join(options)


def parse_options_text(value, fallback):
    options = [line.strip() for line in (value or "").splitlines() if line.strip()]
    return options or list(fallback)


def resolve_radio_choice(value, options, rng):
    if value == "Random" or value not in options:
        choices = [option for option in options if option != "Random"]
        return rng.choice(choices)
    return value


def gender_choices_for_people(people_count):
    if people_count == "1 person":
        return SINGLE_GENDER_OPTIONS
    return GROUP_GENDER_OPTIONS


def normalize_gender_for_people(people_count, gender_mix):
    choices = gender_choices_for_people(people_count)
    if gender_mix in choices:
        return gender_mix
    return choices[0]


def gender_field_visibility(people_count, gender_mix):
    gender_mix = normalize_gender_for_people(people_count, gender_mix)
    show_female = True
    show_male = True
    if people_count == "1 person":
        show_female = gender_mix in ("Only girl", "Only woman")
        show_male = gender_mix in ("Only boy", "Only man")
    elif gender_mix in ("Only girl", "Only woman"):
        show_male = False
    elif gender_mix in ("Only boy", "Only man"):
        show_female = False
    return show_female, show_male


def update_gender_choices(people_count, gender_mix):
    gender_mix = normalize_gender_for_people(people_count, gender_mix)
    choices = gender_choices_for_people(people_count)
    show_female, show_male = gender_field_visibility(people_count, gender_mix)
    return (
        gr.update(choices=choices, value=gender_mix),
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
    )


def update_prompt_option_visibility(people_count, gender_mix):
    show_female, show_male = gender_field_visibility(people_count, gender_mix)
    return (
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
    )


def update_prompt_option_button_visibility(people_count, gender_mix):
    show_female, show_male = gender_field_visibility(people_count, gender_mix)
    return (
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_female),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
        gr.update(visible=show_male),
    )


def action_prompt(action, people_count):
    action_text = (action or "").strip()
    if not action_text:
        return ""
    action_key = action_text.lower()
    if action_key == "random":
        action_key = random.choice([option.lower() for option in ACTION_OPTIONS if option != "Random"])

    if people_count == "2 people":
        if action_key == "walking":
            return "walking side by side"
        if action_key == "running":
            return "running side by side"
        if action_key == "sitting":
            return "sitting together"
        if action_key == "hugging":
            return "hugging each other"
        if action_key == "kissing":
            return "sweet romantic kiss, non-explicit"
        if action_key == "standing":
            return "standing side by side"
        return action_text

    if action_key == "walking":
        return "walking naturally toward the camera"
    if action_key == "running":
        return "running with dynamic movement"
    if action_key == "sitting":
        return "sitting pose, full body visible"
    if action_key == "hugging":
        return "standing with arms softly folded"
    if action_key == "kissing":
        return "blowing a cute kiss toward the camera"
    if action_key == "standing":
        return "standing full body pose"
    return action_text


def subject_prompt(people_count, gender_mix, age):
    adult_age = f"age {age} or older"
    if people_count == "1 person":
        if gender_mix in ("Only boy", "Only man"):
            return f"one adult man {adult_age}"
        return f"one adult woman {adult_age}"

    if gender_mix in ("Only boy", "Only man"):
        return f"two adult men {adult_age}"
    if gender_mix in ("Girl and boy", "Woman and man"):
        return f"one adult woman and one adult man {adult_age}"
    return f"two adult women {adult_age}"


def group_instruction(people_count, gender_mix):
    if people_count != "2 people":
        return "single main subject, no background people"
    if gender_mix in ("Girl and boy", "Woman and man"):
        return "two main subjects only, woman and man, both visible"
    return "two main subjects only, both visible"


def adapt_default_prompt(default_prompt, people_count, gender_mix, image_style):
    prompt = (default_prompt or "").strip()
    if people_count == "2 people":
        if gender_mix in ("Girl and boy", "Woman and man"):
            if image_style == "Realistic":
                return "RAW DSLR photo of a real adult woman and adult man"
            return "full body couple character art"
        if gender_mix in ("Only boy", "Only man"):
            if image_style == "Realistic":
                return "RAW DSLR photo of two real adult men"
            return "full body character art of two men"
        if image_style == "Realistic":
            return "RAW DSLR photo of two real adult women"
        return "full body character art of two women"
    if gender_mix in ("Only boy", "Only man"):
        if image_style == "Realistic":
            return "RAW DSLR photo of a real adult man age 19 or older"
        return "adult man, full body character art, feet visible"
    return prompt


def adapt_gendered_text(value, people_count, gender_mix):
    if not value:
        return value
    if people_count == "2 people":
        if gender_mix in ("Only boy", "Only man"):
            return value.replace("woman", "men").replace("girl", "men")
        if gender_mix in ("Girl and boy", "Woman and man"):
            return value.replace("woman", "adult woman and adult man").replace("girl", "adult woman and adult man")
        return value.replace("woman", "women").replace("girl", "women")
    if gender_mix in ("Only boy", "Only man"):
        return value.replace("woman", "man").replace("girl", "man")
    return value


def image_defaults_for_style(image_style):
    if image_style == "Realistic":
        return {
            "default_prompt": REALISTIC_IMAGE_PROMPT,
            "ethnicity_options": REALISTIC_ETHNICITY_OPTIONS,
            "appearance_options": REALISTIC_APPEARANCE_OPTIONS,
            "outfit_options": REALISTIC_OUTFIT_OPTIONS,
            "male_ethnicity_options": REALISTIC_MALE_ETHNICITY_OPTIONS,
            "male_appearance_options": REALISTIC_MALE_APPEARANCE_OPTIONS,
            "male_outfit_options": REALISTIC_MALE_OUTFIT_OPTIONS,
            "background_options": REALISTIC_BACKGROUND_OPTIONS,
            "pose_options": REALISTIC_POSE_OPTIONS,
            "style_options": REALISTIC_STYLE_OPTIONS,
        }
    return {
        "default_prompt": DEFAULT_IMAGE_PROMPT,
        "ethnicity_options": ETHNICITY_OPTIONS,
        "appearance_options": APPEARANCE_OPTIONS,
        "outfit_options": OUTFIT_OPTIONS,
        "male_ethnicity_options": MALE_ETHNICITY_OPTIONS,
        "male_appearance_options": MALE_APPEARANCE_OPTIONS,
        "male_outfit_options": MALE_OUTFIT_OPTIONS,
        "background_options": BACKGROUND_OPTIONS,
        "pose_options": POSE_OPTIONS,
        "style_options": STYLE_OPTIONS,
    }


def default_prompt_options_for_style(image_style):
    if image_style == "Realistic":
        return REALISTIC_IMAGE_PROMPT_OPTIONS
    return DEFAULT_IMAGE_PROMPT_OPTIONS


def random_single_prompt_option(image_style, option_key):
    if option_key == "default_prompt":
        return random.choice(default_prompt_options_for_style(image_style))
    defaults = image_defaults_for_style(image_style)
    return random.choice(defaults[option_key])


def random_single_video_prompt_option(video_style, option_key):
    return random_single_prompt_option(image_style_from_video_style(video_style), option_key)


def build_random_image_prompt(
    character,
    request_text="",
    image_style="Anime",
    people_count="1 person",
    gender_mix="Only girl",
    action="Standing",
    default_prompt=DEFAULT_IMAGE_PROMPT,
    ethnicity_options_text=None,
    appearance_options_text=None,
    outfit_options_text=None,
    male_ethnicity_options_text=None,
    male_appearance_options_text=None,
    male_outfit_options_text=None,
    background_options_text=None,
    pose_options_text=None,
    style_options_text=None,
):
    char = CHARACTERS[character]
    rng = random.Random(f"{character}-{request_text}-{time.time()}")
    age = max(int(char.get("age", 19) or 19), 19)
    people_count = resolve_radio_choice(people_count, PEOPLE_COUNT_OPTIONS, rng)
    gender_mix = normalize_gender_for_people(people_count, gender_mix)
    gender_mix = resolve_radio_choice(gender_mix, GENDER_MIX_OPTIONS, rng)
    defaults = image_defaults_for_style(image_style)
    ethnicity_options = parse_options_text(ethnicity_options_text, defaults["ethnicity_options"])
    appearance_options = parse_options_text(appearance_options_text, defaults["appearance_options"])
    outfit_options = parse_options_text(outfit_options_text, defaults["outfit_options"])
    male_ethnicity_options = parse_options_text(male_ethnicity_options_text, defaults["male_ethnicity_options"])
    male_appearance_options = parse_options_text(male_appearance_options_text, defaults["male_appearance_options"])
    male_outfit_options = parse_options_text(male_outfit_options_text, defaults["male_outfit_options"])
    background_options = parse_options_text(background_options_text, defaults["background_options"])
    pose_options = parse_options_text(pose_options_text, defaults["pose_options"])
    style_options = parse_options_text(style_options_text, defaults["style_options"])
    ethnicity = rng.choice(ethnicity_options)
    appearance = rng.choice(appearance_options)
    outfit = rng.choice(outfit_options)
    male_ethnicity = rng.choice(male_ethnicity_options)
    male_appearance = rng.choice(male_appearance_options)
    male_outfit = rng.choice(male_outfit_options)
    background = rng.choice(background_options)
    pose = action_prompt(action, people_count)
    style_option = rng.choice(style_options)
    subject = subject_prompt(people_count, gender_mix, age)
    instruction = group_instruction(people_count, gender_mix)
    if gender_mix in ("Only boy", "Only man"):
        ethnicity = male_ethnicity
        appearance = male_appearance
        outfit = male_outfit
    elif people_count == "2 people" and gender_mix in ("Girl and boy", "Woman and man"):
        outfit = f"woman wearing {outfit}, man wearing {male_outfit}"
        ethnicity = f"woman: {ethnicity}; man: {male_ethnicity}"
        appearance = f"woman: {appearance}; man: {male_appearance}"
    else:
        ethnicity = adapt_gendered_text(ethnicity, people_count, gender_mix)
        appearance = adapt_gendered_text(appearance, people_count, gender_mix)
        outfit = adapt_gendered_text(outfit, people_count, gender_mix)
    if not pose or (action or "").strip().lower() == "standing":
        pose = rng.choice(pose_options)

    default_prompt = adapt_default_prompt(default_prompt or defaults["default_prompt"], people_count, gender_mix, image_style)
    style_prefix = "anime style, cel shading, clean manga line art"
    quality_prompt = "full body, feet visible, natural proportions"
    if image_style == "Realistic":
        if people_count == "2 people":
            if gender_mix in ("Girl and boy", "Woman and man"):
                subject = "adult woman and adult man age 19 or older"
            parts = [
                "RAW DSLR photo, photorealistic",
                f"{subject}, full body, both visible",
                appearance,
                outfit if "woman wearing" in outfit else f"wearing {outfit}",
                pose,
                background,
                "real skin pores, realistic face and hands, no extra people",
            ]
        else:
            parts = [
                "RAW DSLR photo, photorealistic",
                f"{subject}, full body, feet visible",
                ethnicity,
                appearance,
                f"wearing {outfit}",
                pose,
                background,
                "natural skin pores, realistic face and hands",
            ]
        return ", ".join(part for part in parts if part)

    if people_count == "2 people":
        parts = [
            "full body anime character art",
            subject,
            instruction,
            f"{ethnicity}",
            f"{appearance}",
            outfit if "woman wearing" in outfit else f"wearing exactly {outfit}",
            f"pose: {pose}",
            f"background: {background}",
            "characters inside this exact background",
            "anime style, clean line art, cel shading, both full body, no extra people",
            default_prompt,
        ]
    else:
        parts = [
            "full body anime character art",
            subject,
            instruction,
            f"{ethnicity}",
            f"{appearance}",
            f"wearing exactly {outfit}",
            f"pose: {pose}",
            f"background: {background}",
            "character inside this exact background",
            style_prefix,
            quality_prompt,
            default_prompt,
            style_option,
        ]
    return ", ".join(part for part in parts if part)


def targeted_negative_prompt(prompt, base_negative=NEGATIVE_IMAGE_PROMPT):
    text = (prompt or "").lower()
    extra = []

    hair_colors = {
        "pink hair": ["white hair", "silver hair", "black hair", "blonde hair", "red hair"],
        "red hair": ["white hair", "silver hair", "black hair", "blonde hair", "pink hair"],
        "silver hair": ["white hair", "black hair", "blonde hair", "red hair", "pink hair"],
        "blonde": ["white hair", "silver hair", "black hair", "red hair", "pink hair"],
        "black hair": ["white hair", "silver hair", "blonde hair", "red hair", "pink hair"],
        "brunette": ["white hair", "silver hair", "blonde hair", "red hair", "pink hair"],
    }
    for wanted, blocked in hair_colors.items():
        if wanted in text:
            extra.extend(blocked)
            break

    if "blue eyes" in text:
        extra.extend(["brown eyes", "amber eyes", "gold eyes", "green eyes", "violet eyes"])
    elif "gold eyes" in text:
        extra.extend(["blue eyes", "brown eyes", "green eyes", "violet eyes"])
    elif "violet eyes" in text:
        extra.extend(["blue eyes", "brown eyes", "gold eyes", "green eyes"])

    if any(word in text for word in ("swimsuit", "bikini")):
        extra.extend(["dress", "coat", "jacket", "school uniform", "long sleeves", "bodysuit"])
    elif "bodysuit" in text:
        extra.extend(["dress", "bikini", "swimsuit", "skirt"])
    elif "dress" in text:
        extra.extend(["bikini", "swimsuit", "bodysuit", "armor"])

    if "neon city rooftop" in text:
        extra.extend(["plain gray background", "studio background", "empty background", "abstract background", "beach", "forest", "indoor room"])
    elif "beach" in text:
        extra.extend(["gray background", "city background", "indoor room", "forest", "mountain", "gold frame", "mirror frame"])
    elif "kitchen" in text:
        extra.extend(["gray background", "beach", "rooftop", "forest", "outdoor landscape"])
    elif "forest" in text or "mountain" in text:
        extra.extend(
            [
                "beach",
                "ocean",
                "tropical water",
                "city",
                "rooftop",
                "kitchen",
                "bedroom",
                "palace",
                "gold frame",
                "mirror frame",
                "decorative frame",
                "indoor room",
                "plain studio background",
            ]
        )
    elif "angkor wat" in text or "temple" in text or "pagoda" in text:
        extra.extend(["beach", "ocean", "city street", "kitchen", "bedroom", "forest cabin", "plain gray background"])

    if "photorealistic" in text or "real human" in text:
        extra.extend(
            [
                "anime",
                "cartoon",
                "illustration",
                "drawing",
                "manga",
                "cel shading",
                "3d render",
                "plastic skin",
                "airbrushed skin",
                "heavy makeup",
                "mature face",
                "old face",
                "fashion model",
                "glamour model",
            ]
        )

    if "two main subjects only" in text:
        extra.extend(["crowd", "background people", "third person", "extra person", "many people"])
    elif "single main subject" in text:
        extra.extend(["crowd", "background people", "second person", "extra person", "many people"])

    if "one adult woman" in text or "only woman" in text or "only girl" in text:
        extra.extend(["man", "male", "boy", "masculine body", "male silhouette"])
    elif "one adult man" in text or "only man" in text or "only boy" in text:
        extra.extend(["woman", "female", "girl", "feminine body"])

    if "wearing exactly" in text:
        extra.extend(["wrong outfit", "different outfit", "changed clothing"])

    seen = set()
    cleaned_extra = []
    for item in extra:
        if item not in seen:
            seen.add(item)
            cleaned_extra.append(item)
    return ", ".join(filter(None, [base_negative, ", ".join(cleaned_extra)]))


def find_prompt_background(prompt):
    text = (prompt or "").lower()
    background_options = BACKGROUND_OPTIONS + REALISTIC_BACKGROUND_OPTIONS
    matches = [option for option in background_options if option.lower() in text]
    if matches:
        return max(matches, key=len)
    background_markers = ["background:", "setting:", "scene:"]
    for marker in background_markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            fragment = prompt[start:].split(",")[0].strip()
            if fragment:
                return fragment
    return ""


def find_prompt_option(prompt, options):
    text = (prompt or "").lower()
    matches = [option for option in options if option.lower() in text]
    if not matches:
        return ""
    return max(matches, key=len)


def find_prompt_subject(prompt):
    match = re.search(
        r"(one adult woman age 19 or older|one adult man age 19 or older|two adult women age 19 or older|two adult men age 19 or older|adult woman and adult man age 19 or older)",
        prompt,
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def find_prompt_outfit(prompt):
    text = prompt or ""
    match = re.search(
        r"wearing exactly\s+(.+?)(?:,\s*pose:|,\s*background:|,\s*anime style|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"wearing\s+(.+?)(?:,\s*pose:|,\s*background:|,\s*anime style|$)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def find_prompt_pose(prompt):
    text = prompt or ""
    match = re.search(r"pose:\s+(.+?)(?:,\s*background:|,\s*anime style|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return find_prompt_option(prompt, POSE_OPTIONS + REALISTIC_POSE_OPTIONS)


def prepare_image_prompt_for_model(prompt, image_style):
    prompt = (prompt or "").strip()
    if image_style != "Anime" or not prompt:
        return prompt

    background = find_prompt_background(prompt)
    subject = find_prompt_subject(prompt)
    ethnicity = find_prompt_option(prompt, ETHNICITY_OPTIONS + MALE_ETHNICITY_OPTIONS)
    appearance = find_prompt_option(prompt, APPEARANCE_OPTIONS + MALE_APPEARANCE_OPTIONS)
    outfit = find_prompt_outfit(prompt)
    pose = find_prompt_pose(prompt)

    if not any([background, subject, ethnicity, appearance, outfit, pose]):
        return prompt

    parts = [
        "full body anime character art",
        subject,
        ethnicity,
        appearance,
        f"wearing exactly {outfit}" if outfit else "",
        f"pose: {pose}" if pose else "",
        f"background: {background}" if background else "",
        "character inside this exact background" if background else "",
        "clean manga line art, cel shading, natural proportions, feet visible, high quality",
    ]
    return ", ".join(part for part in parts if part)



def preview_image_prompt(
    image_style,
    people_count,
    gender_mix,
    action,
    default_prompt,
    ethnicity_options_text,
    appearance_options_text,
    outfit_options_text,
    male_ethnicity_options_text,
    male_appearance_options_text,
    male_outfit_options_text,
    background_options_text,
    pose_options_text,
    style_options_text,
):
    character = get_default_image_character()
    if not character:
        return ""
    return build_random_image_prompt(
        character,
        image_style=image_style,
        people_count=people_count,
        gender_mix=gender_mix,
        action=action,
        default_prompt=default_prompt,
        ethnicity_options_text=ethnicity_options_text,
        appearance_options_text=appearance_options_text,
        outfit_options_text=outfit_options_text,
        male_ethnicity_options_text=male_ethnicity_options_text,
        male_appearance_options_text=male_appearance_options_text,
        male_outfit_options_text=male_outfit_options_text,
        background_options_text=background_options_text,
        pose_options_text=pose_options_text,
        style_options_text=style_options_text,
    )


def random_image_prompt_options(image_style, people_count, gender_mix, action, default_prompt):
    defaults = image_defaults_for_style(image_style)
    default_prompt = default_prompt or defaults["default_prompt"]
    if people_count == "Random":
        people_count = random.choice([option for option in PEOPLE_COUNT_OPTIONS if option != "Random"])
    gender_mix = normalize_gender_for_people(people_count, gender_mix)
    if gender_mix == "Random":
        gender_mix = random.choice([option for option in gender_choices_for_people(people_count) if option != "Random"])
    if action == "Random":
        action = random.choice([option for option in ACTION_OPTIONS if option != "Random"])
    ethnicity = random.choice(defaults["ethnicity_options"])
    appearance = random.choice(defaults["appearance_options"])
    outfit = random.choice(defaults["outfit_options"])
    male_ethnicity = random.choice(defaults["male_ethnicity_options"])
    male_appearance = random.choice(defaults["male_appearance_options"])
    male_outfit = random.choice(defaults["male_outfit_options"])
    background = random.choice(defaults["background_options"])
    pose = random.choice(defaults["pose_options"])
    style = random.choice(defaults["style_options"])
    preview = preview_image_prompt(
        image_style,
        people_count,
        gender_mix,
        action,
        default_prompt,
        ethnicity,
        appearance,
        outfit,
        male_ethnicity,
        male_appearance,
        male_outfit,
        background,
        pose,
        style,
    )
    show_female, show_male = gender_field_visibility(people_count, gender_mix)
    return (
        people_count,
        gender_mix,
        action,
        gr.update(value=ethnicity, visible=show_female),
        gr.update(value=appearance, visible=show_female),
        gr.update(value=outfit, visible=show_female),
        gr.update(value=male_ethnicity, visible=show_male),
        gr.update(value=male_appearance, visible=show_male),
        gr.update(value=male_outfit, visible=show_male),
        background,
        pose,
        style,
        preview,
    )


def load_image_style_prompt_options(image_style):
    defaults = image_defaults_for_style(image_style)
    default_prompt = defaults["default_prompt"]
    ethnicity = random.choice(defaults["ethnicity_options"])
    appearance = random.choice(defaults["appearance_options"])
    outfit = random.choice(defaults["outfit_options"])
    male_ethnicity = random.choice(defaults["male_ethnicity_options"])
    male_appearance = random.choice(defaults["male_appearance_options"])
    male_outfit = random.choice(defaults["male_outfit_options"])
    background = random.choice(defaults["background_options"])
    pose = random.choice(defaults["pose_options"])
    style = random.choice(defaults["style_options"])
    preview = preview_image_prompt(
        image_style,
        "1 person",
        "Only girl",
        "",
        default_prompt,
        ethnicity,
        appearance,
        outfit,
        male_ethnicity,
        male_appearance,
        male_outfit,
        background,
        pose,
        style,
    )
    return (
        default_prompt,
        "1 person",
        "Only girl",
        "",
        gr.update(value=ethnicity, visible=True),
        gr.update(value=appearance, visible=True),
        gr.update(value=outfit, visible=True),
        gr.update(value=male_ethnicity, visible=False),
        gr.update(value=male_appearance, visible=False),
        gr.update(value=male_outfit, visible=False),
        background,
        pose,
        style,
        preview,
    )


def image_style_from_video_style(video_style):
    if video_style == "Anime":
        return "Anime"
    return "Realistic"


def preview_video_prompt(
    video_style,
    people_count,
    gender_mix,
    action,
    default_prompt,
    ethnicity_options_text,
    appearance_options_text,
    outfit_options_text,
    male_ethnicity_options_text,
    male_appearance_options_text,
    male_outfit_options_text,
    background_options_text,
    pose_options_text,
    style_options_text,
):
    image_style = image_style_from_video_style(video_style)
    prompt = preview_image_prompt(
        image_style,
        people_count,
        gender_mix,
        action,
        default_prompt,
        ethnicity_options_text,
        appearance_options_text,
        outfit_options_text,
        male_ethnicity_options_text,
        male_appearance_options_text,
        male_outfit_options_text,
        background_options_text,
        pose_options_text,
        style_options_text,
    )
    if image_style == "Anime":
        return (
            f"{prompt}, anime video, cute design, smooth motion, stable face, clear line art"
        )
    return (
        f"{prompt}, real camera video, photorealistic human, natural skin pores, realistic face, "
        "centered medium full body shot, stable tripod camera, slow gentle motion, natural daylight, no close-up"
    )


def random_video_prompt_options(video_style, people_count, gender_mix, action, default_prompt):
    image_style = image_style_from_video_style(video_style)
    result = random_image_prompt_options(image_style, people_count, gender_mix, action, default_prompt)
    return result[:-1] + (
        preview_video_prompt(
            video_style,
            result[0],
            result[1],
            result[2],
            default_prompt or image_defaults_for_style(image_style)["default_prompt"],
            result[3]["value"] if isinstance(result[3], dict) and "value" in result[3] else result[3],
            result[4]["value"] if isinstance(result[4], dict) and "value" in result[4] else result[4],
            result[5]["value"] if isinstance(result[5], dict) and "value" in result[5] else result[5],
            result[6]["value"] if isinstance(result[6], dict) and "value" in result[6] else result[6],
            result[7]["value"] if isinstance(result[7], dict) and "value" in result[7] else result[7],
            result[8]["value"] if isinstance(result[8], dict) and "value" in result[8] else result[8],
            result[9],
            result[10],
            result[11],
        ),
    )


def load_video_style_prompt_options(video_style):
    image_style = image_style_from_video_style(video_style)
    result = load_image_style_prompt_options(image_style)
    return result[:-1] + (
        preview_video_prompt(
            video_style,
            result[1],
            result[2],
            result[3],
            result[0],
            result[4]["value"] if isinstance(result[4], dict) and "value" in result[4] else result[4],
            result[5]["value"] if isinstance(result[5], dict) and "value" in result[5] else result[5],
            result[6]["value"] if isinstance(result[6], dict) and "value" in result[6] else result[6],
            result[7]["value"] if isinstance(result[7], dict) and "value" in result[7] else result[7],
            result[8]["value"] if isinstance(result[8], dict) and "value" in result[8] else result[8],
            result[9]["value"] if isinstance(result[9], dict) and "value" in result[9] else result[9],
            result[10],
            result[11],
            result[12],
        ),
    )


def generate_image_status():
    return "🎨 Generating image..."


def get_image_pipe(dtype_override=None):
    global image_pipe, image_pipe_dtype
    if image_pipe is not None:
        return image_pipe

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Image model packages are not installed. Run: "
            "pip install diffusers torch transformers accelerate safetensors"
        ) from exc

    device = get_torch_device()
    dtype = dtype_override or get_image_torch_dtype(device)
    image_pipe_dtype = dtype
    variant = "fp16" if device == "cuda" else None
    image_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": not ALLOW_MODEL_DOWNLOAD,
        "safety_checker": None,
        "requires_safety_checker": False,
        "low_cpu_mem_usage": True,
    }
    if variant:
        image_kwargs["variant"] = variant

    try:
        image_pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL_ID, **image_kwargs)
    except Exception as exc:
        if "watermark ratio" in str(exc):
            raise RuntimeError(
                "PyTorch MPS rejected the current memory watermark settings. "
                "Restart the app so PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0 and "
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95 are applied before torch loads."
            ) from exc
        raise RuntimeError(
            "A real anime image model is needed for good Image tab results. "
            f"The configured image model '{IMAGE_MODEL_ID}' is not downloaded on this Mac yet. "
            "Run the app with ALLOW_MODEL_DOWNLOAD=1 one time, or download that model into the "
            "Hugging Face cache, then click Generate New Image again."
        ) from exc

    configure_pipeline_for_performance(image_pipe)
    if device != "cpu":
        image_pipe = image_pipe.to(device=device, dtype=dtype)
    else:
        image_pipe = image_pipe.to("cpu")

    return image_pipe


def reset_image_pipe():
    global image_pipe, image_pipe_dtype
    image_pipe = None
    image_pipe_dtype = None
    clear_torch_cache()


def generate_with_anime_image_model(
    character,
    request_text="",
    update_profile=True,
    image_style="Anime",
    default_prompt=DEFAULT_IMAGE_PROMPT,
    negative_prompt=NEGATIVE_IMAGE_PROMPT,
    ethnicity_options_text=None,
    appearance_options_text=None,
    outfit_options_text=None,
    background_options_text=None,
    pose_options_text=None,
    style_options_text=None,
    final_prompt=None,
):
    prompt = (final_prompt or "").strip()
    if not prompt:
        prompt = build_random_image_prompt(
            character,
            request_text=request_text,
            image_style=image_style,
            default_prompt=default_prompt,
            ethnicity_options_text=ethnicity_options_text,
            appearance_options_text=appearance_options_text,
            outfit_options_text=outfit_options_text,
            background_options_text=background_options_text,
            pose_options_text=pose_options_text,
            style_options_text=style_options_text,
        )
    model_prompt = prepare_image_prompt_for_model(prompt, image_style)
    generator_seed = random.randint(0, 2**31 - 1)
    guidance_scale = 6.5 if image_style == "Realistic" else max(IMAGE_GUIDANCE, 8.0)
    image_steps = max(18, min(IMAGE_STEPS, 24))
    image = None
    last_blank = False

    for retry_index in range(2):
        local_image_pipe = get_image_pipe(torch.float32 if retry_index else None)
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        with torch.inference_mode():
            image = local_image_pipe(
                prompt=model_prompt,
                negative_prompt=targeted_negative_prompt(model_prompt, negative_prompt or NEGATIVE_IMAGE_PROMPT),
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                num_inference_steps=image_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        last_blank = not image.getbbox()
        if not last_blank:
            break
        if get_torch_device() == "mps" and image_pipe_dtype == torch.float16:
            reset_image_pipe()
            continue
        break

    if last_blank:
        raise RuntimeError("Image model returned a blank image even after retrying with the safer dtype.")

    clear_torch_cache()
    GENERATED_DIR.mkdir(exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "image"
    path = (GENERATED_DIR / f"{filename}_anime_{int(time.time())}.png").resolve()
    image.save(path)

    if update_profile:
        CHARACTERS[character]["avatar"] = str(path)
        CHARACTERS[character]["image_model"] = IMAGE_MODEL_ID
        CHARACTERS[character]["image_prompt"] = prompt
        save_characters()
    return str(path), f"Generated {image_style.lower()} image with {IMAGE_MODEL_ID}: {prompt}"


def frame_to_image(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(frame).convert("RGB")


def ltx_frame_count(value):
    requested_frames = max(9, int(value))
    return ((requested_frames - 1) // FRAME_MULTIPLE) * FRAME_MULTIPLE + 1


def generate_with_ltx_image_model(character, request_text="", update_profile=True):
    prompt = build_random_image_prompt(character, request_text=request_text)
    local_ltx_pipe = load_pipeline()
    width = min(LTX_IMAGE_WIDTH, 512)
    height = min(LTX_IMAGE_HEIGHT, 512)
    frame_count = ltx_frame_count(LTX_IMAGE_FRAMES)

    with torch.inference_mode():
        result = local_ltx_pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE_PROMPT,
            width=width,
            height=height,
            num_frames=frame_count,
            num_inference_steps=max(4, min(LTX_IMAGE_STEPS, 12)),
            guidance_scale=min(LTX_IMAGE_GUIDANCE, 3.5),
            max_sequence_length=128,
        )

    generated_frames = result.frames[0]
    image = frame_to_image(generated_frames[len(generated_frames) // 2])
    clear_torch_cache()

    GENERATED_DIR.mkdir(exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "image"
    path = (GENERATED_DIR / f"{filename}_ltx_{int(time.time())}.png").resolve()
    image.save(path)

    if update_profile:
        CHARACTERS[character]["avatar"] = str(path)
        CHARACTERS[character]["image_model"] = "Local LTX model"
        CHARACTERS[character]["image_prompt"] = prompt
        save_characters()
    return str(path), f"Generated image with the same local LTX model: {prompt}"


def generate_character_image(
    character,
    update_profile=True,
    request_text="",
    image_style="Anime",
    default_prompt=DEFAULT_IMAGE_PROMPT,
    negative_prompt=NEGATIVE_IMAGE_PROMPT,
    ethnicity_options_text=None,
    appearance_options_text=None,
    outfit_options_text=None,
    background_options_text=None,
    pose_options_text=None,
    style_options_text=None,
    final_prompt=None,
):
    if not character or character not in CHARACTERS:
        character = get_default_image_character()
    if not character or character not in CHARACTERS:
        return None, "No image character is available."

    GENERATED_DIR.mkdir(exist_ok=True)
    try:
        return generate_with_anime_image_model(
            character,
            request_text=request_text,
            update_profile=update_profile,
            image_style=image_style,
            default_prompt=default_prompt,
            negative_prompt=negative_prompt,
            ethnicity_options_text=ethnicity_options_text,
            appearance_options_text=appearance_options_text,
            outfit_options_text=outfit_options_text,
            background_options_text=background_options_text,
            pose_options_text=pose_options_text,
            style_options_text=style_options_text,
            final_prompt=final_prompt,
        )
    except Exception as exc:
        traceback.print_exc()
        clear_torch_cache()
        return None, f"Anime image generation failed: {exc}"


def generate_profile_image_from_app(
    image_style,
    final_prompt,
):
    return generate_character_image(
        get_default_image_character(),
        update_profile=True,
        image_style=image_style,
        negative_prompt=NEGATIVE_IMAGE_PROMPT,
        final_prompt=final_prompt,
    )


def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def get_image_torch_dtype(device):
    if device == "cuda":
        return torch.float16
    if device == "mps" and IMAGE_MPS_DTYPE in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def is_m2max_profile():
    return PERFORMANCE_PROFILE in ("m2max", "m2_max", "apple_m2_max", "mps_fast")


def clear_torch_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError as exc:
            if "watermark ratio" not in str(exc):
                raise


def configure_pipeline_for_performance(loaded_pipe):
    if not LOW_MEMORY_MODE and is_m2max_profile():
        for method_name in ("disable_attention_slicing", "disable_vae_slicing", "disable_vae_tiling"):
            method = getattr(loaded_pipe, method_name, None)
            if method is not None:
                try:
                    method()
                except TypeError:
                    pass
        return

    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(loaded_pipe, method_name, None)
        if method is not None:
            try:
                method()
            except TypeError:
                try:
                    method("auto")
                except TypeError:
                    pass


def missing_model_files():
    required_options = [
        [[MODEL_DIR / "model_index.json"]],
        [[MODEL_DIR / "scheduler" / "scheduler_config.json"]],
        [[MODEL_DIR / "tokenizer" / "spiece.model"]],
        [[MODEL_DIR / "tokenizer" / "tokenizer_config.json"]],
        [[MODEL_DIR / "transformer" / "config.json"]],
        [
            [MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors"],
            [
                MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
                MODEL_DIR / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors",
                MODEL_DIR / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors",
            ],
        ],
        [[MODEL_DIR / "vae" / "config.json"]],
        [[MODEL_DIR / "vae" / "diffusion_pytorch_model.safetensors"]],
        [[MODEL_DIR / "text_encoder" / "config.json"]],
        [[MODEL_DIR / "text_encoder" / "model-00001-of-00004.safetensors"]],
        [[MODEL_DIR / "text_encoder" / "model-00002-of-00004.safetensors"]],
        [[MODEL_DIR / "text_encoder" / "model-00003-of-00004.safetensors"]],
        [[MODEL_DIR / "text_encoder" / "model-00004-of-00004.safetensors"]],
    ]

    missing = []
    for options in required_options:
        if not any(all(path.exists() for path in option) for option in options):
            missing.append(
                " or ".join(
                    " + ".join(str(path.relative_to(BASE_DIR)) for path in option)
                    for option in options
                )
            )
    return missing


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    missing = missing_model_files()
    if missing:
        missing_list = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "The local LTX model folder is incomplete.\n\n"
            f"Missing files:\n{missing_list}\n\n"
            "Download the missing files, then restart this app. Example:\n"
            'hf download Lightricks/LTX-Video --local-dir models --include "vae/*" '
            '"text_encoder/*" "tokenizer/*" "scheduler/*" "model_index.json"'
        )

    device = get_torch_device()
    dtype = get_torch_dtype(device)
    device_map = "cuda" if device == "cuda" else None

    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map

    loaded_pipe = LTXPipeline.from_pretrained(MODEL_DIR, **kwargs)
    configure_pipeline_for_performance(loaded_pipe)
    if not device_map and device != "cpu":
        loaded_pipe.to(device=device, dtype=dtype)

    pipe = loaded_pipe
    return pipe


def seconds_to_frames(seconds):
    target_frames = max(1, int(round(float(seconds) * VIDEO_FPS)))
    return max(9, ((target_frames - 1) // FRAME_MULTIPLE) * FRAME_MULTIPLE + 1)


def max_frames_for_speed_mode(speed_mode):
    if speed_mode == "Fastest":
        return FASTEST_FRAMES_PER_CLIP
    if speed_mode in ("Balanced", "Clear"):
        return BALANCED_FRAMES_PER_CLIP
    return MAX_FRAMES_PER_CLIP


def split_frame_count(total_frames, max_frames_per_clip=MAX_FRAMES_PER_CLIP):
    remaining = int(total_frames)
    frame_counts = []
    while remaining > 0:
        if remaining <= max_frames_per_clip:
            frame_counts.append(max(9, ((remaining - 1) // FRAME_MULTIPLE) * FRAME_MULTIPLE + 1))
            break

        frame_counts.append(max_frames_per_clip)
        remaining -= max_frames_per_clip

    return frame_counts


def default_auto_prompt(style):
    base_prompt = (
        "A brave warrior holding a glowing sword faces the king of dragons on a stormy mountain peak, "
        "dramatic camera movement, sparks in the air, epic fantasy action scene"
    )
    if style == "Anime":
        return f"{base_prompt}, anime battle scene, dynamic pose, expressive character art"
    if style == "Realistic":
        return f"{base_prompt}, realistic cinematic lighting, detailed armor, volumetric smoke"
    return base_prompt


def fast_preset():
    return 512, 512, 1, 8, 4.0


def balanced_preset():
    return 512, 576, 2, 12, 4.2


def clear_preset():
    return 512, 640, 2, 16, 4.4


def speed_mode_preset(speed_mode):
    if speed_mode == "Fastest":
        return fast_preset()
    if speed_mode == "Balanced":
        return balanced_preset()
    if speed_mode == "Clear":
        return clear_preset()
    return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())


def apply_speed_mode(speed_mode, width, height, duration_seconds, steps, guidance_scale):
    width = int(width)
    height = int(height)
    duration_seconds = float(duration_seconds)
    steps = int(steps)
    guidance_scale = float(guidance_scale)

    if speed_mode == "Fastest":
        return min(width, 512), min(height, 512), duration_seconds, max(8, min(steps, 8)), min(max(guidance_scale, 4.0), 4.0)
    if speed_mode == "Balanced":
        return min(width, 512), min(height, 608), duration_seconds, max(12, min(steps, 12)), min(max(guidance_scale, 4.2), 4.2)
    if speed_mode == "Clear":
        return min(width, 512), min(height, 640), duration_seconds, max(16, min(steps, 18)), min(max(guidance_scale, 4.4), 4.6)
    return width, height, duration_seconds, steps, guidance_scale


def apply_style(prompt, negative_prompt, style):
    prompt = prompt.strip()
    negative_prompt = (negative_prompt or "").strip()
    quality_prompt = (
        "realistic face, natural skin pores, realistic hands, smooth gentle motion, stable tripod camera, clear video"
    )
    negative_prompt = ", ".join(filter(None, [negative_prompt, VIDEO_NEGATIVE_PROMPT]))

    if style == "Anime":
        prompt = f"{prompt}, anime style, expressive 2D animation, clean line art, vibrant colors, {quality_prompt}"
        negative_prompt = ", ".join(filter(None, [negative_prompt, "photorealistic, live action, 3d render"]))
    elif style == "Realistic":
        prompt = (
            f"{prompt}, real camera footage, photorealistic human, natural daylight, "
            f"centered medium full body shot, no close-up, no cutaway, {quality_prompt}"
        )
        negative_prompt = ", ".join(filter(None, [negative_prompt, "illustration, painting, airbrushed skin, plastic skin, doll face, unreal face"]))
    else:
        prompt = f"{prompt}, {quality_prompt}"

    return prompt, negative_prompt


def stitch_videos(segment_paths, output_path):
    with imageio.get_writer(str(output_path), fps=VIDEO_FPS, codec="libx264", macro_block_size=None) as writer:
        for segment_path in segment_paths:
            reader = imageio.get_reader(str(segment_path))
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()


def next_output_path():
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_number = 1
    while True:
        output_path = OUTPUT_DIR / f"ltx_output_{output_number}.mp4"
        if not output_path.exists():
            return output_path, output_number
        output_number += 1


def generate_video(
    prompt,
    negative_prompt,
    style,
    speed_mode,
    width,
    height,
    duration_seconds,
    steps,
    guidance_scale,
    seed,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    try:
        width, height, duration_seconds, steps, guidance_scale = apply_speed_mode(
            speed_mode, width, height, duration_seconds, steps, guidance_scale
        )
        requested_seconds = float(duration_seconds)
        duration_seconds = min(requested_seconds, MAX_SINGLE_PASS_SECONDS)
        styled_prompt, styled_negative_prompt = apply_style(prompt, negative_prompt, style)
        frame_count = seconds_to_frames(duration_seconds)

        generator = None
        if seed is not None and int(seed) >= 0:
            device = "cuda" if get_torch_device() == "cuda" else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        ltx = load_pipeline()
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path, output_number = next_output_path()

        with torch.inference_mode():
            result = ltx(
                prompt=styled_prompt,
                negative_prompt=styled_negative_prompt,
                width=width,
                height=height,
                num_frames=frame_count,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=128,
            )

        export_to_video(result.frames[0], str(output_path), fps=VIDEO_FPS)
        del result
        clear_torch_cache()

        actual_seconds = frame_count / VIDEO_FPS
        device = get_torch_device().upper()
        return (
            str(output_path),
            f"Video generated successfully. Length: about {actual_seconds:.1f}s. "
            f"Used one single-pass {speed_mode} render at {width}x{height}, {steps} steps on {device}."
            + (
                f" Requested {requested_seconds:.1f}s was capped to {MAX_SINGLE_PASS_SECONDS}s to avoid MPS memory crash."
                if requested_seconds > MAX_SINGLE_PASS_SECONDS
                else ""
            ),
        )
    except Exception as exc:
        traceback.print_exc()
        clear_torch_cache()
        return None, f"Error: {exc}"


def random_field_button(visible=True):
    return gr.Button(
        "Random",
        variant="secondary",
        visible=visible,
        min_width=76,
        elem_classes=["field-random-button"],
    )


with gr.Blocks(title="Local LTX Video Generator", css=APP_CSS) as demo:
    gr.Markdown("# Local LTX Video Generator")
    with gr.Tabs():
        with gr.Tab("Generate Image"):
            initial_ethnicity = random.choice(ETHNICITY_OPTIONS)
            initial_appearance = random.choice(APPEARANCE_OPTIONS)
            initial_outfit = random.choice(OUTFIT_OPTIONS)
            initial_male_ethnicity = random.choice(MALE_ETHNICITY_OPTIONS)
            initial_male_appearance = random.choice(MALE_APPEARANCE_OPTIONS)
            initial_male_outfit = random.choice(MALE_OUTFIT_OPTIONS)
            initial_background = random.choice(BACKGROUND_OPTIONS)
            initial_pose = random.choice(POSE_OPTIONS)
            initial_style_option = random.choice(STYLE_OPTIONS)
            image_style = gr.Radio(
                ["Anime", "Realistic"],
                value="Anime",
                label="Image style",
            )
            with gr.Row():
                image_people_count = gr.Radio(
                    PEOPLE_COUNT_OPTIONS,
                    value="1 person",
                    label="People",
                )
                image_gender_mix = gr.Radio(
                    SINGLE_GENDER_OPTIONS,
                    value="Only girl",
                    label="Gender",
                )
            image_action = gr.Textbox(
                value="",
                label="Action",
                lines=1,
            )
            with gr.Row():
                image_default_prompt = gr.Textbox(
                    label="DEFAULT_IMAGE_PROMPT",
                    lines=3,
                    value=DEFAULT_IMAGE_PROMPT,
                    scale=8,
                )
                image_default_random = random_field_button()
            with gr.Accordion("Prompt option lists", open=True):
                image_random_options = gr.Button("Load random prompt options", variant="secondary")
                with gr.Row():
                    image_ethnicity_options = gr.Textbox(
                        label="ETHNICITY_OPTIONS",
                        lines=1,
                        value=initial_ethnicity,
                        scale=8,
                    )
                    image_ethnicity_random = random_field_button()
                with gr.Row():
                    image_appearance_options = gr.Textbox(
                        label="APPEARANCE_OPTIONS",
                        lines=1,
                        value=initial_appearance,
                        scale=8,
                    )
                    image_appearance_random = random_field_button()
                with gr.Row():
                    image_outfit_options = gr.Textbox(
                        label="OUTFIT_OPTIONS",
                        lines=1,
                        value=initial_outfit,
                        scale=8,
                    )
                    image_outfit_random = random_field_button()
                with gr.Row():
                    image_male_ethnicity_options = gr.Textbox(
                        label="MAN_ETHNICITY_OPTIONS",
                        lines=1,
                        value=initial_male_ethnicity,
                        visible=False,
                        scale=8,
                    )
                    image_male_ethnicity_random = random_field_button(visible=False)
                with gr.Row():
                    image_male_appearance_options = gr.Textbox(
                        label="MAN_APPEARANCE_OPTIONS",
                        lines=1,
                        value=initial_male_appearance,
                        visible=False,
                        scale=8,
                    )
                    image_male_appearance_random = random_field_button(visible=False)
                with gr.Row():
                    image_male_outfit_options = gr.Textbox(
                        label="MAN_OUTFIT_OPTIONS",
                        lines=1,
                        value=initial_male_outfit,
                        visible=False,
                        scale=8,
                    )
                    image_male_outfit_random = random_field_button(visible=False)
                with gr.Row():
                    image_background_options = gr.Textbox(
                        label="BACKGROUND_OPTIONS",
                        lines=1,
                        value=initial_background,
                        scale=8,
                    )
                    image_background_random = random_field_button()
                with gr.Row():
                    image_pose_options = gr.Textbox(
                        label="POSE_OPTIONS",
                        lines=1,
                        value=initial_pose,
                        scale=8,
                    )
                    image_pose_random = random_field_button()
                with gr.Row():
                    image_style_options = gr.Textbox(
                        label="STYLE_OPTIONS",
                        lines=1,
                        value=initial_style_option,
                        scale=8,
                    )
                    image_style_random = random_field_button()
            image_prompt_preview = gr.Textbox(
                label="Current prompt preview",
                lines=4,
                value=preview_image_prompt(
                    "Anime",
                    "1 person",
                    "Only girl",
                    "",
                    DEFAULT_IMAGE_PROMPT,
                    initial_ethnicity,
                    initial_appearance,
                    initial_outfit,
                    initial_male_ethnicity,
                    initial_male_appearance,
                    initial_male_outfit,
                    initial_background,
                    initial_pose,
                    initial_style_option,
                ),
                interactive=True,
            )
            image_output = gr.Image(height=520, label="Image")
            image_status = gr.Markdown("")
            image_generate = gr.Button("🎨 Generate New Image", variant="secondary")

            image_prompt_inputs = [
                image_style,
                image_people_count,
                image_gender_mix,
                image_action,
                image_default_prompt,
                image_ethnicity_options,
                image_appearance_options,
                image_outfit_options,
                image_male_ethnicity_options,
                image_male_appearance_options,
                image_male_outfit_options,
                image_background_options,
                image_pose_options,
                image_style_options,
            ]
            for image_prompt_input in image_prompt_inputs:
                image_prompt_input.change(
                    preview_image_prompt,
                    inputs=image_prompt_inputs,
                    outputs=image_prompt_preview,
                )

            image_prompt_random_buttons = [
                image_ethnicity_random,
                image_appearance_random,
                image_outfit_random,
                image_male_ethnicity_random,
                image_male_appearance_random,
                image_male_outfit_random,
            ]
            image_field_randomizers = [
                (image_default_random, "default_prompt", image_default_prompt),
                (image_ethnicity_random, "ethnicity_options", image_ethnicity_options),
                (image_appearance_random, "appearance_options", image_appearance_options),
                (image_outfit_random, "outfit_options", image_outfit_options),
                (image_male_ethnicity_random, "male_ethnicity_options", image_male_ethnicity_options),
                (image_male_appearance_random, "male_appearance_options", image_male_appearance_options),
                (image_male_outfit_random, "male_outfit_options", image_male_outfit_options),
                (image_background_random, "background_options", image_background_options),
                (image_pose_random, "pose_options", image_pose_options),
                (image_style_random, "style_options", image_style_options),
            ]
            for random_button, option_key, target_field in image_field_randomizers:
                random_button.click(
                    lambda selected_style, key=option_key: random_single_prompt_option(selected_style, key),
                    inputs=[image_style],
                    outputs=target_field,
                ).then(
                    preview_image_prompt,
                    inputs=image_prompt_inputs,
                    outputs=image_prompt_preview,
                )

            image_people_count.change(
                update_gender_choices,
                inputs=[image_people_count, image_gender_mix],
                outputs=[
                    image_gender_mix,
                    image_ethnicity_options,
                    image_appearance_options,
                    image_outfit_options,
                    image_male_ethnicity_options,
                    image_male_appearance_options,
                    image_male_outfit_options,
                ],
            )
            image_people_count.change(
                update_prompt_option_button_visibility,
                inputs=[image_people_count, image_gender_mix],
                outputs=image_prompt_random_buttons,
            )

            image_gender_mix.change(
                update_prompt_option_visibility,
                inputs=[image_people_count, image_gender_mix],
                outputs=[
                    image_ethnicity_options,
                    image_appearance_options,
                    image_outfit_options,
                    image_male_ethnicity_options,
                    image_male_appearance_options,
                    image_male_outfit_options,
                ],
            )
            image_gender_mix.change(
                update_prompt_option_button_visibility,
                inputs=[image_people_count, image_gender_mix],
                outputs=image_prompt_random_buttons,
            )

            image_style.change(
                load_image_style_prompt_options,
                inputs=[image_style],
                outputs=[
                    image_default_prompt,
                    image_people_count,
                    image_gender_mix,
                    image_action,
                    image_ethnicity_options,
                    image_appearance_options,
                    image_outfit_options,
                    image_male_ethnicity_options,
                    image_male_appearance_options,
                    image_male_outfit_options,
                    image_background_options,
                    image_pose_options,
                    image_style_options,
                    image_prompt_preview,
                ],
            ).then(
                update_prompt_option_button_visibility,
                inputs=[image_people_count, image_gender_mix],
                outputs=image_prompt_random_buttons,
            )

            image_random_options.click(
                random_image_prompt_options,
                inputs=[
                    image_style,
                    image_people_count,
                    image_gender_mix,
                    image_action,
                    image_default_prompt,
                ],
                outputs=[
                    image_people_count,
                    image_gender_mix,
                    image_action,
                    image_ethnicity_options,
                    image_appearance_options,
                    image_outfit_options,
                    image_male_ethnicity_options,
                    image_male_appearance_options,
                    image_male_outfit_options,
                    image_background_options,
                    image_pose_options,
                    image_style_options,
                    image_prompt_preview,
                ],
            ).then(
                update_prompt_option_button_visibility,
                inputs=[image_people_count, image_gender_mix],
                outputs=image_prompt_random_buttons,
            )

            image_generate.click(generate_image_status, outputs=image_status).then(
                generate_profile_image_from_app,
                inputs=[
                    image_style,
                    image_prompt_preview,
                ],
                outputs=[image_output, image_status],
            )

        with gr.Tab("Generate Video"):
            with gr.Row():
                with gr.Column():
                    style = gr.Radio(
                        ["Normal", "Anime", "Realistic"],
                        value="Realistic",
                        label="Visual style",
                    )
                    video_initial_ethnicity = random.choice(REALISTIC_ETHNICITY_OPTIONS)
                    video_initial_appearance = random.choice(REALISTIC_APPEARANCE_OPTIONS)
                    video_initial_outfit = random.choice(REALISTIC_OUTFIT_OPTIONS)
                    video_initial_male_ethnicity = random.choice(REALISTIC_MALE_ETHNICITY_OPTIONS)
                    video_initial_male_appearance = random.choice(REALISTIC_MALE_APPEARANCE_OPTIONS)
                    video_initial_male_outfit = random.choice(REALISTIC_MALE_OUTFIT_OPTIONS)
                    video_initial_background = random.choice(REALISTIC_BACKGROUND_OPTIONS)
                    video_initial_pose = random.choice(REALISTIC_POSE_OPTIONS)
                    video_initial_style_option = random.choice(REALISTIC_STYLE_OPTIONS)
                    with gr.Row():
                        video_people_count = gr.Radio(
                            PEOPLE_COUNT_OPTIONS,
                            value="1 person",
                            label="People",
                        )
                        video_gender_mix = gr.Radio(
                            SINGLE_GENDER_OPTIONS,
                            value="Only girl",
                            label="Gender",
                        )
                    video_action = gr.Textbox(
                        value="",
                        label="Action",
                        lines=1,
                    )
                    with gr.Row():
                        video_default_prompt = gr.Textbox(
                            label="VIDEO_DEFAULT_PROMPT",
                            lines=3,
                            value=REALISTIC_IMAGE_PROMPT,
                            scale=8,
                        )
                        video_default_random = random_field_button()
                    with gr.Accordion("Video prompt option lists", open=False):
                        video_random_options = gr.Button("Load random video prompt options", variant="secondary")
                        with gr.Row():
                            video_ethnicity_options = gr.Textbox(
                                label="ETHNICITY_OPTIONS",
                                lines=1,
                                value=video_initial_ethnicity,
                                scale=8,
                            )
                            video_ethnicity_random = random_field_button()
                        with gr.Row():
                            video_appearance_options = gr.Textbox(
                                label="APPEARANCE_OPTIONS",
                                lines=1,
                                value=video_initial_appearance,
                                scale=8,
                            )
                            video_appearance_random = random_field_button()
                        with gr.Row():
                            video_outfit_options = gr.Textbox(
                                label="OUTFIT_OPTIONS",
                                lines=1,
                                value=video_initial_outfit,
                                scale=8,
                            )
                            video_outfit_random = random_field_button()
                        with gr.Row():
                            video_male_ethnicity_options = gr.Textbox(
                                label="MAN_ETHNICITY_OPTIONS",
                                lines=1,
                                value=video_initial_male_ethnicity,
                                visible=False,
                                scale=8,
                            )
                            video_male_ethnicity_random = random_field_button(visible=False)
                        with gr.Row():
                            video_male_appearance_options = gr.Textbox(
                                label="MAN_APPEARANCE_OPTIONS",
                                lines=1,
                                value=video_initial_male_appearance,
                                visible=False,
                                scale=8,
                            )
                            video_male_appearance_random = random_field_button(visible=False)
                        with gr.Row():
                            video_male_outfit_options = gr.Textbox(
                                label="MAN_OUTFIT_OPTIONS",
                                lines=1,
                                value=video_initial_male_outfit,
                                visible=False,
                                scale=8,
                            )
                            video_male_outfit_random = random_field_button(visible=False)
                        with gr.Row():
                            video_background_options = gr.Textbox(
                                label="BACKGROUND_OPTIONS",
                                lines=1,
                                value=video_initial_background,
                                scale=8,
                            )
                            video_background_random = random_field_button()
                        with gr.Row():
                            video_pose_options = gr.Textbox(
                                label="POSE_OPTIONS",
                                lines=1,
                                value=video_initial_pose,
                                scale=8,
                            )
                            video_pose_random = random_field_button()
                        with gr.Row():
                            video_style_options = gr.Textbox(
                                label="STYLE_OPTIONS",
                                lines=1,
                                value=video_initial_style_option,
                                scale=8,
                            )
                            video_style_random = random_field_button()
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        value=preview_video_prompt(
                            "Realistic",
                            "1 person",
                            "Only girl",
                            "",
                            REALISTIC_IMAGE_PROMPT,
                            video_initial_ethnicity,
                            video_initial_appearance,
                            video_initial_outfit,
                            video_initial_male_ethnicity,
                            video_initial_male_appearance,
                            video_initial_male_outfit,
                            video_initial_background,
                            video_initial_pose,
                            video_initial_style_option,
                        ),
                    )
                    auto_prompt = gr.Button("Auto generate prompt")
                    speed_mode = gr.Radio(
                        ["Fastest", "Balanced", "Clear", "Manual"],
                        value="Clear",
                        label="Speed mode",
                    )
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=3, value=VIDEO_NEGATIVE_PROMPT)
                    with gr.Row():
                        width = gr.Slider(256, 1024, value=512, step=32, label="Width")
                        height = gr.Slider(256, 1024, value=640, step=32, label="Height")
                    with gr.Row():
                        duration_seconds = gr.Slider(
                            1,
                            MAX_SINGLE_PASS_SECONDS,
                            value=1,
                            step=1,
                            label="Video length (seconds)",
                        )
                        steps = gr.Slider(1, 50, value=16, step=1, label="Quality steps")
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 10.0, value=4.4, step=0.1, label="Guidance")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                    generate = gr.Button("Generate", variant="primary")
                with gr.Column():
                    video = gr.Video(label="Output")
                    status = gr.Textbox(label="Status", interactive=False)

            video_prompt_inputs = [
                style,
                video_people_count,
                video_gender_mix,
                video_action,
                video_default_prompt,
                video_ethnicity_options,
                video_appearance_options,
                video_outfit_options,
                video_male_ethnicity_options,
                video_male_appearance_options,
                video_male_outfit_options,
                video_background_options,
                video_pose_options,
                video_style_options,
            ]
            for video_prompt_input in video_prompt_inputs:
                video_prompt_input.change(
                    preview_video_prompt,
                    inputs=video_prompt_inputs,
                    outputs=prompt,
                )

            video_prompt_random_buttons = [
                video_ethnicity_random,
                video_appearance_random,
                video_outfit_random,
                video_male_ethnicity_random,
                video_male_appearance_random,
                video_male_outfit_random,
            ]
            video_field_randomizers = [
                (video_default_random, "default_prompt", video_default_prompt),
                (video_ethnicity_random, "ethnicity_options", video_ethnicity_options),
                (video_appearance_random, "appearance_options", video_appearance_options),
                (video_outfit_random, "outfit_options", video_outfit_options),
                (video_male_ethnicity_random, "male_ethnicity_options", video_male_ethnicity_options),
                (video_male_appearance_random, "male_appearance_options", video_male_appearance_options),
                (video_male_outfit_random, "male_outfit_options", video_male_outfit_options),
                (video_background_random, "background_options", video_background_options),
                (video_pose_random, "pose_options", video_pose_options),
                (video_style_random, "style_options", video_style_options),
            ]
            for random_button, option_key, target_field in video_field_randomizers:
                random_button.click(
                    lambda selected_style, key=option_key: random_single_video_prompt_option(selected_style, key),
                    inputs=[style],
                    outputs=target_field,
                ).then(
                    preview_video_prompt,
                    inputs=video_prompt_inputs,
                    outputs=prompt,
                )

            video_people_count.change(
                update_gender_choices,
                inputs=[video_people_count, video_gender_mix],
                outputs=[
                    video_gender_mix,
                    video_ethnicity_options,
                    video_appearance_options,
                    video_outfit_options,
                    video_male_ethnicity_options,
                    video_male_appearance_options,
                    video_male_outfit_options,
                ],
            )
            video_people_count.change(
                update_prompt_option_button_visibility,
                inputs=[video_people_count, video_gender_mix],
                outputs=video_prompt_random_buttons,
            )

            video_gender_mix.change(
                update_prompt_option_visibility,
                inputs=[video_people_count, video_gender_mix],
                outputs=[
                    video_ethnicity_options,
                    video_appearance_options,
                    video_outfit_options,
                    video_male_ethnicity_options,
                    video_male_appearance_options,
                    video_male_outfit_options,
                ],
            )
            video_gender_mix.change(
                update_prompt_option_button_visibility,
                inputs=[video_people_count, video_gender_mix],
                outputs=video_prompt_random_buttons,
            )

            style.change(
                load_video_style_prompt_options,
                inputs=[style],
                outputs=[
                    video_default_prompt,
                    video_people_count,
                    video_gender_mix,
                    video_action,
                    video_ethnicity_options,
                    video_appearance_options,
                    video_outfit_options,
                    video_male_ethnicity_options,
                    video_male_appearance_options,
                    video_male_outfit_options,
                    video_background_options,
                    video_pose_options,
                    video_style_options,
                    prompt,
                ],
            ).then(
                update_prompt_option_button_visibility,
                inputs=[video_people_count, video_gender_mix],
                outputs=video_prompt_random_buttons,
            )

            video_random_options.click(
                random_video_prompt_options,
                inputs=[
                    style,
                    video_people_count,
                    video_gender_mix,
                    video_action,
                    video_default_prompt,
                ],
                outputs=[
                    video_people_count,
                    video_gender_mix,
                    video_action,
                    video_ethnicity_options,
                    video_appearance_options,
                    video_outfit_options,
                    video_male_ethnicity_options,
                    video_male_appearance_options,
                    video_male_outfit_options,
                    video_background_options,
                    video_pose_options,
                    video_style_options,
                    prompt,
                ],
            ).then(
                update_prompt_option_button_visibility,
                inputs=[video_people_count, video_gender_mix],
                outputs=video_prompt_random_buttons,
            )

            auto_prompt.click(default_auto_prompt, inputs=[style], outputs=[prompt])
            speed_mode.change(
                speed_mode_preset,
                inputs=[speed_mode],
                outputs=[width, height, duration_seconds, steps, guidance_scale],
            )

            generate.click(
                generate_video,
                inputs=[
                    prompt,
                    negative_prompt,
                    style,
                    speed_mode,
                    width,
                    height,
                    duration_seconds,
                    steps,
                    guidance_scale,
                    seed,
                ],
                outputs=[video, status],
            )


if __name__ == "__main__":
    print(f"Using local model directory: {MODEL_DIR}")
    print(
        f"Performance profile: {PERFORMANCE_PROFILE} "
        f"(LOW_MEMORY_MODE={LOW_MEMORY_MODE}, IMAGE_MPS_DTYPE={IMAGE_MPS_DTYPE}, "
        f"MPS_HIGH_WATERMARK={os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}, "
        f"MPS_LOW_WATERMARK={os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO')})"
    )
    print("Open the app at the local URL printed by Gradio below.")
    demo.launch(server_name="127.0.0.1", server_port=get_server_port())
