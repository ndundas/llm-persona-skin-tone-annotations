import os
import httpx
_original_httpx_init = httpx.Client.__init__
def _patched_httpx_init(self, *args, **kwargs):
    # remove 'proxies' if present, so downstream clients never see it:
    kwargs.pop("proxies", None)
    return _original_httpx_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_init
import csv
import base64
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import time

from dotenv import load_dotenv
from PIL import Image
from anthropic import Anthropic
import aisuite as ai
from google import genai 
from google.genai import types
from anthropic._base_client import SyncHttpxClientWrapper

# ----------------------------------------
# CONFIG & LOGGING
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class APIKeys:
    openai: str
    anthropic: str
    groq: str
    xai: str
    genai: str  

def load_config(env_path: str = "keys.env") -> APIKeys:
    load_dotenv(dotenv_path=env_path)
    return APIKeys(
        openai   = os.getenv("OPENAI_API_KEY", ""),
        anthropic= os.getenv("ANTHROPIC_API_KEY", ""),
        groq     = os.getenv("GROQ_API_KEY", ""),
        xai      = os.getenv("TWITTER_GROK_API_KEY", ""),
        genai    = os.getenv("GENAI_API_KEY", ""),  
    )

def init_clients(keys: APIKeys):
    # 1) Unset any proxy env vars early
    for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
        os.environ.pop(v, None)

    # 2) Set your API keys in env (some SDKs read them automatically)
    os.environ["OPENAI_API_KEY"]    = keys.openai
    os.environ["ANTHROPIC_API_KEY"] = keys.anthropic
    os.environ["GROQ_API_KEY"]      = keys.groq
    os.environ["XAI_API_KEY"]       = keys.xai

    # 3) Instantiate each client *without* extra kwargs
    openai_client    = ai.Client()
    anthropic_client = Anthropic(api_key=keys.anthropic)
    xai_client       = Anthropic(api_key=keys.xai, base_url="https://api.x.ai")
    gemini_client    = genai.Client(api_key=keys.genai)

    return openai_client, anthropic_client, xai_client, gemini_client

# ----------------------------------------
# MODELS & PROMPTS
# ----------------------------------------
GPT_MODEL     = "openai:gpt-4o-2024-11-20"
CLAUDE_MODEL  = "claude-3-5-haiku-20241022"
LLAMA_MODEL   = "groq:meta-llama/llama-4-scout-17b-16e-instruct"
GROK_MODEL    = "grok-2-vision-1212"
GEMINI_MODEL  = "gemini-2.0-flash" 

CORE_TASK = "You are an annotator in a study evaluating skin tone."

MONK_PROMPT_noImage = (
    "Using the Monk scale (1–10), please provide the following for Image 1 (Subject):\n"
    "• One Monk rating (1–10)\n"
    "• One Likert confidence (1–7, where 1 = 'Not at all confident' and 7 = 'Completely confident')\n"
    "• A rationale for your response in 40 words or fewer.\n\n"   
    "Please follow this strict format:\n\n"
    "Monk: [1–10]\n"
    "Confidence: [1–7]\n"
    "Rationale: [up to 40 words]"
)

MONK_PROMPT_noName = (
    "Using the scale (1–10 from left to right) provided in Image 1 (Scale), please provide the following for Image 2 (Subject):\n"
    "• One rating using Image 1 (Scale) (1–10)\n"
    "• One Likert confidence (1–7, where 1 = 'Not at all confident' and 7 = 'Completely confident')\n"
    "• A rationale for your response in 40 words or fewer.\n\n"
    "Please follow this strict format:\n\n"
    "Rating: [1–10]\n"
    "Confidence: [1–7]\n"
    "Rationale: [up to 40 words]"
)

MONK_PROMPT_withImage = (
    "Using the Monk scale (1–10 from left to right) provided in Image 1 (Monk Scale), please provide the following for Image 2 (Subject):\n"
    "• One Monk rating using Image 1 (Monk Scale) (1–10)\n"
    "• One Likert confidence (1–7, where 1 = 'Not at all confident' and 7 = 'Completely confident')\n"
    "• A rationale for your response in 40 words or fewer.\n\n"
    "Please follow this strict format:\n\n"
    "Monk Rating: [1–10]\n"
    "Confidence: [1–7]\n"
    "Rationale: [up to 40 words]"
)

# Only Monk variation remains
PROMPT_VARIATIONS: Dict[str, str] = {
    # "No Monk Scale": MONK_PROMPT_noImage,
    # "No Monk Name": MONK_PROMPT_noName,
    "With Monk Scale": MONK_PROMPT_withImage,
}

# Cultural and other context conditions
CONTEXT_CONDITIONS = [
    ("baseline", ""),
    ("ignore_world", "As an annotator in this study, ignore all world knowledge of structural biases, race as a social concept, or any other social factors that might bias your annotation."),
    ("embrace_world", "As an annotator in this study, consider all available world knowledge and perspectives as you annotate the image, including race, cultural background, socioeconomic factors, or any relevant sociocultural context that could influence your annotation."),
    ("cultural_white", "As an annotator in this study, you are White. "),
    ("cultural_black", "As an annotator in this study, you are Black. "),
    ("cultural_asain", "As an annotator in this study, you are Asian. "),
    ("cultural_hispanic", "As an annotator in this study, you are Hispanic. "),
    ("cultural_indigenous", "As an annotator in this study, you are Indigenous. "),
    ("cultural_africanamerican", "As an annotator in this study, you are African American. "),
    ("cultural_colorism", "As an annotator in this study, you are from a country that experiences high levels of colorism. "),
    ("cultural_lighter", "As an annotator in this study, you are from a country that is predominantly lighter-skin tone. "),
    ("cultural_darker", "As an annotator in this study, you are from a country that is predominantly darker-skin tone. ")
]

# ----------------------------------------
# HANDLER DISPATCH
# ----------------------------------------
def _decode_image_for_gemini(img_b64: str) -> Image.Image:
    raw = base64.b64decode(img_b64)
    return Image.open(BytesIO(raw))

ACTIVE_MODELS = {
    "GPT": (
        GPT_MODEL,
        lambda cli, model, sys, prm, img, scale: cli.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":[
                    {"type":"text", "text": prm},
                    # include scale image if provided
                    *(
                        [{"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{scale}"}}]
                        if scale else []
                    ),
                    {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{img}"}}
                ]}
            ],
            max_tokens=100,
            temperature=0.7
        ).choices[0].message.content
    ),

    "Claude": (
        CLAUDE_MODEL,
        lambda cli, model, sys, prm, img, scale: " ".join(
            block.text for block in cli.messages.create(
                model=model,
                system=sys,
                messages=[
                    {"role":"user","content":
                        # prepend scale if present
                        (f"{prm}\n\nScale (jpeg/base64): {scale}\n\n"
                         if scale else prm + "\n\n")
                        + f"Image (jpeg/base64): {img}"
                    }
                ],
                max_tokens=100,
                temperature=1.0
            ).content if block.type=="text"
        )
    ),

    "Llama": (
        LLAMA_MODEL,
        lambda cli, model, sys, prm, img, scale: cli.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":
                    prm
                    + ("\n\nScale image (base64): " + scale if scale else "")
                    + f"\n\nImage (base64): {img}"
                }
            ],
            temperature=0.9
        ).choices[0].message.content
    ),

    "Grok": (
        GROK_MODEL,
        # just re-use the Claude pattern on the X.AI client
        lambda cli, model, sys, prm, img, scale: ACTIVE_MODELS["Claude"][1](
            cli, model, sys, prm, img, scale
        )
    ),

    "Gemini": (
        GEMINI_MODEL,
        lambda cli, model, sys, prm, img, scale: cli.models.generate_content(
            model=model,
            contents=[
                f"{sys}\n\n{prm}", # system prompt and user prompt together 
                # decode & include scale PIL image if provided
                *([_decode_image_for_gemini(scale)] if scale else []),
                _decode_image_for_gemini(img)
            ],
            config=types.GenerateContentConfig(
                temperature=1.0), 
        ).candidates[0].content.parts[0].text
    ),
}

# ----------------------------------------
# IO HELPERS & PROCESS
# ----------------------------------------
def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def encode_and_resize(path: Path, max_dim: int = 128) -> str:
    """
    Load image from `path`, make a copy, downsample the copy to max_dim×max_dim,
    convert to RGB (dropping any alpha), JPEG‐encode & base64‐encode it.
    """
    # 1) Open & copy
    img = Image.open(path)
    img_copy = img.copy()

    # 2) Downsample
    img_copy.thumbnail((max_dim, max_dim))

    # 3) Drop alpha channel if present
    if img_copy.mode in ("RGBA", "LA") or (img_copy.mode == "P" and "transparency" in img_copy.info):
        img_copy = img_copy.convert("RGB")

    # 4) JPEG‐encode in memory
    buffer = BytesIO()
    img_copy.save(buffer, format="JPEG", quality=95)

    # 5) Base64 output
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def list_images(folder: Path) -> List[Tuple[str,Path]]:
    return [(p.name, p) for p in folder.iterdir() if p.suffix.lower() in (".jpg",".png",".jpeg")]

def flatten_responses(resps: Dict) -> Dict[str,str]:
    flat = {}
    for model, var_dict in resps.items():
        for var, ctxs in var_dict.items():
            for ctx, out in ctxs.items():
                flat[f"{model}_{var}_{ctx}"] = out
    return flat

def process_image(
    name: str,
    path: Path,
    prompt: str,
    clients: Tuple,
    variation: str
) -> Dict[str,str]:
    openai_cli, anthropic_cli, xai_cli, gemini_cli = clients
    #img_b64 = encode_image(path)
    img_b64 = encode_and_resize(path, max_dim=256)
    if variation == "With Monk Scale" or variation == "No Monk Name":
        scale_path = Path("/Users/nicoledundas/Documents/GitHub/LLM_SkinTone/monk_scale.png")
        scale_b64 = encode_and_resize(scale_path, max_dim=256)
    else:
        scale_b64 = None

    # responses[model][variation][ctx]
    responses = {m: {variation: {}} for m in ACTIVE_MODELS}

    for ctx_name, ctx_extra in CONTEXT_CONDITIONS:
        sys_inst = f"{CORE_TASK}\n\n{ctx_extra}"
        for model_key, (model_id, handler) in ACTIVE_MODELS.items():
            cli = {
                "GPT": openai_cli,
                "Claude": anthropic_cli,
                "Llama": openai_cli,
                "Grok": xai_cli,
                "Gemini": gemini_cli
            }[model_key]

            # buffer for claude 
            if model_key == "Claude":
                time.sleep(5.0)   # pause 5 seconds between Claude calls

            try:
                out = handler(cli, model_id, sys_inst, prompt, img_b64, scale_b64)
            except Exception as e:
                logging.error(f"{model_key} error on {name}/{ctx_name}: {e}")
                out = "ERROR"
            responses[model_key][variation][ctx_name] = out

    row = {"image_name": name, "image_path": str(path)}
    row.update(flatten_responses(responses))
    return row

# ----------------------------------------
# MAIN
# ----------------------------------------
def main():
    keys    = load_config()
    clients = init_clients(keys)
    
    # UPDATE PATH HERE AS NEEDED!!! 
    image_dir = Path("/Users/nicoledundas/Desktop/LLM_Reflex_Images_MultiLLMp2/")
    #image_dir = Path("/Users/nicoledundas/Desktop/LLM_Reflex_TEST/")

    images    = list_images(image_dir)
    if not images:
        logging.warning("No images found.")
        return

    for var_name, prompt_text in PROMPT_VARIATIONS.items():
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fp = image_dir / f"{var_name.upper()}_results_{ts}_part2.csv"

        # build CSV columns
        fieldnames = ["image_name","image_path"] + [
            f"{m}_{var_name}_{ctx}"
            for m in ACTIVE_MODELS
            for ctx,_ in CONTEXT_CONDITIONS
        ]

        with out_fp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, path in images:
                logging.info(f"[{var_name}] {name}")
                row = process_image(name, path, prompt_text, clients, var_name)
                writer.writerow(row)
                # Add a delay between requests to avoid rate limits
                time.sleep(60.0) # 60 seconds delay

        logging.info(f"Saved results for '{var_name}' → {out_fp}")

if __name__ == "__main__":
    main()
